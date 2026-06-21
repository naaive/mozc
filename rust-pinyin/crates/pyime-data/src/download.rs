//! Resilient downloads with on-disk caching.
//!
//! `fetch_to_cache` downloads `url` into `corpus_dir/cache_name` (skipping the
//! download if the file already exists), with retries. Special handling:
//!   - If `url` ends in `.zip` but `cache_name` ends in `.csv`/`.txt`, the first
//!     matching member inside the zip is extracted (the upstream sources ship a
//!     zip even when the payload is a single CSV).
//!   - If `url` ends in `.gz`, the body is gunzipped.

use anyhow::{anyhow, Context, Result};
use std::io::Read;
use std::path::{Path, PathBuf};

const RETRIES: usize = 4;

pub fn fetch_to_cache(
    corpus_dir: &Path,
    cache_name: &str,
    url: &str,
    offline: bool,
) -> Result<PathBuf> {
    let dest = corpus_dir.join(cache_name);
    if dest.exists() {
        eprintln!("  cache hit: {}", dest.display());
        return Ok(dest);
    }
    if offline {
        return Err(anyhow!(
            "offline mode but {} not cached (expected from {url})",
            dest.display()
        ));
    }

    let body = download_with_retries(url)?;

    let final_bytes = if url.ends_with(".zip") {
        extract_from_zip(&body, cache_name)
            .with_context(|| format!("extract {cache_name} from zip {url}"))?
    } else if url.ends_with(".gz") {
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(&body[..])
            .read_to_end(&mut out)
            .context("gunzip")?;
        out
    } else {
        body
    };

    std::fs::write(&dest, &final_bytes).with_context(|| format!("write cache {}", dest.display()))?;
    eprintln!("  fetched {} ({} bytes)", dest.display(), final_bytes.len());
    Ok(dest)
}

fn download_with_retries(url: &str) -> Result<Vec<u8>> {
    let mut last_err = None;
    for attempt in 1..=RETRIES {
        match download_once(url) {
            Ok(b) => return Ok(b),
            Err(e) => {
                eprintln!("  download attempt {attempt}/{RETRIES} failed: {e:#}");
                last_err = Some(e);
                std::thread::sleep(std::time::Duration::from_millis(500 * attempt as u64));
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow!("download failed: {url}")))
}

fn download_once(url: &str) -> Result<Vec<u8>> {
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(120))
        .build();
    let resp = agent.get(url).call().with_context(|| format!("GET {url}"))?;
    let mut buf = Vec::new();
    resp.into_reader()
        .read_to_end(&mut buf)
        .context("read body")?;
    if buf.is_empty() {
        return Err(anyhow!("empty body from {url}"));
    }
    Ok(buf)
}

/// Minimal ZIP extractor: find the first stored/deflated member whose name has the
/// same extension as `want` (or just the first member) and return its bytes.
/// Supports STORED (method 0) and DEFLATE (method 8) via the central directory.
fn extract_from_zip(zip: &[u8], want: &str) -> Result<Vec<u8>> {
    let want_ext = Path::new(want)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    // Locate End Of Central Directory record (signature 0x06054b50), scanning back.
    let eocd = find_eocd(zip).ok_or_else(|| anyhow!("zip: no EOCD"))?;
    let cd_offset = read_u32(zip, eocd + 16)? as usize;
    let cd_count = read_u16(zip, eocd + 10)? as usize;

    let mut p = cd_offset;
    let mut chosen: Option<(usize, String)> = None; // (local header offset, name)
    for _ in 0..cd_count {
        if read_u32(zip, p)? != 0x0201_4b50 {
            break;
        }
        let name_len = read_u16(zip, p + 28)? as usize;
        let extra_len = read_u16(zip, p + 30)? as usize;
        let comment_len = read_u16(zip, p + 32)? as usize;
        let local_off = read_u32(zip, p + 42)? as usize;
        let name = String::from_utf8_lossy(&zip[p + 46..p + 46 + name_len]).to_string();
        let ext = Path::new(&name)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        if chosen.is_none() && (ext.eq_ignore_ascii_case(want_ext) || want_ext.is_empty()) {
            chosen = Some((local_off, name.clone()));
        }
        p += 46 + name_len + extra_len + comment_len;
    }
    let (local_off, name) = chosen.ok_or_else(|| anyhow!("zip: no member matching .{want_ext}"))?;

    // Parse local file header at local_off.
    anyhow::ensure!(read_u32(zip, local_off)? == 0x0403_4b50, "zip: bad local header for {name}");
    let method = read_u16(zip, local_off + 8)?;
    let comp_size = read_u32(zip, local_off + 18)? as usize;
    let name_len = read_u16(zip, local_off + 26)? as usize;
    let extra_len = read_u16(zip, local_off + 28)? as usize;
    let data_start = local_off + 30 + name_len + extra_len;
    let data = &zip[data_start..data_start + comp_size];

    match method {
        0 => Ok(data.to_vec()), // STORED
        8 => {
            let mut out = Vec::new();
            flate2::read::DeflateDecoder::new(data)
                .read_to_end(&mut out)
                .context("zip: inflate member")?;
            Ok(out)
        }
        m => Err(anyhow!("zip: unsupported compression method {m}")),
    }
}

fn find_eocd(zip: &[u8]) -> Option<usize> {
    if zip.len() < 22 {
        return None;
    }
    let start = zip.len().saturating_sub(22 + 65536);
    for i in (start..=zip.len() - 22).rev() {
        if zip[i] == 0x50 && zip[i + 1] == 0x4b && zip[i + 2] == 0x05 && zip[i + 3] == 0x06 {
            return Some(i);
        }
    }
    None
}

fn read_u16(b: &[u8], off: usize) -> Result<u16> {
    let s = b.get(off..off + 2).ok_or_else(|| anyhow!("zip: u16 OOB"))?;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}
fn read_u32(b: &[u8], off: usize) -> Result<u32> {
    let s = b.get(off..off + 4).ok_or_else(|| anyhow!("zip: u32 OOB"))?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}
