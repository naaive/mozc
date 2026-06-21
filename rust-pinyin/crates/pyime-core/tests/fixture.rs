//! Self-contained integration tests for pyime-core.
//!
//! The DATA crate that builds the real `data/` directory may not be ready, so these tests build
//! a tiny in-memory dataset (a dozen words) into a temp directory using the `fst` + `rkyv`
//! crates directly — exactly matching the on-disk format contract in DESIGN.md — then load it
//! via `Engine::load` and exercise the full decoder.

use pyime_core::format::{bigram_key, WordEntry};
use pyime_core::{CandidateKind, Engine, EngineConfig, FuzzySet};
use std::path::{Path, PathBuf};

/// Build the fixture `data/` directory and return its path.
fn build_fixture() -> PathBuf {
    // Words: id -> (surface, reading, unigram_cost)
    // readings use canonical pinyin joined by '\''.
    let words: Vec<(&str, &str, u16)> = vec![
        ("你好", "ni'hao", 100), // 0
        ("北京", "bei'jing", 120), // 1
        ("中国", "zhong'guo", 110), // 2
        ("我", "wo", 80),         // 3
        ("用", "yong", 130),      // 4
        ("中文", "zhong'wen", 140), // 5
        ("你", "ni", 150),        // 6
        ("好", "hao", 160),       // 7
        ("中", "zhong", 170),     // 8
        ("国", "guo", 180),       // 9
        ("文", "wen", 190),       // 10
        ("京", "jing", 200),      // 11
    ];

    let entries: Vec<WordEntry> = words
        .iter()
        .map(|(s, _, c)| WordEntry {
            surface: s.to_string(),
            unigram_cost: *c,
            pos: 0,
        })
        .collect();

    // words.bin (rkyv Vec<WordEntry>)
    let words_bytes = rkyv::to_bytes::<_, 4096>(&entries).expect("serialize words");

    // Build reading -> list of (word_id, cost). Multiple words may share a reading.
    use std::collections::BTreeMap;
    let mut readings: BTreeMap<String, Vec<(u32, u16)>> = BTreeMap::new();
    for (id, (_, reading, cost)) in words.iter().enumerate() {
        readings
            .entry((*reading).to_string())
            .or_default()
            .push((id as u32, *cost));
    }

    // postings.bin + lexicon.fst (key=reading -> offset into postings).
    let mut postings: Vec<u8> = Vec::new();
    let mut fst_entries: Vec<(String, u64)> = Vec::new();
    for (reading, list) in &readings {
        let offset = postings.len() as u64;
        postings.extend_from_slice(&(list.len() as u16).to_le_bytes());
        for (id, cost) in list {
            postings.extend_from_slice(&id.to_le_bytes());
            postings.extend_from_slice(&cost.to_le_bytes());
        }
        fst_entries.push((reading.clone(), offset));
    }
    fst_entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut lex_builder = fst::MapBuilder::memory();
    for (k, v) in &fst_entries {
        lex_builder.insert(k.as_bytes(), *v).unwrap();
    }
    let lexicon_fst = lex_builder.into_inner().unwrap();

    // bigram.fst : key = bigram_key(prev,id) BE, value = cost. Keys must be sorted.
    // A few sensible bigrams (lower = more likely).
    let mut bigrams: Vec<(u32, u32, u64)> = vec![
        (3, 4, 200),  // 我 -> 用
        (4, 2, 250),  // 用 -> 中国
        (8, 9, 150),  // 中 -> 国
        (8, 10, 160), // 中 -> 文
        (6, 7, 140),  // 你 -> 好
        (3, 5, 300),  // 我 -> 中文
    ];
    bigrams.sort_by(|a, b| bigram_key(a.0, a.1).cmp(&bigram_key(b.0, b.1)));
    let mut bi_builder = fst::MapBuilder::memory();
    for (p, i, c) in &bigrams {
        bi_builder.insert(bigram_key(*p, *i), *c).unwrap();
    }
    let bigram_fst = bi_builder.into_inner().unwrap();

    // english.fst : a couple of english words (sorted set).
    let mut eng: Vec<&str> = vec!["github", "hello", "google", "rust"];
    eng.sort();
    let mut eng_builder = fst::SetBuilder::memory();
    for w in &eng {
        eng_builder.insert(w.as_bytes()).unwrap();
    }
    let english_fst = eng_builder.into_inner().unwrap();

    // write to a unique temp dir (unique per call so parallel tests don't collide)
    use std::sync::atomic::{AtomicU64, Ordering};
    static CTR: AtomicU64 = AtomicU64::new(0);
    let uniq = CTR.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "pyime_fixture_{}_{}",
        std::process::id(),
        uniq
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write(&dir, "words.bin", &words_bytes);
    write(&dir, "postings.bin", &postings);
    write(&dir, "lexicon.fst", &lexicon_fst);
    write(&dir, "bigram.fst", &bigram_fst);
    write(&dir, "english.fst", &english_fst);

    // meta.json (not strictly required by loader, but part of the contract).
    let meta = format!(
        "{{\"version\":1,\"log_base\":500.0,\"num_words\":{},\"num_readings\":{},\"num_bigrams\":{},\"bytes_total\":0,\"source_notes\":\"fixture\"}}",
        entries.len(),
        readings.len(),
        bigrams.len()
    );
    write(&dir, "meta.json", meta.as_bytes());

    dir
}

fn write(dir: &Path, name: &str, bytes: &[u8]) {
    std::fs::write(dir.join(name), bytes).unwrap();
}

fn engine() -> Engine {
    let dir = build_fixture();
    Engine::load(&dir).expect("load engine")
}

fn top_texts(c: &[pyime_core::Candidate], n: usize) -> Vec<String> {
    c.iter().take(n).map(|x| x.text.clone()).collect()
}

#[test]
fn full_pinyin_nihao() {
    let e = engine();
    let cfg = EngineConfig::default();
    let c = e.convert("nihao", &cfg);
    assert!(!c.is_empty(), "no candidates for nihao");
    assert_eq!(c[0].text, "你好", "top1 = {:?}", top_texts(&c, 5));
    assert_eq!(c[0].kind, CandidateKind::Chinese);
}

#[test]
fn abbrev_bj_includes_beijing() {
    let e = engine();
    let cfg = EngineConfig::default();
    let c = e.convert("bj", &cfg);
    let texts = top_texts(&c, 20);
    assert!(
        texts.iter().any(|t| t == "北京"),
        "bj should include 北京, got {:?}",
        texts
    );
}

#[test]
fn fuzzy_zhongguo_and_zongguo() {
    let e = engine();
    let cfg = EngineConfig::default();

    let c1 = e.convert("zhongguo", &cfg);
    assert_eq!(c1[0].text, "中国", "zhongguo top1 = {:?}", top_texts(&c1, 5));

    let c2 = e.convert("zongguo", &cfg);
    let texts = top_texts(&c2, 10);
    assert!(
        texts.iter().any(|t| t == "中国"),
        "zongguo (fuzzy z->zh) should yield 中国, got {:?}",
        texts
    );
}

#[test]
fn fuzzy_off_disables_zongguo() {
    let e = engine();
    // Disable BOTH fuzzy and correction so neither path can reach 中国 from "zongguo".
    let cfg = EngineConfig {
        fuzzy: FuzzySet::none(),
        enable_correction: false,
        ..Default::default()
    };
    let c = e.convert("zongguo", &cfg);
    let texts = top_texts(&c, 10);
    // Without fuzzy or correction, "zong" is not a path to 中国 (zong is not a fixture reading);
    // 中国 should not surface as top1.
    assert!(
        texts.first().map(|t| t != "中国").unwrap_or(true),
        "with fuzzy+correction off, 中国 should not be top1 for zongguo: {:?}",
        texts
    );
}

#[test]
fn typo_one_edit_finds_word() {
    let e = engine();
    let cfg = EngineConfig::default();
    // "nihai" -> ni + hai; hai is a typo (substitution o->i / fat finger) of hao.
    let c = e.convert("nihai", &cfg);
    let texts = top_texts(&c, 10);
    assert!(
        texts.iter().any(|t| t == "你好"),
        "typo nihai should still find 你好, got {:?}",
        texts
    );
}

#[test]
fn english_passthrough() {
    let e = engine();
    let cfg = EngineConfig::default();
    let c = e.convert("github", &cfg);
    assert!(!c.is_empty());
    let texts = top_texts(&c, 5);
    assert!(
        texts.iter().any(|t| t == "github"),
        "github should pass through, got {:?}",
        texts
    );
    // it should be classified English somewhere in the list
    assert!(c.iter().any(|x| x.kind == CandidateKind::English && x.text == "github"));
}

#[test]
fn mixed_wozhongwen() {
    let e = engine();
    let cfg = EngineConfig::default();
    let c = e.convert("wozhongwen", &cfg);
    let texts = top_texts(&c, 10);
    assert!(
        texts.iter().any(|t| t == "我中文"),
        "wozhongwen should rank 我中文, got {:?}",
        texts
    );
}

#[test]
fn mixed_cn_en() {
    let e = engine();
    let cfg = EngineConfig::default();
    // wo用github : latin 'wo' -> 我, literal '用', latin 'github' -> english
    let c = e.convert("wo用github", &cfg);
    assert!(!c.is_empty());
    let texts = top_texts(&c, 10);
    assert!(
        texts.iter().any(|t| t == "我用github"),
        "mixed should yield 我用github, got {:?}",
        texts
    );
    assert!(
        c.iter().any(|x| x.text == "我用github" && x.kind == CandidateKind::Mixed),
        "mixed candidate should be kind=Mixed"
    );
}

#[test]
fn predict_prefix() {
    let e = engine();
    let cfg = EngineConfig::default();
    // partial "nih" should predict 你好 as a completion.
    let c = e.predict("nih", &cfg);
    let texts = top_texts(&c, 10);
    assert!(
        texts.iter().any(|t| t == "你好" || t == "你"),
        "predict nih should suggest 你/你好, got {:?}",
        texts
    );
}

#[test]
fn perf_long_input() {
    let e = engine();
    let cfg = EngineConfig::default();
    let inputs = ["wozhongwenzhongguobeijing", "nihaozhongguobeijing", "zhongguobeijing"];
    for inp in inputs {
        let t = std::time::Instant::now();
        let mut last = 0u128;
        for _ in 0..50 {
            let c = e.convert(inp, &cfg);
            assert!(!c.is_empty());
            last = c.len() as u128;
        }
        let per = t.elapsed().as_micros() / 50;
        println!("input={:?} len={} ncand={} avg={}us", inp, inp.len(), last, per);
    }
}
