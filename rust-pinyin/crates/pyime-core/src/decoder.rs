//! Lattice Viterbi / beam-search decoder.
//!
//! Strategy:
//!   1. Split the *original* input into chunks: maximal ASCII-letter runs (decoded as
//!      pinyin/English) and non-letter runs (CJK / punctuation, passed through literally).
//!   2. Each latin run is decoded into an N-best list of partial candidates via a left-to-right
//!      beam search over the syllable lattice, using word bigram transition costs. Latin runs
//!      that do not segment into pinyin (or that match `english.fst`) also get an English edge.
//!   3. Chunk N-best lists are combined sequentially (best-first cartesian, bounded) into whole
//!      input candidates. `kind` is Chinese / English / Mixed depending on the chunks used.

use crate::consts::ENGLISH_PEN;
use crate::lexicon::WordMatch;
use crate::segment::{self, Edge, Normalized};
use crate::{Candidate, CandidateKind, Engine, EngineConfig, Segment};
use rustc_hash::FxHashMap;

const LITERAL_PASS_COST: i32 = 50; // tiny cost for literal CJK passthrough segments

/// A partial decoded result over one chunk (or the combined whole).
#[derive(Clone)]
struct Partial {
    text: String,
    score: i32,
    segments: Vec<Segment>,
    last_word_id: Option<u32>,
    first_word_id: Option<u32>,
    kind: CandidateKind,
}

pub fn decode(engine: &Engine, input: &str, cfg: &EngineConfig) -> Vec<Candidate> {
    decode_inner(engine, input, cfg, false)
}

pub fn predict(engine: &Engine, input: &str, cfg: &EngineConfig) -> Vec<Candidate> {
    decode_inner(engine, input, cfg, true)
}

fn decode_inner(engine: &Engine, input: &str, cfg: &EngineConfig, predict: bool) -> Vec<Candidate> {
    if input.is_empty() {
        return Vec::new();
    }

    // 1. chunk by latin vs non-latin over the ORIGINAL input bytes.
    let chunks = chunk_input(input);

    // 2. decode each chunk into an N-best Partial list.
    let mut chunk_lists: Vec<Vec<Partial>> = Vec::new();
    for ch in &chunks {
        let list = match ch {
            Chunk::Latin { text, byte_start } => {
                decode_latin(engine, text, *byte_start, cfg, predict)
            }
            Chunk::Literal { text, byte_start } => {
                vec![literal_partial(text, *byte_start)]
            }
        };
        if list.is_empty() {
            // ensure progress: pass the raw text through literally
            let (text, bs) = match ch {
                Chunk::Latin { text, byte_start } | Chunk::Literal { text, byte_start } => {
                    (text.clone(), *byte_start)
                }
            };
            chunk_lists.push(vec![literal_partial(&text, bs)]);
        } else {
            chunk_lists.push(list);
        }
    }

    // 3. combine sequentially (bounded best-first).
    let combined = combine(engine, chunk_lists, cfg);

    // 4. to Candidate, sorted, truncated.
    let mut cands: Vec<Candidate> = combined
        .into_iter()
        .map(|p| Candidate {
            text: p.text,
            score: p.score as f32,
            segments: p.segments,
            kind: p.kind,
        })
        .collect();
    cands.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
    cands.truncate(cfg.max_candidates);
    cands
}

enum Chunk {
    Latin { text: String, byte_start: usize },
    Literal { text: String, byte_start: usize },
}

fn chunk_input(input: &str) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    let mut cur_is_latin = false;
    let mut cur_start = 0usize;

    let flush = |chunks: &mut Vec<Chunk>, cur: &mut String, is_latin: bool, start: usize| {
        if cur.is_empty() {
            return;
        }
        let text = std::mem::take(cur);
        if is_latin {
            chunks.push(Chunk::Latin { text, byte_start: start });
        } else {
            chunks.push(Chunk::Literal { text, byte_start: start });
        }
    };

    for (b, c) in input.char_indices() {
        // latin = ascii letters, apostrophe, digits, space within a pinyin region.
        let is_latin = c.is_ascii_alphabetic() || c == '\'' || c.is_ascii_digit();
        if cur.is_empty() {
            cur_is_latin = is_latin;
            cur_start = b;
            cur.push(c);
        } else if is_latin == cur_is_latin {
            cur.push(c);
        } else {
            flush(&mut chunks, &mut cur, cur_is_latin, cur_start);
            cur_is_latin = is_latin;
            cur_start = b;
            cur.push(c);
        }
    }
    flush(&mut chunks, &mut cur, cur_is_latin, cur_start);
    chunks
}

fn literal_partial(text: &str, byte_start: usize) -> Partial {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        // whitespace-only literal: zero-width segment, no output
        return Partial {
            text: String::new(),
            score: 0,
            segments: Vec::new(),
            last_word_id: None,
            first_word_id: None,
            kind: CandidateKind::Chinese,
        };
    }
    Partial {
        text: trimmed.to_string(),
        score: LITERAL_PASS_COST,
        segments: vec![Segment {
            text: trimmed.to_string(),
            reading: String::new(),
            input_span: (byte_start, byte_start + text.len()),
        }],
        last_word_id: None,
        first_word_id: None,
        kind: CandidateKind::Chinese,
    }
}

/// Decode a single latin run.
fn decode_latin(
    engine: &Engine,
    text: &str,
    byte_start: usize,
    cfg: &EngineConfig,
    predict: bool,
) -> Vec<Partial> {
    let norm = segment::normalize(text);
    if norm.letters.is_empty() {
        return Vec::new();
    }
    let lattice = segment::build_lattice(&norm, cfg);

    // Pinyin beam search.
    let mut results = beam_search(engine, &norm, &lattice, byte_start, cfg, predict);

    // English passthrough for the whole run (and per-run if it matches english vocab).
    if cfg.enable_english {
        let lower = norm.letters.clone();
        let is_eng = engine.lexicon.is_english(&lower);
        // Cost: english penalty scaled by letters; if it's in vocab, cheaper.
        let base = ENGLISH_PEN;
        let eng_cost = if is_eng {
            base
        } else {
            base + 1500 + (lower.len() as i32) * 200
        };
        // reconstruct original-case surface from text (strip separators)
        let surface: String = text
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect();
        let span = (byte_start, byte_start + text.len());
        results.push(Partial {
            text: surface.clone(),
            score: eng_cost,
            segments: vec![Segment {
                text: surface,
                reading: String::new(),
                input_span: span,
            }],
            last_word_id: None,
            first_word_id: None,
            kind: CandidateKind::English,
        });
    }

    // dedup by text keeping best, sort, truncate to beam width.
    dedup_best(&mut results);
    results.sort_by_key(|p| p.score);
    results.truncate(cfg.max_candidates.max(cfg.beam_width));
    results
}

/// Beam search state keyed by (position, last_word_id). We keep best partials per position.
fn beam_search(
    engine: &Engine,
    norm: &Normalized,
    lattice: &[Vec<Edge>],
    byte_start: usize,
    cfg: &EngineConfig,
    predict: bool,
) -> Vec<Partial> {
    let n = norm.letters.len();
    // beams[pos] = list of partials whose consumed input ends exactly at letter `pos`.
    let mut beams: Vec<Vec<Partial>> = vec![Vec::new(); n + 1];
    beams[0].push(Partial {
        text: String::new(),
        score: 0,
        segments: Vec::new(),
        last_word_id: None,
        first_word_id: None,
        kind: CandidateKind::Chinese,
    });

    // Precompute word matches starting at each position.
    let mut matches_at: Vec<Vec<WordMatch>> = Vec::with_capacity(n);
    for start in 0..n {
        matches_at.push(engine.lexicon.match_from(lattice, start));
    }

    for pos in 0..n {
        if beams[pos].is_empty() {
            continue;
        }
        // prune this beam frontier
        prune(&mut beams[pos], cfg.beam_width);
        let frontier = beams[pos].clone();

        for wm in &matches_at[pos] {
            // choose the best (cheapest) word from this reading's postings; also expand a few.
            let mut words = wm.words.clone();
            words.sort_by_key(|(_, c)| *c);
            let take = words.len().min(4);
            for &(word_id, reading_cost) in &words[..take] {
                let surface = match engine.lexicon.surface(word_id) {
                    Some(s) => s,
                    None => continue,
                };
                for prev in &frontier {
                    let trans = engine.lm.transition_cost(prev.last_word_id, word_id) as i32;
                    let add = reading_cost as i32 + wm.edit_cost + trans;
                    let new_score = prev.score + add;
                    let mut segs = prev.segments.clone();
                    let span_start = byte_start + norm.orig_byte[wm.start];
                    let span_end = byte_start + norm.orig_end[wm.end - 1];
                    // reconstruct reading text from the lattice path is non-trivial; store the
                    // consumed input slice as reading approximation.
                    let reading: String =
                        norm.letters[wm.start..wm.end].to_string();
                    segs.push(Segment {
                        text: surface.clone(),
                        reading,
                        input_span: (span_start, span_end),
                    });
                    let cand = Partial {
                        text: format!("{}{}", prev.text, surface),
                        score: new_score,
                        segments: segs,
                        last_word_id: Some(word_id),
                        first_word_id: prev.first_word_id.or(Some(word_id)),
                        kind: CandidateKind::Chinese,
                    };
                    beams[wm.end].push(cand);
                }
            }
        }
    }

    // Collect full-coverage results (or prefix results in predict mode).
    let mut out = Vec::new();
    if predict {
        // any partial that has consumed at least one syllable is a completion candidate;
        // prefer longer coverage with a mild bonus.
        for pos in 1..=n {
            for p in &beams[pos] {
                if p.segments.is_empty() {
                    continue;
                }
                let coverage_bonus = (n - pos) as i32 * 100; // penalize leaving input uncovered
                let mut q = p.clone();
                q.score += coverage_bonus;
                out.push(q);
            }
        }
    } else {
        out.extend(beams[n].iter().cloned());
    }
    out
}

fn prune(beam: &mut Vec<Partial>, width: usize) {
    if beam.len() <= width {
        return;
    }
    beam.sort_by_key(|p| p.score);
    beam.truncate(width);
}

fn dedup_best(list: &mut Vec<Partial>) {
    let mut best: FxHashMap<String, usize> = FxHashMap::default();
    let mut keep = vec![true; list.len()];
    for (i, p) in list.iter().enumerate() {
        match best.get(&p.text) {
            Some(&j) => {
                if list[j].score <= p.score {
                    keep[i] = false;
                } else {
                    keep[j] = false;
                    best.insert(p.text.clone(), i);
                }
            }
            None => {
                best.insert(p.text.clone(), i);
            }
        }
    }
    let mut idx = 0;
    list.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

/// Combine chunk N-best lists sequentially. Bounded best-first product.
fn combine(engine: &Engine, lists: Vec<Vec<Partial>>, cfg: &EngineConfig) -> Vec<Partial> {
    let cap = (cfg.max_candidates * 3).max(cfg.beam_width);
    let mut acc: Vec<Partial> = vec![Partial {
        text: String::new(),
        score: 0,
        segments: Vec::new(),
        last_word_id: None,
        first_word_id: None,
        kind: CandidateKind::Chinese,
    }];

    for list in lists {
        if list.is_empty() {
            continue;
        }
        let mut next: Vec<Partial> = Vec::new();
        for a in &acc {
            for b in &list {
                // cross-chunk bigram transition between a's last word and b's first word.
                let trans = match (a.last_word_id, b.first_word_id) {
                    (Some(p), Some(first)) => engine.lm.transition_cost(Some(p), first) as i32,
                    _ => 0,
                };
                let kind = merge_kind(a.kind, b.kind, a.segments.is_empty(), b.segments.is_empty());
                let mut segs = a.segments.clone();
                segs.extend(b.segments.iter().cloned());
                next.push(Partial {
                    text: format!("{}{}", a.text, b.text),
                    score: a.score + b.score + trans,
                    segments: segs,
                    last_word_id: b.last_word_id.or(a.last_word_id),
                    first_word_id: a.first_word_id.or(b.first_word_id),
                    kind,
                });
            }
        }
        next.sort_by_key(|p| p.score);
        next.truncate(cap);
        acc = next;
    }
    acc
}

fn merge_kind(a: CandidateKind, b: CandidateKind, a_empty: bool, b_empty: bool) -> CandidateKind {
    if a_empty {
        return b;
    }
    if b_empty {
        return a;
    }
    if a == b {
        a
    } else {
        CandidateKind::Mixed
    }
}
