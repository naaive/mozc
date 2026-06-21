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
use crate::segment::{self, Edge, Normalized};
use crate::{Candidate, CandidateKind, Engine, EngineConfig, Segment};
use rustc_hash::FxHashMap;

const LITERAL_PASS_COST: i32 = 50; // tiny cost for literal CJK passthrough segments

/// Per-letter English passthrough penalty (scales with run length so long latin runs that are
/// really pinyin don't win by being "english").
const ENGLISH_PER_CHAR_PEN: i32 = 220;
/// Extra penalty when an out-of-vocabulary latin run *also* fully segments into clean pinyin
/// (e.g. `nihao`, `zhongguo`, `womendoushihaohaizi`): the Chinese reading should win. A Chinese
/// sentence's cost grows with its word count (unigram + bigram-backoff per word), so this penalty
/// is applied *per input letter* — a long clean-pinyin run then always loses to its Chinese
/// reading, while the literal passthrough still survives in the list.
const ENGLISH_FULLY_SEGMENTS_PER_CHAR: i32 = 950;
/// Floor for the fully-segments penalty so very short clean runs still lose to Chinese.
const ENGLISH_FULLY_SEGMENTS_MIN: i32 = 4000;
/// Penalty for an out-of-vocabulary latin run that is *not* a real English word and does not
/// fully segment into pinyin (junk). Lower than the fully-segments case but still a clear band.
const ENGLISH_OOV_PEN: i32 = 1500;

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
        let segments_clean = segment::fully_segments(&norm);
        // Cost model (DESIGN.md): ENGLISH_PEN + PER_CHAR_PEN*run_len, reduced when the token is a
        // real English word (in english.fst) so `github`/`hello` win, and raised when the run also
        // fully segments into clean pinyin so `nihao`→你好 / `zhongguo`→中国 win.
        let base = ENGLISH_PEN + ENGLISH_PER_CHAR_PEN * (lower.len() as i32);
        let eng_cost = if is_eng {
            // Real English word: keep cheap. If it also happens to read as pinyin, nudge up a
            // little but stay competitive (real words like `hello` are still wanted top-1).
            base + if segments_clean { 600 } else { 0 }
        } else if segments_clean {
            base + (ENGLISH_FULLY_SEGMENTS_PER_CHAR * (lower.len() as i32))
                .max(ENGLISH_FULLY_SEGMENTS_MIN)
        } else {
            base + ENGLISH_OOV_PEN
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

/// A flattened word-lattice edge: word `word_id` covers letters `[start, end)` for `cost`.
/// Derived ONCE from the memoized `match_from` results, independent of decode history.
#[derive(Clone)]
struct WordEdge {
    start: usize,
    end: usize,
    word_id: u32,
    /// reading_cost + edit_cost (history-independent part of the edge cost).
    cost: i32,
}

/// A node in the Viterbi DP arena. `prev` indexes back into the arena (usize::MAX = origin).
/// We never copy strings/segments during search — only scores + backpointers. Top-N candidates
/// are reconstructed from backpointers at the end.
#[derive(Clone, Copy)]
struct VNode {
    score: i32,
    word_id: u32,
    edge: u32, // index into the flattened word-edge list (for span/reading reconstruction)
    prev: usize,
}

/// Build the flattened word lattice from memoized `match_from` results. Each reading keeps only
/// the cheapest `WORDS_PER_READING` words (postings are already capped in the lexicon), so the
/// number of word edges per start position is bounded.
fn build_word_lattice(engine: &Engine, lattice: &[Vec<Edge>], n: usize) -> Vec<Vec<WordEdge>> {
    const WORDS_PER_READING: usize = 4;
    /// Cap on word edges kept per start position. The lattice already bounds matches, but a single
    /// start can still yield hundreds of (reading × word) edges; we keep only the cheapest few per
    /// distinct end position so the DP frontier stays small (this is the dominant perf lever).
    const EDGES_PER_START: usize = 24;
    /// Per (start,end) span, keep at most this many cheapest words (different surfaces, same span).
    const WORDS_PER_SPAN: usize = 6;

    let mut edges_from: Vec<Vec<WordEdge>> = vec![Vec::new(); n];
    for start in 0..n {
        // match_from is computed ONCE per start position here (no per-path recomputation).
        let matches = engine.lexicon.match_from(lattice, start);
        let mut bucket: Vec<WordEdge> = Vec::new();
        for wm in &matches {
            let mut words = wm.words.clone();
            words.sort_unstable_by_key(|(_, c)| *c);
            let take = words.len().min(WORDS_PER_READING);
            for &(word_id, reading_cost) in &words[..take] {
                bucket.push(WordEdge {
                    start: wm.start,
                    end: wm.end,
                    word_id,
                    cost: reading_cost as i32 + wm.edit_cost,
                });
            }
        }
        // Cheapest first; then keep only WORDS_PER_SPAN per end position, capped overall.
        bucket.sort_unstable_by(|a, b| a.cost.cmp(&b.cost));
        let mut per_end: FxHashMap<usize, usize> = FxHashMap::default();
        let mut kept: Vec<WordEdge> = Vec::with_capacity(EDGES_PER_START);
        for e in bucket {
            let c = per_end.entry(e.end).or_insert(0);
            if *c >= WORDS_PER_SPAN {
                continue;
            }
            *c += 1;
            kept.push(e);
            if kept.len() >= EDGES_PER_START {
                break;
            }
        }
        edges_from[start] = kept;
    }
    edges_from
}

/// Viterbi / beam DP over the word lattice with state `(position, last_word_id)`.
///
/// `best[pos]` keeps, per `last_word_id`, the cheapest arena node ending at `pos` (so distinct
/// language-model histories survive), beam-pruned to `cfg.beam_width`. No per-path string/segment
/// work happens inside the loop — only score arithmetic and backpointer bookkeeping.
fn beam_search(
    engine: &Engine,
    norm: &Normalized,
    lattice: &[Vec<Edge>],
    byte_start: usize,
    cfg: &EngineConfig,
    predict: bool,
) -> Vec<Partial> {
    let n = norm.letters.len();
    if n == 0 {
        return Vec::new();
    }
    let _dbg = std::env::var("PYIME_DBG").is_ok();
    let _t0 = std::time::Instant::now();
    let word_edges = build_word_lattice(engine, lattice, n);
    if _dbg {
        let ne: usize = word_edges.iter().map(|b| b.len()).sum();
        eprintln!("  build_word_lattice n={n} edges={ne} {:.2}ms", _t0.elapsed().as_secs_f64()*1000.0);
    }

    // Arena of DP nodes. Node 0 is the origin (empty prefix at pos 0).
    let mut arena: Vec<VNode> = Vec::with_capacity(n * cfg.beam_width.max(1));
    arena.push(VNode { score: 0, word_id: u32::MAX, edge: u32::MAX, prev: usize::MAX });

    // frontier[pos] = arena node indices whose consumed input ends exactly at letter `pos`.
    let mut frontier: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    frontier[0].push(0);
    // We need a stable view of word edges by global index for reconstruction.
    // Flatten edges into one vector with absolute indices, indexed per start.
    let mut flat: Vec<WordEdge> = Vec::new();
    let mut edge_start: Vec<(usize, usize)> = vec![(0, 0); n]; // (offset, len) into flat
    for (s, bucket) in word_edges.iter().enumerate() {
        edge_start[s] = (flat.len(), bucket.len());
        flat.extend_from_slice(bucket);
    }

    for pos in 0..n {
        if frontier[pos].is_empty() {
            continue;
        }
        // Beam-prune the frontier at this position (cheapest first).
        beam_prune(&mut frontier[pos], &arena, cfg.beam_width);
        let frontier_pos = frontier[pos].clone();

        let (off, len) = edge_start[pos];
        for ei in off..off + len {
            let we = &flat[ei];
            for &prev_idx in &frontier_pos {
                let prev = arena[prev_idx];
                let prev_word = if prev.word_id == u32::MAX { None } else { Some(prev.word_id) };
                let trans = engine.lm.transition_cost(prev_word, we.word_id) as i32;
                let new_score = prev.score + we.cost + trans;
                let node_idx = arena.len();
                arena.push(VNode {
                    score: new_score,
                    word_id: we.word_id,
                    edge: ei as u32,
                    prev: prev_idx,
                });
                frontier[we.end].push(node_idx);
            }
        }
    }

    // Reconstruct candidate Partials from terminal nodes (full coverage, or any prefix in predict).
    let mut out: Vec<Partial> = Vec::new();
    let reconstruct = |mut idx: usize, extra: i32| -> Option<Partial> {
        let mut node = arena[idx];
        if node.prev == usize::MAX {
            return None; // origin only, no words consumed
        }
        let final_score = node.score + extra;
        let last_word_id = Some(node.word_id);
        let mut segs: Vec<Segment> = Vec::new();
        let mut text = String::new();
        let mut first_word_id = None;
        // Walk backpointers, collecting edges (reverse order).
        let mut chain: Vec<u32> = Vec::new();
        loop {
            if node.edge == u32::MAX {
                break;
            }
            chain.push(node.edge);
            first_word_id = Some(node.word_id);
            idx = node.prev;
            node = arena[idx];
        }
        chain.reverse();
        for &ei in &chain {
            let we = &flat[ei as usize];
            let surface = engine.lexicon.surface(we.word_id).unwrap_or_default();
            let span_start = byte_start + norm.orig_byte[we.start];
            let span_end = byte_start + norm.orig_end[we.end - 1];
            let reading = norm.letters[we.start..we.end].to_string();
            text.push_str(&surface);
            segs.push(Segment { text: surface, reading, input_span: (span_start, span_end) });
        }
        Some(Partial {
            text,
            score: final_score,
            segments: segs,
            last_word_id,
            first_word_id,
            kind: CandidateKind::Chinese,
        })
    };

    if predict {
        for pos in 1..=n {
            let coverage_bonus = (n - pos) as i32 * 100; // penalize leaving input uncovered
            for &idx in &frontier[pos] {
                if let Some(p) = reconstruct(idx, coverage_bonus) {
                    out.push(p);
                }
            }
        }
    } else {
        for &idx in &frontier[n] {
            if let Some(p) = reconstruct(idx, 0) {
                out.push(p);
            }
        }
    }
    out
}

/// Beam-prune a frontier of arena node indices to `width`, keeping the cheapest. Also dedups by
/// `last_word_id` keeping the best per language-model history (so the beam carries diverse states
/// rather than `width` copies of the same word).
fn beam_prune(frontier: &mut Vec<usize>, arena: &[VNode], width: usize) {
    if frontier.len() > 1 {
        // keep cheapest per last_word_id
        let mut best: FxHashMap<u32, usize> = FxHashMap::default();
        for &idx in frontier.iter() {
            let wid = arena[idx].word_id;
            match best.get(&wid) {
                Some(&j) if arena[j].score <= arena[idx].score => {}
                _ => {
                    best.insert(wid, idx);
                }
            }
        }
        frontier.clear();
        frontier.extend(best.into_values());
    }
    if frontier.len() <= width {
        frontier.sort_unstable_by_key(|&i| arena[i].score);
        return;
    }
    frontier.sort_unstable_by_key(|&i| arena[i].score);
    frontier.truncate(width);
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
