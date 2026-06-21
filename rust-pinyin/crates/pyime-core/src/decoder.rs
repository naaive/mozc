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
/// Penalty for a plain out-of-vocabulary latin run that is not a real English word, does not fully
/// segment into pinyin, and is neither abbreviation-shaped nor `pinyin+english`-shaped (e.g.
/// `github`, `keyboard`, `linux`). Kept moderate so these stay as a clean whole-run literal; the
/// mixed/abbrev shapes get their own, larger penalties so the lattice parse wins for them instead.
const ENGLISH_OOV_PEN: i32 = 1500;
/// Penalty added to the whole-run literal when the run is recognizably `clean_pinyin_prefix +
/// trailing_fst_english_word` (e.g. `shoubulesearch`, `shuiluanpenhello`). In that case the run is
/// almost certainly a Chinese+English mix and the lattice's interleaved parse should win, so the
/// opaque whole-run literal is pushed down. Plain OOV latin tokens (github, keyboard, linux) do not
/// match this shape and keep their cheap literal.
const ENGLISH_MIXY_PEN: i32 = 9000;
/// Per-letter component of the mixy penalty so the whole-run literal stays below the (length-
/// scaled) cost of the interleaved pinyin+english parse even for long mixed runs.
const ENGLISH_MIXY_PER_CHAR: i32 = 700;

/// Maximum length of an all-initials run still treated as a pinyin abbreviation (`sbl`, `yghq`,
/// `hlfx`, `ssyb`, ...). Beyond this it is more likely a real (English) token.
const ABBREV_MAX_LEN: usize = 8;
/// Penalty added to the whole-run English literal when the run is abbreviation-shaped, so the
/// abbreviation expansions (e.g. `sbl` → 受不了) win the top ranks instead of the opaque literal.
/// Scales per initial because each extra abbreviated syllable adds a full word (and bigram) to the
/// Chinese expansion's cost; a flat penalty would let the literal resurface for 4+ letter abbrevs.
const ABBREV_LITERAL_PEN: i32 = 6000;
const ABBREV_LITERAL_PER_CHAR: i32 = 3500;

/// Flat per-Chinese-word penalty added to every dictionary word edge when decoding abbreviation-
/// shaped input. It makes a path's cost grow with its WORD COUNT, so covering the abbreviation with
/// fewer, longer words (北京 vs 不+部, 我们 vs 我+们, 受不了 vs 受+不+了) wins. Tuned so a real
/// multi-syllable word beats the equivalent single-char chain without inverting genuine phrase
/// boundaries (a true 2-word abbreviation still decodes as 2 words). Abbrev input only.
const ABBREV_WORD_PER_EDGE: i32 = 1200;
/// Extra penalty on a *single-hanzi* word edge that covers a single initial, applied only for
/// abbreviation-shaped input. The user typing `bj` almost never wants the bare character 不; this
/// demotes such 1-char-per-initial readings beneath any genuine multi-character word covering the
/// same initials, while leaving normal full-pinyin single-char answers (我/的) untouched (they are
/// not abbrev-shaped, so this never fires for them).
const ABBREV_SINGLE_CHAR_PEN: i32 = 1200;
/// Per-extra-initial reward for a multi-character dictionary word matched by its initials, so a word
/// covering MORE of the abbreviation (受不了 = 3 initials) outranks a shorter word covering only a
/// prefix (首播 = 2 initials). Applied on top of the per-initial ABBR_PEN refund. Abbrev input only.
const ABBREV_MULTISYL_BONUS: i32 = 1500;

/// Minimum length of an in-lattice English sub-span edge. Short 1-2 letter "words" in english.fst
/// (a, i, of, ...) would over-fire and interfere with pinyin segmentation, so we require ≥3.
const ENGLISH_MIN_LEN: usize = 4;
/// Base cost of an in-lattice English edge. Tuned so a genuine embedded English word (e.g.
/// `search`, `browser`) competes with — and usually beats — reading those letters as junk pinyin,
/// while a single English edge spanning a whole clean-pinyin run stays more expensive than the
/// Chinese reading (handled by the per-char term + the whole-run passthrough penalty band).
const ENGLISH_EDGE_BASE: i32 = 1800;
/// Per-letter cost of an in-lattice English edge (keeps long latin runs from preferring one big
/// English edge over their pinyin reading).
const ENGLISH_EDGE_PER_CHAR: i32 = 110;
/// Surcharge for an English edge that does NOT reach the end of the latin run. The canonical mixed
/// pattern is `pinyin_prefix + english_suffix` (the English word terminates the run), so an English
/// edge embedded in the middle — which would leave a forced-junk pinyin tail like `gith` + 不 in an
/// OOV latin word — is made costly. This keeps OOV-but-latin words (github, message) as a clean
/// whole-run literal while still allowing a true trailing English word to split off.
const ENGLISH_EDGE_NONTERMINAL_PEN: i32 = 4000;

/// A partial decoded result over one chunk (or the combined whole).
#[derive(Clone)]
struct Partial {
    text: String,
    score: i32,
    segments: Vec<Segment>,
    last_word_id: Option<u32>,
    first_word_id: Option<u32>,
    kind: CandidateKind,
    /// Word-id sequence of this path, in surface order, used by the 4-gram rescoring pass. Each
    /// entry is the dictionary word id of a Chinese edge, or `lm::SENTENCE_START` (== u32::MAX) for
    /// an English edge / literal passthrough / chunk boundary that carries no LM identity (a history
    /// reset). The rescoring walk treats a `SENTENCE_START` entry as a context break, exactly as the
    /// trigram beam does. Empty when the 4-gram model is absent (the chain is then never built).
    word_chain: Vec<u32>,
    /// Sum of the *trigram* transition costs the beam (and `combine`) added along this path. The
    /// rescoring pass replaces this with the 4-gram transition sum: `new_score = score -
    /// trigram_trans_sum + fourgram_trans_sum`, so every NON-LM cost (reading/edit/fuzzy/abbrev/
    /// english penalties, bonuses, literal demotion) is preserved byte-for-byte. Only meaningful
    /// when `word_chain` is populated (4-gram model present).
    trigram_trans_sum: i32,
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
    let mut combined = combine(engine, chunk_lists, cfg);

    // 3b. 4-gram N-best RESCORING (convert only). When `fourgram.fst` is loaded, recompute the LM
    // portion of each retained candidate's TOTAL cost with the 4-gram model (4→3→2→1 stupid-backoff)
    // and re-sort. Inert (no-op) when the 4-gram model is absent, so there is no regression without
    // the file. This only REORDERS the existing N-best — non-LM costs are untouched — so latency
    // stays flat (a few hundred fst lookups). MUST run BEFORE the literal demotion below, so the
    // literal is demoted relative to the FINAL (4-gram-adjusted) Chinese score; otherwise a 4-gram
    // that raises the best Chinese path's cost can let the already-demoted literal flip back to #1.
    if !predict && engine.lm.has_fourgram() {
        rescore_fourgram(engine, &mut combined);
    }

    // 3c. Whole-input literal/English passthrough demotion. When a Chinese reading covers the ENTIRE
    // input (no leftover latin) AND is high quality, the opaque whole-input literal must not rank #1.
    // This generalizes the clean-pinyin `fully_segments` demotion to typo/fuzzy/abbrev-corrected
    // inputs, which do not segment as exact pinyin but still have a full Chinese reading.
    if !predict {
        demote_full_input_literal(input, &mut combined);
    }

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

/// Per-input-letter cost ceiling for a full-coverage Chinese reading to be considered "high quality"
/// enough to demote the whole-input literal beneath it. A genuine typo/fuzzy/abbrev-corrected pinyin
/// sentence reads many input letters into each word, so its total path cost per input letter stays
/// low (typo examples observed ~660–1240/letter). A real English / junk OOV token (`hello`, `linux`,
/// `github`, `asdf`) has NO cheap full Chinese reading — forcing one costs ~2000–3300/letter — so it
/// stays above this ceiling and the literal correctly keeps the top rank. Tuned empirically on
/// `data/gold*.jsonl`; the gap between the two regimes is wide (~1240 vs ~2018), so 1500 is robust.
const FULL_COVER_CN_PER_LETTER_MAX: i32 = 1500;

/// Demote the whole-input literal/English passthrough below the best *full-coverage* Chinese
/// candidate, when such a candidate exists and is high quality.
///
/// The whole-input literal is the `English`-kind candidate whose surface is exactly the input's latin
/// letters (the opaque passthrough produced in `decode_latin`). A "full-coverage Chinese candidate"
/// is one whose output contains NO ascii letters — i.e. the entire input was read as Chinese with no
/// leftover latin fragment (this distinguishes `我只能说` from mixed leftovers like `gith不` /
/// `可以board`, which keep latin and so do NOT trigger demotion → real English passthrough is safe).
///
/// The English-vs-typo distinguishing signal is the best full Chinese reading's cost PER INPUT LETTER
/// (see `FULL_COVER_CN_PER_LETTER_MAX`): typo'd pinyin has a cheap full reading, real English does not.
/// On a match we raise the literal's score to sit just above the best full Chinese candidate, so the
/// corrected sentence wins #1 while the literal stays present in the list (just demoted).
fn demote_full_input_literal(input: &str, combined: &mut [Partial]) {
    // Number of latin letters in the input (what a full Chinese reading must cover).
    let input_letters: String = input.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    let n_letters = input_letters.len() as i32;
    if n_letters == 0 {
        return;
    }
    let lower_letters = input_letters.to_ascii_lowercase();

    // Find the best (cheapest) full-coverage Chinese candidate: no ascii letters in its output.
    let mut best_cn: Option<i32> = None;
    for p in combined.iter() {
        if p.kind == CandidateKind::Chinese && !p.text.chars().any(|c| c.is_ascii_alphabetic()) {
            best_cn = Some(best_cn.map_or(p.score, |b| b.min(p.score)));
        }
    }
    let Some(best_cn) = best_cn else { return };

    // Quality gate: the full Chinese reading must be cheap per input letter (a real corrected pinyin
    // sentence), not a forced reading of an English/junk token.
    if best_cn > FULL_COVER_CN_PER_LETTER_MAX * n_letters {
        return;
    }

    // Raise the whole-input literal just above the best full Chinese candidate (keep it in the list).
    for p in combined.iter_mut() {
        if p.kind == CandidateKind::English && p.text.eq_ignore_ascii_case(&lower_letters) {
            // Only ever raise (never lower) the literal's cost.
            let demoted = best_cn + 1;
            if p.score < demoted {
                p.score = demoted;
            }
        }
    }
}

/// 4-gram N-best rescoring pass. For each retained whole-input candidate, replace the trigram LM
/// transition sum the beam accumulated (`trigram_trans_sum`) with the 4-gram transition sum computed
/// by walking the candidate's `word_chain` with `transition_cost4` (4→3→2→1 stupid-backoff). All
/// NON-LM costs (reading/edit/fuzzy/abbrev/english penalties, abbrev word-length boosts, the literal
/// demotion) are preserved EXACTLY: we mutate only the score, by `-trigram_trans_sum + fourgram_sum`.
///
/// The `word_chain` carries `lm::SENTENCE_START` (== u32::MAX) for English edges / literal
/// passthroughs / chunk boundaries that reset the language-model history; the walk treats a sentinel
/// as a context break (and never charges a transition INTO it), mirroring the trigram beam, so the
/// 4-gram cost is computed on the same contexts the trigram cost was.
fn rescore_fourgram(engine: &Engine, combined: &mut [Partial]) {
    use crate::lm::SENTENCE_START;
    for p in combined.iter_mut() {
        // Candidates whose chain has < 2 real words cannot have any 4-gram context that differs from
        // the trigram (transition_cost4 falls straight through to transition_cost3 on short history),
        // so rescoring them is a strict no-op — skip the work.
        if p.word_chain.len() < 2 {
            continue;
        }
        let mut fourgram_sum: i32 = 0;
        // History window: the previous three (real-or-sentinel) words in surface order.
        let mut w0 = SENTENCE_START;
        let mut w1 = SENTENCE_START;
        let mut w2 = SENTENCE_START;
        for &w3 in &p.word_chain {
            if w3 == SENTENCE_START {
                // History reset (English / literal / boundary): no transition charged into it.
                w0 = SENTENCE_START;
                w1 = SENTENCE_START;
                w2 = SENTENCE_START;
                continue;
            }
            let unigram_w3 = engine.lexicon.unigram_cost(w3).unwrap_or(0) as u32;
            fourgram_sum += engine.lm.transition_cost4(w0, w1, w2, w3, unigram_w3);
            // Slide the window forward.
            w0 = w1;
            w1 = w2;
            w2 = w3;
        }
        // Swap the LM portion: total = old_total - trigram_sum + fourgram_sum.
        p.score = p.score - p.trigram_trans_sum + fourgram_sum;
        // Keep the bookkeeping consistent if anything reads it again (idempotent re-rescoring).
        p.trigram_trans_sum = fourgram_sum;
    }
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
            word_chain: Vec::new(),
            trigram_trans_sum: 0,
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
        // Literal passthrough carries no LM identity: a single history-reset sentinel so a 4-gram
        // walk treats it as a context break (matches the bigram beam, where literals reset history).
        word_chain: vec![crate::lm::SENTENCE_START],
        trigram_trans_sum: 0,
    }
}

/// True if `letters` splits as `clean_pinyin_prefix + trailing_english_word`, i.e. there is a cut
/// `k` (with a non-empty pinyin prefix) such that `letters[..k]` tiles entirely into canonical
/// pinyin syllables and `letters[k..]` is a complete word in english.fst (length ≥ ENGLISH_MIN_LEN).
/// This is the canonical Chinese+English mixed shape (`shoubule`+`search`, `shuiluanpen`+`hello`);
/// plain OOV latin tokens (github, keyboard, linux) do not match it.
fn is_pinyin_prefix_plus_english(engine: &Engine, letters: &str) -> bool {
    let n = letters.len();
    if n < 2 {
        return false;
    }
    // reachable[i] = letters[..i] tiles into pinyin syllables.
    let bytes = letters.as_bytes();
    let mut reachable = vec![false; n + 1];
    reachable[0] = true;
    for s in 0..n {
        if !reachable[s] {
            continue;
        }
        for (_, len) in crate::syllable::prefix_syllables(&letters[s..]) {
            reachable[s + len] = true;
        }
    }
    let _ = bytes;
    for k in 1..n {
        if !reachable[k] {
            continue;
        }
        if (n - k) < ENGLISH_MIN_LEN {
            continue;
        }
        // The suffix is a (near-)complete English word if matching from k reaches the run end.
        if engine
            .lexicon
            .english_matches_from(letters, k, ENGLISH_MIN_LEN)
            .iter()
            .any(|&e| e == n)
        {
            return true;
        }
    }
    false
}

/// True if a latin run looks like a pure first-initial pinyin abbreviation (`sbl` → 受不了, `yghq`,
/// `hlfx`): it can be tiled *entirely* by consonant-initial abbreviation tokens (b, p, ..., zh/ch/sh)
/// without using any full syllable, and it is short enough to be an abbreviation rather than a word.
/// Such runs should not be passed through as an opaque English literal; the abbreviation expansions
/// (recombined by the bigram LM) should rank at the top instead.
fn is_abbrev_shaped(letters: &str) -> bool {
    let n = letters.len();
    if n < 2 || n > ABBREV_MAX_LEN {
        return false;
    }
    // Greedily tile by initials (prefer 2-char zh/ch/sh). Every position must be an initial.
    let bytes = letters.as_bytes();
    let mut i = 0;
    let mut tokens = 0;
    while i < n {
        let two = if i + 2 <= n { Some(&letters[i..i + 2]) } else { None };
        if let Some(t) = two {
            if crate::syllable::is_initial(t) {
                i += 2;
                tokens += 1;
                continue;
            }
        }
        let one = &letters[i..i + 1];
        if crate::syllable::is_initial(one) {
            i += 1;
            tokens += 1;
            continue;
        }
        return false;
    }
    let _ = bytes;
    // Need at least two abbreviated syllables (a single initial is just a partial syllable).
    tokens >= 2
}

/// Generalized abbreviation-shape detector covering the *initial + full-syllable mix* that pure
/// `is_abbrev_shaped` misses: `kyi` = k(initial) + yi(syllable), `bjing` = b + jing, `nihao` is NOT
/// abbrev (it tiles entirely as full syllables with zero bare initials). Returns `Some(n_initials)`
/// — the number of bare-initial tokens in the chosen tiling — when the run is short and tiles by a
/// mix of {bare initials, full syllables} with at least one bare initial; else `None`.
///
/// The caller MUST additionally gate on `!fully_segments` (clean full pinyin like `wo`, `de`,
/// `nihao`, `xian` reads as syllables and must keep its normal single-/multi-char answers). This
/// detector only decides the *shape*; it is the conjunction (`abbrev-shaped AND not clean pinyin`)
/// that flags genuine 简拼 input.
///
/// Tiling preference: at each position we try a full syllable (longest first) OR a bare initial,
/// via a forward-reachability DP that minimizes the number of tokens (fewest, longest pieces) and,
/// among those, records how many were bare initials. This keeps `kyi` = k + yi (1 initial) rather
/// than k + y + i nonsense, matching how a user reads a mixed abbreviation.
fn abbrev_shape(letters: &str) -> Option<usize> {
    let n = letters.len();
    if n < 2 || n > ABBREV_MAX_LEN {
        return None;
    }
    // dp[i] = Some((tokens, initials)) = best tiling of letters[..i]: minimize tokens, then any.
    let mut dp: Vec<Option<(u32, u32)>> = vec![None; n + 1];
    dp[0] = Some((0, 0));
    for i in 0..n {
        let Some((tok, ini)) = dp[i] else { continue };
        // full syllables leaving position i
        for (_, len) in crate::syllable::prefix_syllables(&letters[i..]) {
            let cand = (tok + 1, ini);
            let slot = &mut dp[i + len];
            if slot.map_or(true, |(t, _)| cand.0 < t) {
                *slot = Some(cand);
            }
        }
        // bare initials leaving position i (1- or 2-char)
        for (_, len) in crate::syllable::prefix_initials(&letters[i..]) {
            let cand = (tok + 1, ini + 1);
            let slot = &mut dp[i + len];
            if slot.map_or(true, |(t, _)| cand.0 < t || (cand.0 == t && cand.1 > slot.unwrap().1)) {
                *slot = Some(cand);
            }
        }
    }
    match dp[n] {
        Some((tokens, initials)) if tokens >= 2 && initials >= 1 => Some(initials as usize),
        _ => None,
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
        // Is the run shaped like `clean_pinyin_prefix + trailing_fst_english_word`? If so it is
        // almost certainly a Chinese+English mix and the interleaved lattice parse should win.
        let mixy = !is_eng && is_pinyin_prefix_plus_english(engine, &lower);
        // Is the run abbreviation-shaped (pure initials `sbl`/`yghq`, OR an initial + full-syllable
        // mix `kyi`/`bjing`)? Then demote the opaque literal so the abbreviation expansions win the
        // top ranks. Gated on NOT being clean full pinyin so real words like `nihao` are unaffected.
        //
        // Short abbreviations (`nh`, `ky`, `wm`, ≤ 3 letters) are also present in english.fst as
        // 2-3 letter tokens, but as IME input they are pinyin abbreviations (你好/可以/我们), so for
        // them abbrev-shape OVERRIDES `is_eng`. For longer runs we keep the `!is_eng` gate: a real
        // English word (`world`=wo+r+l+d, `code`, `hello`, all ≥4 letters and in english.fst) would
        // otherwise be spuriously flagged abbrev-shaped and wrongly demoted. The gold `english`
        // bucket is entirely ≥4-letter real words, so this boundary keeps it at 1.000.
        let short_abbrev = lower.len() <= 3;
        let abbrevy = !segments_clean
            && (short_abbrev || !is_eng)
            && (is_abbrev_shaped(&lower) || abbrev_shape(&lower).is_some());
        let eng_cost = if abbrevy {
            base + ABBREV_LITERAL_PEN + ABBREV_LITERAL_PER_CHAR * (lower.len() as i32)
        } else if is_eng {
            // Real English word: keep cheap. If it also happens to read as pinyin, nudge up a
            // little but stay competitive (real words like `hello` are still wanted top-1).
            base + if segments_clean { 600 } else { 0 }
        } else if segments_clean {
            base + (ENGLISH_FULLY_SEGMENTS_PER_CHAR * (lower.len() as i32))
                .max(ENGLISH_FULLY_SEGMENTS_MIN)
        } else if mixy {
            base + ENGLISH_MIXY_PEN + ENGLISH_MIXY_PER_CHAR * (lower.len() as i32)
        } else if abbrevy {
            base + ABBREV_LITERAL_PEN + ABBREV_LITERAL_PER_CHAR * (lower.len() as i32)
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
            // Whole-run English literal: a single history-reset sentinel (no LM identity).
            word_chain: vec![crate::lm::SENTENCE_START],
            trigram_trans_sum: 0,
        });
    }

    // dedup by text keeping best, sort, truncate to beam width.
    dedup_best(&mut results);
    results.sort_by_key(|p| p.score);
    results.truncate(cfg.max_candidates.max(cfg.beam_width));
    results
}

/// Sentinel `word_id` marking an English (latin-surface) edge inside the word lattice. These edges
/// let a contiguous latin run interleave pinyin syllables with English words (`shoubulesearch` →
/// 受不了 + search), instead of treating the whole run as all-pinyin or all-english.
const ENGLISH_EDGE: u32 = u32::MAX;

/// A flattened word-lattice edge: word `word_id` covers letters `[start, end)` for `cost`.
/// Derived ONCE from the memoized `match_from` results, independent of decode history.
/// When `word_id == ENGLISH_EDGE`, the edge emits the latin surface `letters[start..end]` instead
/// of a dictionary word, and contributes no language-model history.
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
fn build_word_lattice(
    engine: &Engine,
    norm: &Normalized,
    lattice: &[Vec<Edge>],
    n: usize,
    cfg: &EngineConfig,
    ambiguous_short: bool,
    is_abbrev: bool,
) -> Vec<Vec<WordEdge>> {
    // Short, highly-ambiguous runs (abbreviations like `yghq` → 浴缸很浅) need many alternative
    // words per span to be retained — the correct multi-syllable word is frequently NOT the most
    // frequent reading of its initials, so a tight per-span cap drops it and tanks coverage. We
    // widen the word caps only for those (and only when short, so latency stays bounded); clean
    // full pinyin and long runs keep the tight caps that are the dominant perf lever.
    let (words_per_reading, edges_per_start, words_per_span) = if is_abbrev {
        // Pure abbreviations are the most ambiguous and the cheapest to decode (short, ~3 ms p95),
        // so they get the widest caps to maximize coverage of the correct multi-syllable words.
        (28usize, 280usize, 48usize)
    } else if ambiguous_short {
        if n <= 6 { (16usize, 160usize, 28usize) } else { (12, 96, 18) }
    } else {
        (9, 56, 12)
    };

    let mut edges_from: Vec<Vec<WordEdge>> = vec![Vec::new(); n];
    for start in 0..n {
        // match_from is computed ONCE per start position here (no per-path recomputation).
        // Abbreviation-shaped input gets a much larger node budget so deep multi-initial readings
        // are reached; everything else keeps the tight default (the dominant latency lever).
        let matches = if is_abbrev {
            engine.lexicon.match_from_budgeted(
                lattice,
                start,
                crate::lexicon::ABBREV_NODE_BUDGET,
                crate::lexicon::ABBREV_MAX_MATCHES,
            )
        } else {
            engine.lexicon.match_from(lattice, start)
        };
        let mut bucket: Vec<WordEdge> = Vec::new();
        for wm in &matches {
            let mut words = wm.words.clone();
            words.sort_unstable_by_key(|(_, c)| *c);
            let take = words.len().min(words_per_reading);
            // Reward a *whole multi-syllable dictionary word* matched by its initial sequence (e.g.
            // 受不了 from `sbl`, 浴缸 from `yg`): such a word stacks one ABBR_PEN per initial, which
            // otherwise buries it under single-character abbrev recombinations. Refund most of the
            // per-initial penalties beyond the first so a real word reachable by its initials ranks
            // — and survives the per-span cap — alongside the single-char paths.
            let span = wm.end - wm.start;
            let abbrev_word_bonus = if is_abbrev && wm.abbrev && span >= 2 {
                ((span as i32 - 1) * crate::consts::ABBR_PEN * 3) / 4
            } else {
                0
            };
            for &(word_id, reading_cost) in &words[..take] {
                // Commercial 简拼 ranking strongly favors covering the input with FEWER, LONGER
                // dictionary words. We bias the per-edge cost (abbrev input only — gated by the
                // caller on `!clean && abbrev_shape`):
                //   * ABBREV_WORD_PER_EDGE  — a flat penalty per Chinese word, so a path made of
                //     many words pays more in aggregate than one long word covering the same input
                //     (favors 北京 over 不+部, 我们 over 我+们).
                //   * ABBREV_SINGLE_CHAR_PEN — an extra penalty on any *single-hanzi* reading, so a
                //     bare high-frequency character (一/不/部) does not dominate when a real
                //     multi-char reading of the abbreviation exists (covers both abbrev-matched and
                //     typo-corrected single chars like `bj`→不).
                // Both are skipped for English edges (handled later) and for non-abbrev decoding.
                let (word_penalty, single_pen, multisyl_bonus) = if is_abbrev {
                    let nchars = engine
                        .lexicon
                        .surface(word_id)
                        .map(|s| s.chars().count())
                        .unwrap_or(0);
                    // Demote any single-hanzi reading (whether it covers one initial via abbrev, or
                    // the whole short run via a typo-corrected single syllable like `bj`→不): the
                    // user typing an abbreviation wants a word, not a bare frequent character.
                    let sp = if nchars <= 1 { ABBREV_SINGLE_CHAR_PEN } else { 0 };
                    // Extra reward for a multi-character word that covers MORE of the abbreviation:
                    // a single dictionary word spanning the whole 简拼 (受不了 for `sbl`, 中国人 for
                    // `zgr`) should beat shorter words that cover only a prefix (首播 for `sb`). The
                    // reward grows with the number of initials covered beyond the first.
                    let ms = if nchars >= 2 && wm.abbrev {
                        (span as i32 - 1) * ABBREV_MULTISYL_BONUS
                    } else {
                        0
                    };
                    (ABBREV_WORD_PER_EDGE, sp, ms)
                } else {
                    (0, 0, 0)
                };
                bucket.push(WordEdge {
                    start: wm.start,
                    end: wm.end,
                    word_id,
                    cost: reading_cost as i32 + wm.edit_cost - abbrev_word_bonus
                        + word_penalty
                        + single_pen
                        - multisyl_bonus,
                });
            }
        }
        // Cheapest first; then keep only WORDS_PER_SPAN per end position, capped overall.
        bucket.sort_unstable_by(|a, b| a.cost.cmp(&b.cost));
        let mut per_end: FxHashMap<usize, usize> = FxHashMap::default();
        let mut kept: Vec<WordEdge> = Vec::with_capacity(edges_per_start);
        for e in bucket {
            let c = per_end.entry(e.end).or_insert(0);
            if *c >= words_per_span {
                continue;
            }
            *c += 1;
            kept.push(e);
            if kept.len() >= edges_per_start {
                break;
            }
        }

        // English sub-span edges: any prefix of `letters[start..]` that is a real English word
        // (in english.fst) becomes an edge so the lattice can interleave English with pinyin.
        // These are added on top of the dictionary edges (and not subject to the per-end cap) so
        // an embedded English run like `search` in `shoubulesearch` always survives.
        if cfg.enable_english {
            for end in engine.lexicon.english_matches_from(&norm.letters, start, ENGLISH_MIN_LEN) {
                // Skip English edges over spans that also read as clean pinyin: those should be
                // decoded as Chinese (e.g. `hehe` → 呵呵), not diverted to a latin surface. Genuine
                // English suffixes (`search`, `browser`, `model`) don't fully segment, so they keep
                // their edge and the lattice can interleave them with pinyin.
                if segment::span_fully_segments(&norm.letters, start, end) {
                    continue;
                }
                let len = (end - start) as i32;
                let mut cost = ENGLISH_EDGE_BASE + ENGLISH_EDGE_PER_CHAR * len;
                if end != n {
                    cost += ENGLISH_EDGE_NONTERMINAL_PEN;
                }
                kept.push(WordEdge { start, end, word_id: ENGLISH_EDGE, cost });
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

    // Highly-ambiguous SHORT runs that do NOT read as clean full pinyin — i.e. abbreviations
    // (`yghq` → 浴缸很浅) and short fuzzy/typo inputs — benefit from a much wider beam and word
    // lattice, and short runs have ample latency headroom. Clean full pinyin (`nihao`, `zhongguo`)
    // takes the tight, fast path so its p95 stays < 5 ms (asserted by the latency budget test).
    let clean = segment::fully_segments(norm);
    let ambiguous_short = !clean && n <= 9;
    // Abbreviation-shaped input: pure first-initial (`sbl`, `bj`) OR an initial + full-syllable mix
    // (`kyi` = k+yi, `bjing` = b+jing). Only here do we (a) reward whole multi-syllable words
    // reachable by their initials, (b) penalize per word so fewer/longer words win, and (c) demote
    // bare single characters. Gating on `!clean` prevents any of this from leaking into clean full
    // pinyin (`zhongguo` must stay 中国 not 郑洞国; `wo`→我 / `de`→的 keep their single-char answers).
    // A real, ≥4-letter English word (`world`, `code`, `hello`) can spuriously match the abbrev
    // shape (`world` = wo+r+l+d); excluding `is_english` runs longer than the short-abbrev window
    // keeps the literal passthrough winning for them and avoids spending the large abbrev budget.
    let is_abbrev = !clean
        && abbrev_shape(&norm.letters).is_some()
        && !(n > 3 && engine.lexicon.is_english(&norm.letters));
    let word_edges = build_word_lattice(engine, norm, lattice, n, cfg, ambiguous_short, is_abbrev);

    // Effective beam width: the DP cost scales ~ n × width × edges. Short ambiguous runs get a much
    // wider beam (cheap, big recall win); very long runs get a slightly narrower beam to keep the
    // p99/max latency tail comfortably under the budget with negligible recall loss.
    let width = if is_abbrev {
        // Abbreviations have the largest fan-out (each initial → ~30 candidate characters) and the
        // gold phrase is often NOT the locally-cheapest per-character path, so it needs the widest
        // beam to survive pruning. These inputs are short (≤ ABBREV_MAX_LEN) with ample latency
        // headroom, so a large multiplier is affordable.
        if n <= 4 { cfg.beam_width * 8 } else { cfg.beam_width * 5 }
    } else if ambiguous_short {
        if n <= 6 { cfg.beam_width * 6 } else { cfg.beam_width * 3 }
    } else if n > 22 {
        (cfg.beam_width * 3) / 4
    } else {
        cfg.beam_width
    };

    // Arena of DP nodes. Node 0 is the origin (empty prefix at pos 0).
    let mut arena: Vec<VNode> = Vec::with_capacity(n * width.max(1));
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
        // Beam-prune the frontier at this position (cheapest first). The trigram DP keys state on
        // the previous TWO words, so the per-history keep dedups on (w_prev, w_prevprev).
        beam_prune(&mut frontier[pos], &arena, width);
        let frontier_pos = frontier[pos].clone();

        let (off, len) = edge_start[pos];
        for ei in off..off + len {
            let we = &flat[ei];
            // Unigram cost of the target word (needed by the stupid-backoff transition). English
            // edges have no LM identity, so it is never consulted for them.
            let unigram_w3 = if we.word_id == ENGLISH_EDGE {
                0
            } else {
                engine.lexicon.unigram_cost(we.word_id).unwrap_or(0) as u32
            };
            for &prev_idx in &frontier_pos {
                let prev = arena[prev_idx];
                // Trigram history: w_prev = the previous edge's word, w_prevprev = the word on the
                // node BEFORE that (read straight from the arena). Both default to the
                // `SENTENCE_START` (== u32::MAX) sentinel, which `transition_cost3` interprets as
                // "no context": the origin node and English edges both carry u32::MAX, so they
                // correctly reset the language-model history.
                let w_prev = prev.word_id; // u32::MAX at origin / after an English edge
                let w_prevprev = if prev.prev == usize::MAX {
                    crate::lm::SENTENCE_START
                } else {
                    arena[prev.prev].word_id
                };
                // English edges carry no language-model identity: no transition into them.
                // `transition_cost3` is signed (log-ratio over unigram): it can be NEGATIVE when the
                // context makes `we.word_id` more likely than its unigram prior, lowering the path.
                let trans: i32 = if we.word_id == ENGLISH_EDGE {
                    0
                } else {
                    engine
                        .lm
                        .transition_cost3(w_prevprev, w_prev, we.word_id, unigram_w3)
                };
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
    // The word-id chain + trigram transition sum are only needed by the 4-gram rescoring pass, so we
    // build them solely when a 4-gram model is loaded (zero overhead on the trigram-only path).
    let want_chain = engine.lm.has_fourgram();
    let mut out: Vec<Partial> = Vec::new();
    let reconstruct = |mut idx: usize, extra: i32| -> Option<Partial> {
        let mut node = arena[idx];
        if node.prev == usize::MAX {
            return None; // origin only, no words consumed
        }
        let final_score = node.score + extra;
        // last_word_id is the LM history this partial exposes to its right neighbor; an English
        // edge carries no LM identity, so fall through to None for it.
        let last_word_id = if node.word_id == ENGLISH_EDGE { None } else { Some(node.word_id) };
        let mut segs: Vec<Segment> = Vec::new();
        let mut text = String::new();
        let mut first_word_id = None;
        let mut saw_chinese = false;
        let mut saw_english = false;
        // Walk backpointers, collecting edges (reverse order).
        let mut chain: Vec<u32> = Vec::new();
        loop {
            if node.edge == u32::MAX {
                break;
            }
            chain.push(node.edge);
            first_word_id = if node.word_id == ENGLISH_EDGE { None } else { Some(node.word_id) };
            idx = node.prev;
            node = arena[idx];
        }
        chain.reverse();
        for &ei in &chain {
            let we = &flat[ei as usize];
            let span_start = byte_start + norm.orig_byte[we.start];
            let span_end = byte_start + norm.orig_end[we.end - 1];
            let reading = norm.letters[we.start..we.end].to_string();
            if we.word_id == ENGLISH_EDGE {
                // English sub-span: emit the latin surface verbatim.
                let surface = norm.letters[we.start..we.end].to_string();
                saw_english = true;
                text.push_str(&surface);
                segs.push(Segment { text: surface, reading: String::new(), input_span: (span_start, span_end) });
            } else {
                let surface = engine.lexicon.surface(we.word_id).unwrap_or_default();
                saw_chinese = true;
                text.push_str(&surface);
                segs.push(Segment { text: surface, reading, input_span: (span_start, span_end) });
            }
        }
        let kind = match (saw_chinese, saw_english) {
            (true, true) => CandidateKind::Mixed,
            (false, true) => CandidateKind::English,
            _ => CandidateKind::Chinese,
        };
        // Build the word-id chain + recompute the trigram transition sum the beam added along this
        // path, for the 4-gram rescoring pass. Recomputing from the chain reproduces the beam's
        // transition cost EXACTLY (same `transition_cost3` with the same SENTENCE_START sentinels at
        // the origin and after English edges), so subtracting it and re-adding the 4-gram sum changes
        // ONLY the LM portion of the score — all non-LM costs are preserved.
        let (word_chain, trigram_trans_sum) = if want_chain {
            let mut wc: Vec<u32> = Vec::with_capacity(chain.len());
            let mut tg_sum: i32 = 0;
            for &ei in &chain {
                let we = &flat[ei as usize];
                let w3 = if we.word_id == ENGLISH_EDGE {
                    // English edge: no LM identity → history reset (sentinel), no transition.
                    crate::lm::SENTENCE_START
                } else {
                    we.word_id
                };
                if w3 != crate::lm::SENTENCE_START {
                    // History from the chain we've built so far (last two real-or-sentinel words).
                    let w_prev = wc.last().copied().unwrap_or(crate::lm::SENTENCE_START);
                    let w_prevprev = if wc.len() >= 2 {
                        wc[wc.len() - 2]
                    } else {
                        crate::lm::SENTENCE_START
                    };
                    let unigram_w3 = engine.lexicon.unigram_cost(we.word_id).unwrap_or(0) as u32;
                    tg_sum += engine.lm.transition_cost3(w_prevprev, w_prev, w3, unigram_w3);
                }
                wc.push(w3);
            }
            (wc, tg_sum)
        } else {
            (Vec::new(), 0)
        };
        Some(Partial {
            text,
            score: final_score,
            segments: segs,
            last_word_id,
            first_word_id,
            kind,
            word_chain,
            trigram_trans_sum,
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
        // Reconstruct only the cheapest terminal nodes (reconstruction allocates strings, so
        // bounding this is a key latency lever). The cap is generous enough to keep N-best diverse.
        let mut terminals = frontier[n].clone();
        terminals.sort_unstable_by_key(|&i| arena[i].score);
        // Reconstruct a generous N-best. When a 4-gram model is loaded we widen this modestly so the
        // rescoring pass sees more diverse word-id paths (the gold path is sometimes ranked below the
        // trigram top-K but wins after the 4-gram upgrade). Reconstruction allocates strings, so this
        // is bounded — the extra cap is small and only taken on the (already cheaper) 4-gram path.
        let recon_cap = if engine.lm.has_fourgram() {
            (cfg.max_candidates * 4).max(width)
        } else {
            (cfg.max_candidates * 3).max(width)
        };
        terminals.truncate(recon_cap);
        for &idx in &terminals {
            if let Some(p) = reconstruct(idx, 0) {
                out.push(p);
            }
        }
    }
    out
}

/// Beam-prune a frontier of arena node indices to `width`, keeping the cheapest. Also dedups by
/// the trigram DP state `(w_prev, w_prevprev)` keeping the best few per language-model history (so
/// the beam carries diverse states rather than `width` copies of the same 2-word context).
///
/// Keying on the FULL trigram state (both history words) is what makes the trigram beam correct:
/// two nodes ending at the same position with the same last word but DIFFERENT prior words are now
/// distinct DP states (they expand to different trigram costs), so they must not collapse together.
/// The state space is larger than the bigram decoder's, so the per-state keep is kept tight and the
/// overall width cap (applied below) bounds latency.
fn beam_prune(frontier: &mut Vec<usize>, arena: &[VNode], width: usize) {
    /// Keep up to this many distinct DP nodes per (w_prev, w_prevprev) state (rather than
    /// collapsing to one), so alternate language-model histories sharing the same 2-word context
    /// survive into reconstruction. This is the main recall lever for N-best diversity without
    /// inflating the raw beam width. Kept at 3 (vs the bigram decoder's 5) because the trigram
    /// state key is finer-grained — distinct prior words already create distinct buckets — so a
    /// smaller per-state keep preserves the same N-best diversity while holding the beam (and thus
    /// latency) bounded against the larger trigram state space.
    const PER_STATE_KEEP: usize = 3;
    // The w_prevprev of a node is the last word of its parent (origin / English edge → sentinel).
    let prevprev = |idx: usize| -> u32 {
        let p = arena[idx].prev;
        if p == usize::MAX { crate::lm::SENTENCE_START } else { arena[p].word_id }
    };
    if frontier.len() > 1 {
        // keep the cheapest few per (w_prev, w_prevprev) trigram state
        frontier.sort_unstable_by_key(|&i| arena[i].score);
        let mut per_state: FxHashMap<(u32, u32), usize> = FxHashMap::default();
        let mut kept: Vec<usize> = Vec::with_capacity(frontier.len().min(width));
        for &idx in frontier.iter() {
            let key = (arena[idx].word_id, prevprev(idx));
            let c = per_state.entry(key).or_insert(0);
            if *c >= PER_STATE_KEEP {
                continue;
            }
            *c += 1;
            kept.push(idx);
        }
        *frontier = kept;
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
    let want_chain = engine.lm.has_fourgram();
    // Widen the retained product modestly when 4-gram rescoring is active so the gold path is more
    // likely to be in the N-best that gets re-ranked. Bounded so latency stays flat.
    let cap = if want_chain {
        (cfg.max_candidates * 4).max(cfg.beam_width)
    } else {
        (cfg.max_candidates * 3).max(cfg.beam_width)
    };
    let mut acc: Vec<Partial> = vec![Partial {
        text: String::new(),
        score: 0,
        segments: Vec::new(),
        last_word_id: None,
        first_word_id: None,
        kind: CandidateKind::Chinese,
        word_chain: Vec::new(),
        trigram_trans_sum: 0,
    }];

    for list in lists {
        if list.is_empty() {
            continue;
        }
        let mut next: Vec<Partial> = Vec::new();
        for a in &acc {
            for b in &list {
                // cross-chunk bigram transition between a's last word and b's first word.
                let trans: i32 = match (a.last_word_id, b.first_word_id) {
                    (Some(p), Some(first)) => engine.lm.transition_cost(Some(p), first),
                    _ => 0,
                };
                let kind = merge_kind(a.kind, b.kind, a.segments.is_empty(), b.segments.is_empty());
                let mut segs = a.segments.clone();
                segs.extend(b.segments.iter().cloned());
                // Concatenate the per-chunk word chains for the 4-gram rescoring pass, and fold the
                // cross-chunk LM transition the beam just added into the (subtracted-then-replaced)
                // trigram transition sum so the rescoring re-add stays exact.
                let (word_chain, trigram_trans_sum) = if want_chain {
                    let mut wc = a.word_chain.clone();
                    wc.extend_from_slice(&b.word_chain);
                    (wc, a.trigram_trans_sum + b.trigram_trans_sum + trans)
                } else {
                    (Vec::new(), 0)
                };
                next.push(Partial {
                    text: format!("{}{}", a.text, b.text),
                    score: a.score + b.score + trans,
                    segments: segs,
                    last_word_id: b.last_word_id.or(a.last_word_id),
                    first_word_id: a.first_word_id.or(b.first_word_id),
                    kind,
                    word_chain,
                    trigram_trans_sum,
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



