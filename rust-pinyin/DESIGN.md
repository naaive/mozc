# pyime — A Rust Pinyin IME Engine (Mozc-inspired)

A high-performance Chinese **Pinyin input method ENGINE** (no GUI/shell) exposed via a CLI for
testing. Inspired by Google Mozc's lattice + cost-minimization decoder, but redesigned for
Chinese pinyin with a data-driven **word bigram** language model and an **FST**-based lexicon.

## Scope of input features (MUST all work)
- 全拼  Full pinyin            — `nihao` → 你好
- 简拼  Abbreviated (initials) — `nh`, `bj` → 你好 / 北京 ; mixed `bei'j`
- 模糊音 Fuzzy syllables        — zh↔z, ch↔c, sh↔s, n↔l, f↔h, r↔l, an↔ang, en↔eng, in↔ing, ...
- 纠错  Typo correction        — edit-distance ≤1 on syllables (`wo3` , `nij` , transposition, fat-finger key map)
- 英文  English passthrough     — `hello`, `github` stay as-is, ranked sanely
- 混合  Mixed CN/EN             — `wo用github` → 我用 github
- (Explicitly **NOT** 双拼 / shuangpin)

## Workspace layout
```
rust-pinyin/
  Cargo.toml                 # virtual workspace; members = the 4 crates below
  DESIGN.md                  # this contract
  data/                      # BUILT artifacts (committed if < budget, else regen). Engine reads these.
  corpus/                    # raw downloaded sources (gitignored)
  crates/
    pyime-core/   # syllable table, fuzzy, segmentation lattice, decoder, LM, Engine  (NO network)
    pyime-data/   # data BUILD pipeline (downloads corpora, emits data/*)  +  data LOADERS
    pyime-eval/   # evaluation harness: metrics, gold-set generation, reports
    pyime-cli/    # clap CLI: `interactive`, `convert`, `eval`, `build-data`
```

## Crate ownership (subagents do not edit other crates' src)
- pyime-core  → CORE agent
- pyime-data  → DATA agent
- pyime-eval  → EVAL agent
- pyime-cli   → CLI agent (integration)

## Data format contract (v1)  — produced by pyime-data, consumed by pyime-core
All files live in `data/`. Loaded via mmap where possible. Costs are integers
`cost = round(-LOG_BASE * ln(prob))`, `LOG_BASE = 500` (Mozc-like). Lower = better.

| file            | format                | meaning |
|-----------------|-----------------------|---------|
| `meta.json`     | JSON                  | version, counts, LOG_BASE, sizes |
| `words.bin`     | rkyv `Vec<WordEntry>` | id→{surface:String, unigram_cost:u16, pos:u8} |
| `lexicon.fst`   | `fst::Map`            | key = canonical pinyin syllables joined by `'` (e.g. `ni'hao`); value = u64 = offset into `postings.bin` |
| `postings.bin`  | raw                   | at each offset: `u16 n` then `n × (u32 word_id, u16 cost)` (cost already in WordEntry too; postings cost is the reading-specific unigram cost) |
| `bigram.fst`    | `fst::Map`            | key = 8 bytes BE = (u32 prev_id, u32 id); value = u64 = bigram_cost |
| `english.fst`   | `fst::Set`            | english/letter vocabulary for passthrough ranking (lowercased) |

### v2 additions (commercial-grade upgrade)
| file               | format     | meaning |
|--------------------|------------|---------|
| `trigram.fst`      | `fst::Map` | key = 12 bytes BE = (u32 w1, u32 w2, u32 w3); value = u64 = trigram_cost. **Optional** — engine works without it (bigram-only). |
| `fourgram.fst`     | `fst::Map` | key = 16 bytes BE = (u32 w0,w1,w2,w3) via `fourgram_key`; value = u64 = 4-gram cost (same signed-log-ratio encoding as bigram/trigram). **Optional** — used as an N-best rescoring pass on top of the trigram beam (recompute each candidate's path cost with 4-gram→3→2→1 stupid-backoff, re-sort), so latency stays flat and the beam state does not grow. |
| `word_pinyin.tsv`  | TSV        | `word<TAB>canonical_reading` (syllables joined by `'`), the curated lexicon export used by eval to generate CORRECT-reading gold via longest-match tokenization. |

**Language model (stupid-backoff):** `P(w3|w1,w2)` cost = `trigram.fst[(w1,w2,w3)]` if present,
else `TRIGRAM_BACKOFF + ( bigram.fst[(w2,w3)] if present else BIGRAM_BACKOFF + unigram_cost(w3) )`.
Lexicon readings/weights are sourced primarily from the hand-curated **rime-ice** dictionaries
(correct polyphone readings + frequencies), supplemented by jieba for coverage.

Canonical pinyin syllable = standard Hanyu Pinyin without tone marks (v/ü → `v`).
The decoder performs syllable segmentation + fuzzy/abbrev/correction expansion itself; the
lexicon stores ONLY canonical readings. Matching is done by walking the FST with a custom
`fst::Automaton` (`PinyinAutomaton`) that consumes the user input and accepts canonical keys
reachable under the enabled fuzzy/abbrev/correction rules, reporting the consumed input length.

## Core public API (pyime-core)  — STABLE contract for eval/cli
```rust
pub struct Engine { /* holds loaded data */ }

pub struct EngineConfig {
    pub fuzzy: FuzzySet,        // which fuzzy rules enabled
    pub enable_correction: bool,
    pub correction_max_edits: u8,
    pub enable_english: bool,
    pub max_candidates: usize,  // N-best
    pub beam_width: usize,
}
impl Default for EngineConfig { /* sensible defaults, all features on */ }

pub struct Candidate {
    pub text: String,           // converted output, e.g. "你好"
    pub score: f32,             // total path cost (lower better)
    pub segments: Vec<Segment>, // word boundaries + matched reading
    pub kind: CandidateKind,    // Chinese | English | Mixed
}

impl Engine {
    /// Load built data from a directory.
    pub fn load(data_dir: &std::path::Path) -> anyhow::Result<Engine>;
    /// Convert a raw input buffer into ranked candidates (best first).
    pub fn convert(&self, input: &str, cfg: &EngineConfig) -> Vec<Candidate>;
    /// Prefix prediction / completion for a partial input.
    pub fn predict(&self, input: &str, cfg: &EngineConfig) -> Vec<Candidate>;
}
```

## Decoder design (pyime-core)
1. **Tokenize input** into a char stream; classify ASCII-letter runs vs separators (`'`, space, digits).
2. **Syllable lattice**: for each start index, enumerate candidate syllables via a max-munch
   trie of the 410 canonical syllables, expanded by: fuzzy rules, abbreviation (single initial
   letter = a 1-char "syllable" placeholder that the lexicon automaton treats as initial-match),
   and correction (edit-distance ≤1). Each lattice edge = (start, end, canonical_syllable, edit_cost).
3. **Word lattice**: walk `lexicon.fst` with `PinyinAutomaton` over each syllable-path prefix to
   collect dictionary words covering ranges; edge cost = unigram_cost + edit_cost penalties.
4. **Viterbi / beam search** with word **bigram** transition costs (`bigram.fst`, backoff to
   unigram) to produce N-best whole-sentence candidates. Also surface single-/multi-word
   prefix candidates for prediction.
5. **English / mixed**: ASCII runs that don't segment into pinyin (or match `english.fst`)
   become English edges with their own cost band so mixed sentences rank correctly.

## Evaluation system (pyime-eval) — MUST be comprehensive & quantified
Gold set: built by DATA/EVAL from a held-out sentence corpus, converting each sentence to
pinyin (full + simulated abbrev/fuzzy/typo variants) to create `(input, expected)` pairs,
bucketed by scenario.

Scenarios (buckets): `full`, `abbr`, `fuzzy`, `typo`, `english`, `mixed`, `long_sentence`, `short_word`.

Metrics per bucket and overall:
- **Top-1 accuracy** (exact sentence match)
- **Top-5 / Top-10 inclusion rate** (候选命中率)
- **MRR** (mean reciprocal rank of the gold answer)
- **Char-level accuracy** (best candidate vs gold, normalized edit similarity)
- **CER** (character error rate of top-1)
- **KSPC-ish**: avg input length / output chars (compression)
- **Latency**: p50 / p95 / mean per convert (ms), throughput (conv/s)
- **Memory**: RSS after load; data size on disk
Output: a human-readable table + a `report.json`. `cli eval` prints it. Regenerate easily.

## User dictionary + online adaptation (pyime-core) — v3 addition
Every commercial IME learns from what the user commits. `pyime-core` gains an optional, persistent
**user model** that personalizes ranking online (CPU-trivial, no training).

State (`UserModel`, serialized to a small JSON/bincode file in a user data dir):
- **user unigram counts**: `word → count` (and a monotonic `last_used` tick for recency/LRU).
- **user bigram counts**: `(prev_word, word) → count` for personalized transitions.
- **user phrases**: `pinyin_key → surface` committed as a unit — AUTO-LEARNS new words/phrases not
  in the base lexicon, surfaced as high-priority candidates when the input matches.

API additions (STABLE):
```rust
impl Engine {
    /// Attach a user model, loading prior history from `path` if it exists.
    pub fn with_user_model(self, path: Option<std::path::PathBuf>) -> Engine;
    /// Learn from a committed selection: input buffer → chosen output. Updates counts/recency/phrases.
    pub fn commit(&self, input: &str, chosen: &str);
    /// Persist the user model to its path (no-op if none/unset).
    pub fn save_user(&self) -> anyhow::Result<()>;
}
// EngineConfig gains:  pub user_weight: i32   // strength of personalization (0 = off)
```
The user model is read during `convert`/`predict` (so `commit` uses interior mutability that stays
`Sync` for the rayon eval sweep — e.g. `RwLock`). Scoring blends a **user bonus** (negative cost):
recently/frequently committed words get a capped cost reduction `≈ -user_weight·f(count, recency)`;
a matching user phrase is injected as a candidate with a strong bonus; user bigrams refine the
transition. Bonuses are capped so personalization re-ranks within the N-best without breaking
clean-input correctness.

Evaluation (extends pyime-eval): a **personalization benchmark** — simulate a user session as a
stream of `(input, gold)` where a subset of phrases/words recur or are user-specific; measure
top-1/MRR with the user model OFF vs. ON (online: `commit(gold)` after each item) and report the
lift per scenario. This quantifies the adaptation, per the "全面量化" requirement.

## Performance / size budgets
- Total `data/` + release binary **< 100 MB**. Prefer separate mmap'd data files; if embedding,
  zstd-compress. Quantize bigram aggressively (prune low-count pairs, u16 costs).
- Decode latency target: p95 < 5 ms for typical 6–12 char inputs.
- Use `fst` (mmap), `rkyv`/`memmap2`, `rayon` for eval, `clap` for CLI. Release profile: LTO, `opt-level=3`, `strip=true`, `panic=abort`, `codegen-units=1`.

## Cost/scaling constants (shared)
- `LOG_BASE = 500.0`
- fuzzy edge penalty: +`FUZZY_PEN` (e.g. 1200) per fuzzy substitution
- correction edge penalty: +`TYPO_PEN` (e.g. 2400) per edit
- abbreviation penalty: +`ABBR_PEN` (e.g. 1800) per abbreviated syllable
- english edge cost: tuned so pure-pinyin wins when it segments cleanly
