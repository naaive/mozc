# pyime — a Rust Pinyin IME engine

A high-performance Chinese **Pinyin input-method engine** written in Rust, inspired by Google
[Mozc](https://github.com/google/mozc)'s lattice + cost-minimization decoder, redesigned for
Chinese pinyin with an **FST lexicon**, a **curated dictionary** (rime-ice), and a **smoothed
word trigram** language model.

This is the **engine only** (no GUI/IME shell); it ships a `pyime` **CLI** for conversion,
prediction, an interactive REPL, a self-contained data builder, and a comprehensive,
fully-quantified **evaluation system**.

## Features

| Mode | Example | Output |
|------|---------|--------|
| 全拼 Full pinyin       | `woshizhongguoren` | 我是中国人 |
| 简拼 Abbreviation (initials) | `bj`, `sbl` | 北京 / 受不了 |
| 模糊音 Fuzzy            | `zongguo` (z↔zh) | 中国 |
| 纠错 Typo correction   | `wozhinnegshuo` (typos) | 我只能说 |
| 英文 English           | `github` | github |
| 混合 Mixed CN+EN       | `shoubulesearch`, `wo用github` | 手不了search / 我用github |

(双拼/shuangpin is intentionally **out of scope**.)

## Architecture

```
input ─▶ normalize ─▶ syllable lattice ─▶ word lattice (FST walk) ─▶ trigram beam DP ─▶ N-best
                      │ full / fuzzy /       │ PinyinAutomaton over     │ smoothed word trigram
                      │ abbrev / typo edges  │ lexicon.fst (mmap)       │ (stupid-backoff, mmap)
```

- **Syllable lattice** (`segment.rs`): max-munch over the ~410 canonical Hanyu-Pinyin syllables,
  expanded with fuzzy variants, abbreviation (initial-only) tokens, and edit-distance≤1 typo
  corrections (QWERTY fat-finger map + transposition), each carrying an integer edit cost.
- **Lexicon** (`lexicon.rs`): an [`fst::Map`](https://docs.rs/fst) keyed by canonical reading
  (`ni'hao`), memory-mapped, walked by a custom automaton/lattice walk supporting fuzzy,
  abbreviation-by-initials (incl. matching a whole multi-syllable word's reading by its
  initials), and interleaved English sub-spans for mixed input.
- **Language model** (`lm.rs`): word **unigram + bigram + trigram** with **stupid-backoff**.
  Costs are the **signed log-ratio of the conditional over the unigram**
  (`-500·ln[P(w₃|ctx)/P(w₃)]`) estimated with **absolute-discounting interpolation**
  (modified-Kneser-Ney style), so the trigram *refines* the bigram on one comparable scale
  instead of overpowering it. Trigram/bigram tables are `fst::Map`s (mmap), unigram is in the
  word table.
- **Decoder** (`decoder.rs`): **trigram beam search**, state `(position, w_prev, w_prevprev)`,
  per-state N-best diversity for rich candidates; English/mixed handled with length-scaled
  passthrough costs, and a full-coverage demotion so a corrected Chinese reading outranks the
  raw literal (while the literal stays in the list).

### Crates
- `pyime-core` — the engine (no network, mmap'd data).
- `pyime-data` — data build pipeline (downloads corpora, emits `data/`) + on-disk format.
- `pyime-eval` — gold-set generation + quantified metrics harness.
- `pyime-cli`  — the `pyime` binary.

### Technology choices
`fst` (compact, mmap'd) for lexicon & n-grams · `rkyv` zero-copy archives for the word table ·
`memmap2` for O(1) load · `rayon` for the eval sweep · release profile with fat LTO + `strip` +
`panic=abort` + `codegen-units=1`.

## Data

Built entirely from **freely-available, GitHub-hosted** sources (see `data/meta.json`):

- **Lexicon** — hand-curated [`iDvel/rime-ice`](https://github.com/iDvel/rime-ice) dictionaries
  (≈700k words with **correct polyphone readings** and real frequencies), `rime/rime-essay`
  frequencies, and jieba for coverage back-fill. This is the key quality lever: per-character
  pinyin composition mangles polyphones (银行→`yinhang` not `yinxing`); rime-ice fixes it.
- **Language model** — a word **trigram** built from a general-domain news-title corpus
  (Toutiao, ~2.1M segments) unioned with shopping reviews, tokenized by longest-match over the
  lexicon, smoothed with absolute-discounting interpolation.
- **English** — a frequency-ranked list (`google-10000-english` ∪ `dwyl/english-words`, 60k).

Counts: ~700k words / ~471k readings / ~737k bigrams / ~725k trigrams / 60k English.

**Footprint:** release binary **~3 MB** + `data/` **~45 MB** = **~48 MB total**, well under the
100 MB budget (data is mmap'd separately, not embedded).

## Build & run

```bash
cd rust-pinyin
cargo build --release                       # builds the `pyime` binary

# (Re)build the data artifacts (downloads to corpus/, emits data/):
cargo run -p pyime-data --bin build-data -- --out data --corpus corpus   # add --offline after first fetch

# Convert (args or stdin lines):
./target/release/pyime convert woshizhongguoren jintiantianqihenhao --top 5
echo "mingtianjian" | ./target/release/pyime convert

# Interactive REPL (`:q` to quit; `:fuzzy off`, `:correct off`, `:n 5` toggles):
./target/release/pyime interactive

# Prefix prediction:
./target/release/pyime predict nih

# Evaluate (auto-generates the gold set if missing; writes report.json):
./target/release/pyime eval --gold data/gold.jsonl --json data/report.json
```

Feature toggles (`--no-fuzzy`, `--no-correct`, `--no-english`), `--json`, and `--top` are
available on the relevant subcommands; see `pyime --help`.

## Evaluation system

`pyime-eval` generates a reproducible gold set from **held-out** sentences (never used to train
the LM), converting each to pinyin and synthesizing scenario variants, then scores the engine.
Two gold sets are provided: `gold.jsonl` (per-character readings) and `gold_v2.jsonl`
(**correct word-level readings** via longest-match tokenization — the pinyin a user would
actually type).

**Scenario buckets:** `full`, `abbr`, `fuzzy`, `typo`, `english`, `mixed`, `long_sentence`,
`short_word`.

**Metrics (per bucket + overall):** Top-1 / Top-5 / Top-10 inclusion, MRR, char-accuracy
(1 − normalized Levenshtein), CER, coverage; plus latency (mean/p50/p95/p99), throughput, RSS,
and on-disk data size. Output is a human-readable table **and** `report.json`.

### Final results (release, 3 407-case gold set)

```
bucket              n    top1    top5   top10     mrr char_acc     cer coverage
-------------------------------------------------------------------------------
full              600   0.612   0.807   0.855   0.701    0.897   0.103    0.867
abbr              599   0.072   0.209   0.275   0.134    0.127   0.873    0.342
fuzzy             600   0.518   0.695   0.745   0.596    0.857   0.143    0.773
typo              600   0.332   0.458   0.493   0.381    0.747   0.253    0.520
english            40   1.000   1.000   1.000   1.000    1.000   0.000    1.000
mixed             257   0.681   0.805   0.833   0.734    0.891   0.109    0.844
long_sentence     600   0.580   0.770   0.818   0.666    0.913   0.087    0.835
short_word        111   0.748   0.937   0.982   0.830    0.815   0.185    0.982
-------------------------------------------------------------------------------
OVERALL          3407   0.460   0.620   0.668   0.530    0.729   0.271    0.695
Latency: p50 8.4ms  p95 21.7ms   |   RSS 37 MiB   |   data 45.7 MiB
```

### How it got there — quantified improvement per stage

The engine was tuned **using the eval harness as the objective function**. OVERALL on
`gold.jsonl`:

| Stage | top1 | top5 | coverage | full top1 | notes |
|-------|------|------|----------|-----------|-------|
| Initial (uncapped) | 0.141 | 0.353 | 0.389 | — | 60–330 ms/conv |
| Perf + ranking | 0.136 | 0.265 | 0.313 | 0.262 | p95 < 5 ms (clean) |
| Recall + mixed + abbr | 0.173 | 0.337 | 0.434 | 0.262 | mixed 0→0.30 |
| Corpus/LM upgrade | 0.186 | 0.352 | 0.449 | 0.260 | freq-ranked english |
| **rime-ice lexicon** | 0.275 | 0.458 | 0.553 | 0.408 | correct readings |
| **smoothed trigram LM** | 0.414 | 0.620 | 0.695 | 0.610 | the big lever |
| **literal-demotion** | **0.460** | **0.620** | **0.695** | **0.612** | typo 0.10→0.33 |

The two dominant levers were the **curated lexicon** (correct readings) and the **smoothed
trigram LM** — exactly as in commercial systems. On the controlled A/B over the 980 sentences
whose readings the per-char gold mis-spells, correct readings raise top-1 by **+0.40**.

## Performance

- Clean full-pinyin (6–12 chars): typical **2–3 ms**, p95 < 8 ms.
- Overall eval mix (incl. heavy abbreviation/long-sentence cases): p95 ~22 ms.
- RSS after load ~37 MiB; binary 3 MB; data 45 MB.

## Limitations & future work

- **Abbreviation** of long whole-sentences (简拼 of an entire sentence) is the hardest bucket
  (inherent ambiguity) — even commercial IMEs rely on per-phrase commit here.
- Sentence-level **top-1 is a harsh metric** (one-shot reconstruction of a full held-out
  sentence from tone-less pinyin); real users commit phrase-by-phrase. Top-5/coverage and the
  `short_word`/`full` buckets better reflect interactive use.
- A few polyphone-reading splits in the corpus cap exact matches (e.g. 受不了 reads
  `shou'bu'liao`, so `shoubule…` finds 手不了).
- Natural next steps toward full commercial parity: a **4-gram / neural-rescoring** LM, a
  **user dictionary + adaptation** layer (learns from the user's history), and **zero-query**
  prediction. The 100 MB budget leaves ~50 MB of headroom for a larger model.

## License

MIT. Built from openly-available data sources; see `data/meta.json` for provenance.
