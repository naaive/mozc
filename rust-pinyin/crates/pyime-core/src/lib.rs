//! pyime-core — the Pinyin IME engine (lattice + word-bigram decoder).
//!
//! This file defines the STABLE public API contract (see ../../DESIGN.md). The CORE agent
//! implements the modules below. Other crates depend only on the items re-exported here.

use std::path::Path;

pub mod format;
pub mod fuzzy;
pub mod syllable;
pub mod segment;
pub mod lexicon;
pub mod lm;
pub mod decoder;

pub use fuzzy::FuzzySet;

/// Shared cost/scaling constants (see DESIGN.md).
pub mod consts {
    pub const LOG_BASE: f32 = 500.0;
    pub const FUZZY_PEN: i32 = 1200;
    pub const TYPO_PEN: i32 = 2400;
    pub const ABBR_PEN: i32 = 1800;
    /// Penalty applied per English edge so clean pinyin segmentation wins.
    pub const ENGLISH_PEN: i32 = 3000;

    /// Backoff surcharge (in LOG_BASE cost units) added to the SIGNED log-ratio transition when a
    /// *bigram* `(w2,w3)` is absent. With the interpolated log-ratio LM (see `lm.rs`), the stored
    /// transition is `-500·ln[ P(w3|ctx) / P(w3) ]`; a missing bigram means the conditional reduces
    /// to ~`λ(w2)·P(w3)`, i.e. the ratio is ~λ < 1, so the natural surcharge is `-500·ln λ`. λ is
    /// typically 0.1–0.4, giving ~450–1150. We use a single representative value: the absence of a
    /// bigram is mild evidence (the unigram cost is still paid on the edge), so it stays small.
    pub const BIGRAM_BACKOFF: i32 = 700;
    /// Backoff surcharge added when a *trigram* `(w1,w2,w3)` is absent and the model falls through
    /// to the (signed) bigram log-ratio. With interpolation a trigram-miss reduces to essentially
    /// the bigram estimate scaled by the trigram interpolation weight λ3, so the extra surcharge for
    /// dropping one order is small. Kept low so a present trigram is preferred but a missing one
    /// still ranks cleanly on its bigram evidence (no longer dwarfs the bigram scale).
    pub const TRIGRAM_BACKOFF: i32 = 300;
    /// Backoff surcharge added when a *4-gram* `(w0,w1,w2,w3)` is absent and the rescoring pass
    /// falls through to the (signed) trigram log-ratio. Same rationale as `TRIGRAM_BACKOFF`: with
    /// interpolation, dropping one order is mild evidence, so the surcharge is small. Tuned by the
    /// 4-gram rescoring agent against the eval harness.
    pub const FOURGRAM_BACKOFF: i32 = 300;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateKind {
    Chinese,
    English,
    Mixed,
}

#[derive(Debug, Clone)]
pub struct Segment {
    /// Output surface for this segment (e.g. "你好" or "github").
    pub text: String,
    /// Canonical reading matched (pinyin), empty for English.
    pub reading: String,
    /// Byte range in the (normalized) input consumed by this segment.
    pub input_span: (usize, usize),
}

#[derive(Debug, Clone)]
pub struct Candidate {
    pub text: String,
    pub score: f32,
    pub segments: Vec<Segment>,
    pub kind: CandidateKind,
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub fuzzy: FuzzySet,
    pub enable_correction: bool,
    pub correction_max_edits: u8,
    pub enable_english: bool,
    pub max_candidates: usize,
    pub beam_width: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            fuzzy: FuzzySet::all(),
            enable_correction: true,
            correction_max_edits: 1,
            enable_english: true,
            max_candidates: 20,
            beam_width: 20,
        }
    }
}

/// The engine. Loads built data and converts input → ranked candidates.
pub struct Engine {
    pub(crate) lexicon: lexicon::Lexicon,
    pub(crate) lm: lm::LanguageModel,
}

impl Engine {
    /// Load built data from a directory (see DESIGN.md data contract).
    pub fn load(data_dir: &Path) -> anyhow::Result<Engine> {
        let lexicon = lexicon::Lexicon::load(data_dir)?;
        let lm = lm::LanguageModel::load(data_dir)?;
        Ok(Engine { lexicon, lm })
    }

    /// Convert a raw input buffer into ranked candidates (best first).
    pub fn convert(&self, input: &str, cfg: &EngineConfig) -> Vec<Candidate> {
        decoder::decode(self, input, cfg)
    }

    /// Prefix prediction / completion for a partial input.
    pub fn predict(&self, input: &str, cfg: &EngineConfig) -> Vec<Candidate> {
        decoder::predict(self, input, cfg)
    }
}
