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
