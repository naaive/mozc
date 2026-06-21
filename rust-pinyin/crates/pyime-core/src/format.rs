//! Shared on-disk data format types (see DESIGN.md). Used by BOTH the data builder
//! (pyime-data, writer) and the engine (pyime-core, reader) so they cannot drift.

use rkyv::{Archive, Deserialize, Serialize};

/// A dictionary word. `words.bin` is an rkyv-archived `Vec<WordEntry>` indexed by word id.
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct WordEntry {
    pub surface: String,
    /// Global unigram cost (lower = more frequent).
    pub unigram_cost: u16,
    /// Coarse part-of-speech / category tag (0 = generic).
    pub pos: u8,
}

/// JSON sidecar describing a built data set.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct Meta {
    pub version: u32,
    pub log_base: f32,
    pub num_words: u64,
    pub num_readings: u64,
    pub num_bigrams: u64,
    pub bytes_total: u64,
    pub source_notes: String,
}

pub const FORMAT_VERSION: u32 = 1;

/// Encode a (prev_id, id) bigram pair into the 8-byte big-endian fst key.
#[inline]
pub fn bigram_key(prev_id: u32, id: u32) -> [u8; 8] {
    let mut k = [0u8; 8];
    k[..4].copy_from_slice(&prev_id.to_be_bytes());
    k[4..].copy_from_slice(&id.to_be_bytes());
    k
}

/// Encode a (w1, w2, w3) trigram into the 12-byte big-endian fst key.
#[inline]
pub fn trigram_key(w1: u32, w2: u32, w3: u32) -> [u8; 12] {
    let mut k = [0u8; 12];
    k[..4].copy_from_slice(&w1.to_be_bytes());
    k[4..8].copy_from_slice(&w2.to_be_bytes());
    k[8..].copy_from_slice(&w3.to_be_bytes());
    k
}
