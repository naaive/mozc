//! Optional, persistent **user model** for online personalization (v3 addition — see DESIGN.md).
//!
//! Every commercial IME learns from what the user commits. This module implements a CPU-trivial,
//! no-training adaptation layer: it records what the user selected and turns those observations into
//! small, **capped** negative costs (bonuses) that re-rank candidates *within* the N-best without
//! ever overriding clean-input correctness.
//!
//! ## State
//!   * **user unigram counts** `word_id → UnigramStat{count, last_used}` — how often (and how
//!     recently) the user committed a word.
//!   * **user bigram counts** `(prev_word_id, word_id) → count` — personalized transitions.
//!   * **user phrases** `normalized_pinyin_key → surface` — a full committed selection stored as a
//!     unit, so a previously-committed phrase (INCLUDING auto-learned new words not in the base
//!     lexicon, e.g. names/neologisms) is re-surfaced at/near #1 next time the same input is typed.
//!   * a monotonic `tick` used for recency / LRU.
//!
//! ## Bonus formula (negative cost = a discount)
//! For a word with count `c` and recency age `a = tick - last_used` (in commits):
//!   `freq_bonus  = round(FREQ_SCALE · ln(1 + c))`                         (saturating, frequency)
//!   `recency_bonus = if a == 0 { RECENCY_FRESH } else { RECENCY_DECAY / (1 + a) }`  (LRU boost)
//!   `unigram_bonus = min(freq_bonus + recency_bonus, UNIGRAM_CAP)`
//! The bigram bonus is `min(round(BIGRAM_SCALE · ln(1 + bc)), BIGRAM_CAP)` for a committed
//! `(prev, word)` pair seen `bc` times. Every term is **capped** so the personalization budget is a
//! few thousand cost units — enough to re-rank the N-best, never enough to beat a clearly-correct
//! clean reading by a wide margin.
//!
//! `bonus()` returns the (pre-`user_weight`) raw discount; the decoder scales it by
//! `cfg.user_weight / USER_WEIGHT_STD` so `user_weight == 0` disables personalization entirely and
//! the default (`USER_WEIGHT_STD`) applies the formula as written.

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Reference `user_weight` at which the raw bonus formula is applied 1:1. The decoder scales every
/// bonus by `user_weight / USER_WEIGHT_STD`, so this is the "standard strength".
pub const USER_WEIGHT_STD: i32 = 800;

// --- Bonus tuning constants (raw, pre-weight cost units). All discounts are capped. ---
/// Frequency term scale: `FREQ_SCALE · ln(1+count)`.
const FREQ_SCALE: f32 = 900.0;
/// Cap on the per-word unigram bonus (freq + recency). Keeps personalization from dominating.
const UNIGRAM_CAP: i32 = 2600;
/// Extra discount for the single most-recently-used word (age 0).
const RECENCY_FRESH: i32 = 700;
/// Decayed recency discount for older words: `RECENCY_DECAY / (1 + age)`.
const RECENCY_DECAY: i32 = 1400;
/// Bigram term scale: `BIGRAM_SCALE · ln(1+count)`.
const BIGRAM_SCALE: f32 = 700.0;
/// Cap on the per-transition user bigram bonus.
const BIGRAM_CAP: i32 = 2000;

/// Raw discount applied to an injected user-phrase candidate (before `user_weight` scaling). Large
/// but still capped — a previously-committed phrase should rank at/near #1, yet a single big clean
/// reading can still win if its cost gap to the phrase exceeds this scaled budget. Tuned so a
/// committed phrase reliably reaches #1 on re-entry of the exact input.
pub const PHRASE_BONUS: i32 = 9000;

/// Per-word frequency/recency stat.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct UnigramStat {
    count: u32,
    last_used: u64,
}

/// The persistent user model. Serialized compactly to JSON via serde.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserModel {
    /// Monotonic commit counter (drives recency / LRU).
    tick: u64,
    /// word_id → frequency/recency stat.
    unigram: FxHashMap<u32, UnigramStat>,
    /// (prev_word_id, word_id) → count. Serialized as a string key `"prev,word"` for compact JSON.
    #[serde(with = "bigram_serde")]
    bigram: FxHashMap<(u32, u32), u32>,
    /// normalized pinyin key → committed surface (auto-learned units, even outside the lexicon).
    phrases: FxHashMap<String, String>,
}

impl UserModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load a user model from `path`. A missing file yields a fresh, empty model (first run). Only a
    /// present-but-corrupt file is an error.
    pub fn load(path: &Path) -> anyhow::Result<UserModel> {
        if !path.exists() {
            return Ok(UserModel::new());
        }
        let bytes = std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("read user model {}: {e}", path.display()))?;
        if bytes.is_empty() {
            return Ok(UserModel::new());
        }
        let model: UserModel = serde_json::from_slice(&bytes)
            .map_err(|e| anyhow::anyhow!("parse user model {}: {e}", path.display()))?;
        Ok(model)
    }

    /// Persist the user model to `path` (compact JSON). Creates parent directories as needed.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).ok();
            }
        }
        let bytes = serde_json::to_vec(self)
            .map_err(|e| anyhow::anyhow!("serialize user model: {e}"))?;
        std::fs::write(path, bytes)
            .map_err(|e| anyhow::anyhow!("write user model {}: {e}", path.display()))?;
        Ok(())
    }

    /// Normalize a raw input buffer into the phrase key: lowercase ASCII letters/digits only,
    /// dropping separators (`'`, spaces, punctuation). `ni'hao` and `nihao` share a key.
    pub fn normalize_key(input: &str) -> String {
        input
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_lowercase())
            .collect()
    }

    /// Record a committed selection. `word_ids` is the lexicon tokenization of `chosen` (computed by
    /// the Engine, which owns the lexicon); it may be empty for an out-of-vocabulary phrase — the
    /// phrase entry is stored regardless so auto-learned new words are still re-surfaced.
    ///
    ///   * stores the user phrase `normalize_key(input) → chosen`,
    ///   * bumps per-word unigram counts + recency for every id in `word_ids`,
    ///   * bumps per-(prev,word) bigram counts along the id sequence.
    pub fn record(&mut self, input: &str, chosen: &str, word_ids: &[u32]) {
        if chosen.is_empty() {
            return;
        }
        self.tick += 1;
        let tick = self.tick;

        let key = Self::normalize_key(input);
        if !key.is_empty() {
            self.phrases.insert(key, chosen.to_string());
        }

        for &id in word_ids {
            let stat = self.unigram.entry(id).or_default();
            stat.count += 1;
            stat.last_used = tick;
        }
        for pair in word_ids.windows(2) {
            *self.bigram.entry((pair[0], pair[1])).or_insert(0) += 1;
        }
    }

    /// Raw (pre-`user_weight`) unigram bonus (negative-cost magnitude) for `word_id`, combining a
    /// frequency term and a recency/LRU boost, capped at `UNIGRAM_CAP`. Returns 0 for an unseen word.
    pub fn unigram_bonus(&self, word_id: u32) -> i32 {
        let Some(stat) = self.unigram.get(&word_id) else {
            return 0;
        };
        if stat.count == 0 {
            return 0;
        }
        let freq = (FREQ_SCALE * (1.0 + stat.count as f32).ln()).round() as i32;
        let age = self.tick.saturating_sub(stat.last_used);
        let recency = if age == 0 {
            RECENCY_FRESH
        } else {
            RECENCY_DECAY / (1 + age as i32)
        };
        (freq + recency).min(UNIGRAM_CAP)
    }

    /// Raw (pre-`user_weight`) bigram bonus for a committed `(prev, word)` transition, capped at
    /// `BIGRAM_CAP`. Returns 0 when `prev` is absent or the pair was never committed.
    pub fn bigram_bonus(&self, prev_word_id: Option<u32>, word_id: u32) -> i32 {
        let Some(prev) = prev_word_id else { return 0 };
        let Some(&c) = self.bigram.get(&(prev, word_id)) else {
            return 0;
        };
        if c == 0 {
            return 0;
        }
        ((BIGRAM_SCALE * (1.0 + c as f32).ln()).round() as i32).min(BIGRAM_CAP)
    }

    /// Combined raw bonus for placing `word_id` after `prev_word_id` (unigram + bigram), as a
    /// negative-cost magnitude. The decoder subtracts `bonus · user_weight / USER_WEIGHT_STD`.
    pub fn bonus(&self, word_id: u32, prev_word_id: Option<u32>) -> i32 {
        self.unigram_bonus(word_id) + self.bigram_bonus(prev_word_id, word_id)
    }

    /// The committed surface for an exact normalized input key, if any (auto-learned phrase lookup).
    pub fn phrase_for(&self, input: &str) -> Option<&str> {
        let key = Self::normalize_key(input);
        self.phrases.get(&key).map(|s| s.as_str())
    }

    /// True if the model carries no learned state (used to keep an empty model a strict no-op).
    pub fn is_empty(&self) -> bool {
        self.unigram.is_empty() && self.bigram.is_empty() && self.phrases.is_empty()
    }
}

/// Compact serde for the `(u32,u32) → u32` bigram map: JSON object keyed by `"prev,word"`.
mod bigram_serde {
    use rustc_hash::FxHashMap;
    use serde::de::{Deserialize, Deserializer};
    use serde::ser::{Serialize, Serializer};
    use std::collections::BTreeMap;

    pub fn serialize<S: Serializer>(
        map: &FxHashMap<(u32, u32), u32>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        // BTreeMap gives a stable, deterministic on-disk ordering.
        let str_map: BTreeMap<String, u32> = map
            .iter()
            .map(|((p, w), c)| (format!("{p},{w}"), *c))
            .collect();
        str_map.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<FxHashMap<(u32, u32), u32>, D::Error> {
        let str_map: BTreeMap<String, u32> = BTreeMap::deserialize(d)?;
        let mut out: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        for (k, v) in str_map {
            if let Some((p, w)) = k.split_once(',') {
                if let (Ok(p), Ok(w)) = (p.parse(), w.parse()) {
                    out.insert((p, w), v);
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_bonus_grow_with_count() {
        let mut m = UserModel::new();
        let b0 = m.unigram_bonus(3);
        assert_eq!(b0, 0, "unseen word has no bonus");
        m.record("wo", "我", &[3]);
        let b1 = m.unigram_bonus(3);
        assert!(b1 > 0, "committed word gets a positive bonus, got {b1}");
        m.record("wo", "我", &[3]);
        m.record("wo", "我", &[3]);
        let b3 = m.unigram_bonus(3);
        assert!(b3 >= b1, "more commits → at least as large a bonus ({b3} >= {b1})");
        assert!(b3 <= UNIGRAM_CAP, "bonus stays capped ({b3} <= {UNIGRAM_CAP})");
    }

    #[test]
    fn bigram_bonus_recorded() {
        let mut m = UserModel::new();
        m.record("woyong", "我用", &[3, 4]);
        assert!(m.bigram_bonus(Some(3), 4) > 0, "committed bigram gets a bonus");
        assert_eq!(m.bigram_bonus(Some(4), 3), 0, "uncommitted bigram has none");
        assert_eq!(m.bigram_bonus(None, 4), 0, "no prev → no bigram bonus");
    }

    #[test]
    fn phrase_key_normalization() {
        let mut m = UserModel::new();
        m.record("ni'hao", "你好", &[0]);
        assert_eq!(m.phrase_for("nihao"), Some("你好"));
        assert_eq!(m.phrase_for("ni'hao"), Some("你好"));
        assert_eq!(m.phrase_for("nihaa"), None);
    }

    #[test]
    fn roundtrip_serde() {
        let mut m = UserModel::new();
        m.record("woyong", "我用github", &[3, 4]);
        m.record("nihao", "你好", &[0]);
        let dir = std::env::temp_dir().join(format!("pyime_user_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("user.json");
        m.save(&path).unwrap();
        let m2 = UserModel::load(&path).unwrap();
        assert_eq!(m2.phrase_for("woyong"), Some("我用github"));
        assert_eq!(m2.phrase_for("nihao"), Some("你好"));
        assert!(m2.bigram_bonus(Some(3), 4) > 0);
        assert!(m2.unigram_bonus(3) > 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_model_is_noop() {
        let m = UserModel::new();
        assert!(m.is_empty());
        assert_eq!(m.bonus(0, None), 0);
        assert_eq!(m.bonus(5, Some(3)), 0);
        assert_eq!(m.phrase_for("nihao"), None);
    }
}
