//! Word unigram + bigram + (optional) trigram language model with stupid-backoff.
//!
//! `bigram.fst` is an `fst::Map` keyed by `bigram_key(prev_id, id)` (8 bytes BE), value =
//! bigram cost. `trigram.fst` (OPTIONAL — v2) is an `fst::Map` keyed by
//! `trigram_key(w1, w2, w3)` (12 bytes BE), value = trigram cost. Missing n-grams back off to a
//! fixed penalty plus the lower-order estimate so the decoder always has a transition cost.
//!
//! Stupid-backoff transition (see DESIGN.md):
//!   `cost(w3 | w1, w2)` = `trigram[(w1,w2,w3)]`                          if present
//!                       = `TRIGRAM_BACKOFF + bigram[(w2,w3)]`            else if bigram present
//!                       = `TRIGRAM_BACKOFF + BIGRAM_BACKOFF + unigram(w3)` otherwise.
//! The 2-word case (sentence start / first transition) keeps using `transition_cost`.

use crate::consts::{BIGRAM_BACKOFF as CONST_BIGRAM_BACKOFF, TRIGRAM_BACKOFF as CONST_TRIGRAM_BACKOFF};
use crate::format::{bigram_key, trigram_key};
use fst::raw::Fst;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// Backoff penalty (in LOG_BASE cost units) added when a bigram is absent. Kept as a re-export of
/// the canonical `consts::BIGRAM_BACKOFF` for backwards compatibility with existing call sites.
pub const BIGRAM_BACKOFF: u32 = CONST_BIGRAM_BACKOFF;
/// Backoff penalty added when a trigram is absent (stupid-backoff). Mirror of `consts::TRIGRAM_BACKOFF`.
pub const TRIGRAM_BACKOFF: u32 = CONST_TRIGRAM_BACKOFF;

/// Sentinel word id meaning "no word yet" (sentence start / before the first word). Matches the
/// `u32::MAX` sentinel the decoder stores in `VNode.word_id` for the origin node, so the decoder
/// can pass raw ids straight through to `transition_cost3` without translating to `Option`.
pub const SENTENCE_START: u32 = u32::MAX;

pub struct LanguageModel {
    _mmap: Mmap,
    fst: Fst<&'static [u8]>,
    /// Optional trigram model. Present only when `data/trigram.fst` exists (v2 data). When absent
    /// the model runs bigram-only, exactly reproducing the v1 behavior.
    trigram: Option<TrigramModel>,
}

struct TrigramModel {
    _mmap: Mmap,
    fst: Fst<&'static [u8]>,
}

impl LanguageModel {
    pub fn load(data_dir: &Path) -> anyhow::Result<LanguageModel> {
        let path = data_dir.join("bigram.fst");
        let f = File::open(&path)
            .map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))?;
        // SAFETY: read-only mmap held for the lifetime of this struct.
        let mmap = unsafe { Mmap::map(&f) }
            .map_err(|e| anyhow::anyhow!("mmap {}: {e}", path.display()))?;
        let slice: &'static [u8] =
            unsafe { std::mem::transmute::<&[u8], &'static [u8]>(&mmap[..]) };
        let fst = Fst::new(slice).map_err(|e| anyhow::anyhow!("invalid bigram.fst: {e}"))?;

        // Optional trigram model: load if present, else stay bigram-only. A missing file MUST NOT
        // be an error — the engine degrades gracefully to the v1 bigram decoder.
        let trigram = Self::load_trigram(data_dir)?;

        Ok(LanguageModel { _mmap: mmap, fst, trigram })
    }

    /// Load `trigram.fst` if it exists. Returns `Ok(None)` when the file is absent (graceful
    /// bigram-only fallback); only a present-but-corrupt file is an error.
    fn load_trigram(data_dir: &Path) -> anyhow::Result<Option<TrigramModel>> {
        let path = data_dir.join("trigram.fst");
        if !path.exists() {
            return Ok(None);
        }
        let f = File::open(&path)
            .map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))?;
        // SAFETY: read-only mmap held for the lifetime of this struct.
        let mmap = unsafe { Mmap::map(&f) }
            .map_err(|e| anyhow::anyhow!("mmap {}: {e}", path.display()))?;
        let slice: &'static [u8] =
            unsafe { std::mem::transmute::<&[u8], &'static [u8]>(&mmap[..]) };
        let fst = Fst::new(slice).map_err(|e| anyhow::anyhow!("invalid trigram.fst: {e}"))?;
        Ok(Some(TrigramModel { _mmap: mmap, fst }))
    }

    /// True if a trigram model is loaded (v2 data). Useful for tests/diagnostics.
    pub fn has_trigram(&self) -> bool {
        self.trigram.is_some()
    }

    /// Bigram cost for `(prev_id, id)`, or `None` if absent (caller applies backoff).
    pub fn bigram_cost(&self, prev_id: u32, id: u32) -> Option<u32> {
        let key = bigram_key(prev_id, id);
        self.fst.get(key).map(|o| o.value() as u32)
    }

    /// Trigram cost for `(w1, w2, w3)`, or `None` if absent / no trigram model loaded.
    pub fn trigram_cost(&self, w1: u32, w2: u32, w3: u32) -> Option<u32> {
        let tg = self.trigram.as_ref()?;
        let key = trigram_key(w1, w2, w3);
        tg.fst.get(key).map(|o| o.value() as u32)
    }

    /// Bigram transition `cost(w3 | w2)` used as the stupid-backoff lower-order term: the bigram
    /// cost if present, else a flat `BIGRAM_BACKOFF` penalty.
    ///
    /// NOTE on the unigram term: the DESIGN stupid-backoff formula writes the bigram-miss case as
    /// `BIGRAM_BACKOFF + unigram_cost(w3)`. In THIS decoder the per-word unigram (reading) cost is
    /// already added separately as the word-edge cost (`WordEdge.cost`), so re-adding it here would
    /// double-count it and distort ranking. We therefore contribute only the `BIGRAM_BACKOFF`
    /// surcharge — identical to the v1 `transition_cost` backoff — so the unigram-arrival total
    /// (`edge.cost + BIGRAM_BACKOFF`) matches the spec while staying consistent with v1 behavior.
    #[inline]
    fn bigram_backoff_cost(&self, w2: u32, w3: u32, _unigram_w3: u32) -> u32 {
        self.bigram_cost(w2, w3).unwrap_or(BIGRAM_BACKOFF)
    }

    /// Transition cost from `prev` to `id`: bigram if present, else backoff penalty. Used for the
    /// 2-word case (sentence start / the first transition, where there is no `w_prevprev`).
    pub fn transition_cost(&self, prev_id: Option<u32>, id: u32) -> u32 {
        match prev_id {
            Some(p) => self.bigram_cost(p, id).unwrap_or(BIGRAM_BACKOFF),
            None => 0,
        }
    }

    /// Stupid-backoff trigram transition cost `cost(w3 | w1, w2)`.
    ///
    /// `unigram_w3` is the global unigram cost of `w3` (from its `WordEntry`), threaded in because
    /// the LM does not own the words table. `w1` and/or `w2` may be `SENTENCE_START`:
    ///   * `w2 == SENTENCE_START` means `w3` is the very first word — no history, cost 0.
    ///   * `w1 == SENTENCE_START` (but `w2` real) means only one word of history exists, so this
    ///     degrades to the bigram transition `cost(w3 | w2)` (no trigram lookup is possible).
    pub fn transition_cost3(&self, w1: u32, w2: u32, w3: u32, unigram_w3: u32) -> u32 {
        // No left context at all: first word of the sentence/run.
        if w2 == SENTENCE_START {
            return 0;
        }
        // Only one word of context: this is the second word, fall back to the bigram transition.
        if w1 == SENTENCE_START {
            return self.bigram_backoff_cost(w2, w3, unigram_w3);
        }
        // No trigram model loaded at all (bigram-only data): reduce EXACTLY to the v1 bigram
        // transition so the decoder does not regress when `trigram.fst` is absent. The
        // `TRIGRAM_BACKOFF` surcharge only has meaning relative to *present* trigram entries, which
        // do not exist without the file, so applying it here would only distort ranking vs the
        // (transition-free) English edges.
        let Some(tg) = self.trigram.as_ref() else {
            return self.bigram_backoff_cost(w2, w3, unigram_w3);
        };
        // Full trigram context.
        if let Some(o) = tg.fst.get(trigram_key(w1, w2, w3)) {
            return o.value() as u32;
        }
        TRIGRAM_BACKOFF + self.bigram_backoff_cost(w2, w3, unigram_w3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `LanguageModel` directly from in-memory bigram/(optional) trigram maps so the
    /// stupid-backoff arithmetic can be unit-tested without the full data pipeline. Writes the two
    /// fst maps to a temp dir and loads via `LanguageModel::load` (exercising the real loader,
    /// including the graceful trigram-absent path).
    fn lm(bigrams: &[(u32, u32, u64)], trigrams: Option<&[(u32, u32, u32, u64)]>) -> LanguageModel {
        use std::sync::atomic::{AtomicU64, Ordering};
        static CTR: AtomicU64 = AtomicU64::new(0);
        let uniq = CTR.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("pyime_lm_{}_{}", std::process::id(), uniq));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut bg: Vec<_> = bigrams.to_vec();
        bg.sort_by(|a, b| bigram_key(a.0, a.1).cmp(&bigram_key(b.0, b.1)));
        let mut bb = fst::MapBuilder::memory();
        for (p, i, c) in &bg {
            bb.insert(bigram_key(*p, *i), *c).unwrap();
        }
        std::fs::write(dir.join("bigram.fst"), bb.into_inner().unwrap()).unwrap();

        if let Some(tg) = trigrams {
            let mut tv: Vec<_> = tg.to_vec();
            tv.sort_by(|a, b| trigram_key(a.0, a.1, a.2).cmp(&trigram_key(b.0, b.1, b.2)));
            let mut tb = fst::MapBuilder::memory();
            for (a, b, c, cost) in &tv {
                tb.insert(trigram_key(*a, *b, *c), *cost).unwrap();
            }
            std::fs::write(dir.join("trigram.fst"), tb.into_inner().unwrap()).unwrap();
        }
        LanguageModel::load(&dir).unwrap()
    }

    #[test]
    fn sentinels_and_first_transitions() {
        let m = lm(&[(0, 1, 50)], None);
        assert!(!m.has_trigram());
        // First word: no history -> free.
        assert_eq!(m.transition_cost3(SENTENCE_START, SENTENCE_START, 0, 80), 0);
        // Second word (only one history word): degrades to the bigram transition.
        assert_eq!(m.transition_cost3(SENTENCE_START, 0, 1, 90), 50);
        // Second word with a missing bigram: flat BIGRAM_BACKOFF (no unigram double-count).
        assert_eq!(m.transition_cost3(SENTENCE_START, 0, 9, 90), BIGRAM_BACKOFF);
    }

    #[test]
    fn bigram_only_does_not_apply_trigram_backoff() {
        // No trigram model: a 3-word context must reduce EXACTLY to the bigram transition (no
        // TRIGRAM_BACKOFF surcharge), so the bigram-only decoder does not regress.
        let m = lm(&[(0, 1, 60), (1, 2, 70)], None);
        assert_eq!(m.transition_cost3(0, 1, 2, 100), 70, "present bigram, no trigram surcharge");
        assert_eq!(
            m.transition_cost3(0, 1, 9, 100),
            BIGRAM_BACKOFF,
            "missing bigram, no trigram surcharge when trigram model absent"
        );
    }

    #[test]
    fn trigram_present_and_backoff() {
        // Trigram (0,1,3) is cheap; (0,1,2) is absent so it backs off to TRIGRAM_BACKOFF + bigram.
        let m = lm(&[(1, 2, 70), (1, 3, 300)], Some(&[(0, 1, 3, 1)]));
        assert!(m.has_trigram());
        let c3 = m.transition_cost3(0, 1, 3, 400); // present trigram
        let c2 = m.transition_cost3(0, 1, 2, 100); // missing trigram -> backoff
        assert_eq!(c3, 1, "present trigram cost used verbatim");
        assert_eq!(c2, TRIGRAM_BACKOFF + 70, "trigram miss = TRIGRAM_BACKOFF + bigram(1->2)");
        assert!(c3 < c2, "the present (cheap) trigram must win");
        // Trigram miss AND bigram miss -> TRIGRAM_BACKOFF + BIGRAM_BACKOFF.
        assert_eq!(m.transition_cost3(0, 1, 9, 100), TRIGRAM_BACKOFF + BIGRAM_BACKOFF);
    }
}
