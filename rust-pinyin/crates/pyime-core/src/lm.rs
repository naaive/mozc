//! Word unigram + bigram + (optional) trigram language model — **signed log-ratio** transitions
//! with absolute-discounting smoothing (computed in pyime-data, consumed here).
//!
//! ## Cost convention (v3 — log-ratio over unigram)
//! The decoder already adds `unigram_cost(w3)` ≈ `-500·ln P(w3)` as the word-edge cost. The LM
//! transition is therefore the *bonus/penalty over the unigram*, on a single comparable scale:
//!   * `transition_bigram(w2,w3)  = -500·ln[ P(w3|w2)    / P(w3) ]`
//!   * `transition_trigram(w1,w2,w3) = -500·ln[ P(w3|w1,w2) / P(w3) ]`
//! so `unigram_cost(w3) + transition = -500·ln P(w3|context)` — a proper conditional. The ratio
//! can exceed 1 (context makes `w3` *more* likely), so transitions are **signed** (can be negative).
//! The trigram now *refines* the bigram on the same scale instead of dwarfing it.
//!
//! `P(·)` is estimated with absolute discounting + interpolation (D≈0.75) so rare n-grams do not
//! overfit; the writer bakes the final signed costs into the FSTs.
//!
//! ## On-disk encoding
//! `fst::Map` values are `u64`, but our costs are signed `i32`. We store `(cost + LM_COST_BIAS)` as
//! a `u64` and recover `cost = stored - LM_COST_BIAS`. `LM_COST_BIAS` is large enough that every
//! representable cost stays non-negative on disk.
//!
//! ## Backoff (signed)
//!   `cost(w3 | w1, w2)` = `trigram[(w1,w2,w3)]`                         if present
//!                       = `TRIGRAM_BACKOFF + bigram[(w2,w3)]`           else if bigram present
//!                       = `TRIGRAM_BACKOFF + BIGRAM_BACKOFF`            otherwise
//! (the unigram term lives on the word edge, so backoff is just the surcharge). The 2-word case
//! (sentence start / first transition) uses `transition_cost`.

use crate::consts::{
    BIGRAM_BACKOFF as CONST_BIGRAM_BACKOFF, FOURGRAM_BACKOFF as CONST_FOURGRAM_BACKOFF,
    TRIGRAM_BACKOFF as CONST_TRIGRAM_BACKOFF,
};
use crate::format::{bigram_key, fourgram_key, trigram_key};
use fst::raw::Fst;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// Signed backoff surcharge added when a bigram is absent. Re-export of `consts::BIGRAM_BACKOFF`.
pub const BIGRAM_BACKOFF: i32 = CONST_BIGRAM_BACKOFF;
/// Signed backoff surcharge added when a trigram is absent. Re-export of `consts::TRIGRAM_BACKOFF`.
pub const TRIGRAM_BACKOFF: i32 = CONST_TRIGRAM_BACKOFF;
/// Signed backoff surcharge added when a 4-gram is absent. Re-export of `consts::FOURGRAM_BACKOFF`.
pub const FOURGRAM_BACKOFF: i32 = CONST_FOURGRAM_BACKOFF;

/// Bias added to a signed transition cost before storing it as the (unsigned) `u64` FST value, and
/// subtracted on read. MUST exceed the most-negative representable cost in magnitude. Shared with
/// the data builder so writer and reader agree. `-500·ln(ratio)` for a strongly-boosted n-gram is
/// at worst a few thousand negative; 1<<20 leaves enormous headroom and is trivially decodable.
pub const LM_COST_BIAS: i64 = 1 << 20;

/// Decode a stored FST value back into the signed cost it encodes.
#[inline]
pub fn decode_cost(stored: u64) -> i32 {
    (stored as i64 - LM_COST_BIAS) as i32
}

/// Encode a signed cost into the `u64` FST value (used by the data builder via re-export).
#[inline]
pub fn encode_cost(cost: i32) -> u64 {
    (cost as i64 + LM_COST_BIAS) as u64
}

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
    /// Optional 4-gram model. Present only when `data/fourgram.fst` exists. Used ONLY by the N-best
    /// rescoring pass (`transition_cost4`), never by the trigram beam. When absent the engine behaves
    /// exactly as the trigram-only decoder (no rescoring, no regression).
    fourgram: Option<FourgramModel>,
}

struct TrigramModel {
    _mmap: Mmap,
    fst: Fst<&'static [u8]>,
}

struct FourgramModel {
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
        // Optional 4-gram model: load if present, else the rescoring pass is inert. A missing file
        // MUST NOT be an error — the engine degrades gracefully to the trigram-only decoder.
        let fourgram = Self::load_fourgram(data_dir)?;

        Ok(LanguageModel { _mmap: mmap, fst, trigram, fourgram })
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

    /// Load `fourgram.fst` if it exists. Returns `Ok(None)` when the file is absent (graceful
    /// trigram-only fallback — the rescoring pass becomes a no-op); only a present-but-corrupt file
    /// is an error. Mirrors `load_trigram`.
    fn load_fourgram(data_dir: &Path) -> anyhow::Result<Option<FourgramModel>> {
        let path = data_dir.join("fourgram.fst");
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
        let fst = Fst::new(slice).map_err(|e| anyhow::anyhow!("invalid fourgram.fst: {e}"))?;
        Ok(Some(FourgramModel { _mmap: mmap, fst }))
    }

    /// True if a trigram model is loaded (v2 data). Useful for tests/diagnostics.
    pub fn has_trigram(&self) -> bool {
        self.trigram.is_some()
    }

    /// True if a 4-gram model is loaded (the N-best rescoring pass is active). Useful for
    /// tests/diagnostics and to let the decoder skip the rescoring pass entirely when absent.
    pub fn has_fourgram(&self) -> bool {
        self.fourgram.is_some()
    }

    /// Signed bigram transition cost for `(prev_id, id)`, or `None` if absent (caller applies
    /// backoff). The stored FST value is the biased encoding of `-500·ln[P(id|prev)/P(id)]`.
    pub fn bigram_cost(&self, prev_id: u32, id: u32) -> Option<i32> {
        let key = bigram_key(prev_id, id);
        self.fst.get(key).map(|o| decode_cost(o.value()))
    }

    /// Signed trigram transition cost for `(w1, w2, w3)`, or `None` if absent / no trigram model.
    pub fn trigram_cost(&self, w1: u32, w2: u32, w3: u32) -> Option<i32> {
        let tg = self.trigram.as_ref()?;
        let key = trigram_key(w1, w2, w3);
        tg.fst.get(key).map(|o| decode_cost(o.value()))
    }

    /// Bigram transition `cost(w3 | w2)` used as the lower-order term: the signed bigram log-ratio
    /// cost if present, else a flat `BIGRAM_BACKOFF` surcharge. The unigram cost itself lives on the
    /// word edge (`WordEdge.cost`), so this contributes ONLY the relative bonus/penalty — adding the
    /// unigram here would double-count it.
    #[inline]
    fn bigram_backoff_cost(&self, w2: u32, w3: u32) -> i32 {
        self.bigram_cost(w2, w3).unwrap_or(BIGRAM_BACKOFF)
    }

    /// Transition cost from `prev` to `id`: bigram log-ratio if present, else backoff. Used for the
    /// 2-word case (sentence start / the first transition, where there is no `w_prevprev`).
    pub fn transition_cost(&self, prev_id: Option<u32>, id: u32) -> i32 {
        match prev_id {
            Some(p) => self.bigram_cost(p, id).unwrap_or(BIGRAM_BACKOFF),
            None => 0,
        }
    }

    /// Signed log-ratio trigram transition cost `cost(w3 | w1, w2)` with backoff.
    ///
    /// `_unigram_w3` is accepted for API stability (the log-ratio already factors the unigram out,
    /// so it is no longer consulted here). `w1` and/or `w2` may be `SENTENCE_START`:
    ///   * `w2 == SENTENCE_START` → `w3` is the very first word, no history, cost 0.
    ///   * `w1 == SENTENCE_START` (but `w2` real) → only one word of history; use the bigram term.
    pub fn transition_cost3(&self, w1: u32, w2: u32, w3: u32, _unigram_w3: u32) -> i32 {
        // No left context at all: first word of the sentence/run.
        if w2 == SENTENCE_START {
            return 0;
        }
        // Only one word of context: this is the second word, fall back to the bigram transition.
        if w1 == SENTENCE_START {
            return self.bigram_backoff_cost(w2, w3);
        }
        // No trigram model loaded (bigram-only data): reduce EXACTLY to the bigram transition so the
        // decoder does not regress when `trigram.fst` is absent (no TRIGRAM_BACKOFF surcharge — it
        // only has meaning relative to *present* trigram entries).
        let Some(tg) = self.trigram.as_ref() else {
            return self.bigram_backoff_cost(w2, w3);
        };
        // Full trigram context.
        if let Some(o) = tg.fst.get(trigram_key(w1, w2, w3)) {
            return decode_cost(o.value());
        }
        TRIGRAM_BACKOFF + self.bigram_backoff_cost(w2, w3)
    }

    /// Signed 4-gram transition cost for `(w0, w1, w2, w3)`, or `None` if absent / no 4-gram model.
    /// The stored FST value uses the SAME signed-log-ratio encoding as bigram/trigram.
    pub fn fourgram_cost(&self, w0: u32, w1: u32, w2: u32, w3: u32) -> Option<i32> {
        let fg = self.fourgram.as_ref()?;
        let key = fourgram_key(w0, w1, w2, w3);
        fg.fst.get(key).map(|o| decode_cost(o.value()))
    }

    /// Signed log-ratio 4-gram transition cost `cost(w3 | w0, w1, w2)` with stupid-backoff:
    ///   * full 4-gram `(w0,w1,w2,w3)` if present,
    ///   * else `FOURGRAM_BACKOFF + transition_cost3(w1, w2, w3)` (which itself backs off 3→2→1).
    ///
    /// Used ONLY by the decoder's N-best rescoring pass. Any of `w0`/`w1`/`w2` may be the
    /// `SENTENCE_START` sentinel; in that case there is no full 4-gram context, so it falls straight
    /// through to `transition_cost3` (consistent with how the trigram beam handles sentence starts).
    /// The `unigram_w3` term is paid on the word edge by the caller, mirroring `transition_cost3`.
    pub fn transition_cost4(&self, w0: u32, w1: u32, w2: u32, w3: u32, unigram_w3: u32) -> i32 {
        // No full 4-gram history (sentence start) or no 4-gram model loaded: fall through to the
        // trigram transition with NO surcharge (FOURGRAM_BACKOFF has meaning only relative to a
        // *present* 4-gram entry, so adding it on a sentinel/absent-model path would distort the
        // trigram-equivalent cost and could regress).
        if w0 == SENTENCE_START || w1 == SENTENCE_START || w2 == SENTENCE_START {
            return self.transition_cost3(w1, w2, w3, unigram_w3);
        }
        if self.fourgram.is_none() {
            return self.transition_cost3(w1, w2, w3, unigram_w3);
        }
        if let Some(c) = self.fourgram_cost(w0, w1, w2, w3) {
            return c;
        }
        FOURGRAM_BACKOFF + self.transition_cost3(w1, w2, w3, unigram_w3)
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
        lm_full(bigrams, trigrams, None)
    }

    /// Like `lm`, but also optionally writes a `fourgram.fst` so `transition_cost4` can be tested.
    fn lm_full(
        bigrams: &[(u32, u32, u64)],
        trigrams: Option<&[(u32, u32, u32, u64)]>,
        fourgrams: Option<&[(u32, u32, u32, u32, u64)]>,
    ) -> LanguageModel {
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
            // The fixtures express costs as plain (signed) values; store them with the on-disk bias.
            bb.insert(bigram_key(*p, *i), encode_cost(*c as i32)).unwrap();
        }
        std::fs::write(dir.join("bigram.fst"), bb.into_inner().unwrap()).unwrap();

        if let Some(tg) = trigrams {
            let mut tv: Vec<_> = tg.to_vec();
            tv.sort_by(|a, b| trigram_key(a.0, a.1, a.2).cmp(&trigram_key(b.0, b.1, b.2)));
            let mut tb = fst::MapBuilder::memory();
            for (a, b, c, cost) in &tv {
                tb.insert(trigram_key(*a, *b, *c), encode_cost(*cost as i32)).unwrap();
            }
            std::fs::write(dir.join("trigram.fst"), tb.into_inner().unwrap()).unwrap();
        }

        if let Some(fg) = fourgrams {
            let mut fv: Vec<_> = fg.to_vec();
            fv.sort_by(|a, b| {
                fourgram_key(a.0, a.1, a.2, a.3).cmp(&fourgram_key(b.0, b.1, b.2, b.3))
            });
            let mut fb = fst::MapBuilder::memory();
            for (a, b, c, d, cost) in &fv {
                fb.insert(fourgram_key(*a, *b, *c, *d), encode_cost(*cost as i32)).unwrap();
            }
            std::fs::write(dir.join("fourgram.fst"), fb.into_inner().unwrap()).unwrap();
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

    #[test]
    fn fourgram_absent_reduces_to_trigram() {
        // No fourgram.fst: transition_cost4 must equal transition_cost3 on the right 3 words, with
        // NO surcharge, so the rescoring pass is a strict no-op without the file.
        let m = lm_full(&[(1, 2, 70), (2, 3, 80)], Some(&[(1, 2, 3, 5)]), None);
        assert!(!m.has_fourgram());
        // Full history present, no 4-gram model -> exactly the trigram cost.
        assert_eq!(
            m.transition_cost4(0, 1, 2, 3, 100),
            m.transition_cost3(1, 2, 3, 100)
        );
        assert_eq!(m.transition_cost4(0, 1, 2, 3, 100), 5);
    }

    #[test]
    fn fourgram_present_and_backoff() {
        // 4-gram (0,1,2,3) is cheap; (0,1,2,9) is absent so it backs off to FOURGRAM_BACKOFF +
        // trigram(1,2,9) which itself backs off (no trigram (1,2,9)) to TRIGRAM_BACKOFF+bigram(2,9).
        let m = lm_full(
            &[(1, 2, 70), (2, 3, 80)],
            Some(&[(1, 2, 3, 5)]),
            Some(&[(0, 1, 2, 3, 2)]),
        );
        assert!(m.has_fourgram());
        // Present 4-gram used verbatim.
        assert_eq!(m.transition_cost4(0, 1, 2, 3, 100), 2);
        // Missing 4-gram -> FOURGRAM_BACKOFF + trigram(1,2,3) (which is present = 5).
        assert_eq!(m.transition_cost4(9, 1, 2, 3, 100), FOURGRAM_BACKOFF + 5);
        // Missing 4-gram AND missing trigram -> FOURGRAM_BACKOFF + (TRIGRAM_BACKOFF + bigram(2,9)).
        // bigram(2,9) is absent -> BIGRAM_BACKOFF.
        assert_eq!(
            m.transition_cost4(0, 1, 2, 9, 100),
            FOURGRAM_BACKOFF + TRIGRAM_BACKOFF + BIGRAM_BACKOFF
        );
        // Sentence-start sentinel anywhere in the history -> falls through to trigram, no surcharge.
        assert_eq!(
            m.transition_cost4(SENTENCE_START, 1, 2, 3, 100),
            m.transition_cost3(1, 2, 3, 100)
        );
    }
}
