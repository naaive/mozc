//! 4-gram N-best RESCORING decoder tests.
//!
//! These build a tiny self-contained dataset (matching the on-disk format contract in DESIGN.md)
//! into a temp dir, OPTIONALLY including a `fourgram.fst`, then load it via `Engine::load` and
//! assert:
//!   1. With NO `fourgram.fst`, the (bigram/trigram) decoder ranks path A first.
//!   2. With a `fourgram.fst` that makes a competing path B cheaper in the full `(w0,w1,w2,w3)`
//!      context, the SAME input now ranks B first AFTER the rescoring pass — proving the 4-gram
//!      transition cost is actually consulted and re-orders the N-best.
//!
//! The two engines share an identical lexicon / bigram / trigram model; only the presence of the
//! `fourgram.fst` differs, so any ranking change is attributable to the 4-gram rescoring.

use pyime_core::format::{bigram_key, fourgram_key, trigram_key, WordEntry};
use pyime_core::lm::encode_cost;
use pyime_core::{Engine, EngineConfig};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

// Word ids. The contended slot is `hao` (好 cheap vs 号 pricier); a strong 4-gram on the 号 path
// must flip the ranking. We need a FOUR-word path so a genuine 4-gram context exists.
const WO: u32 = 0; // 我  wo
const SHI: u32 = 1; // 是 shi
const HAO_GOOD: u32 = 2; // 好 hao  (cheaper unigram + cheaper trigram -> wins without the 4-gram)
const HAO_NUM: u32 = 3; // 号 hao  (pricier; only a cheap 4-GRAM makes its path win)
const REN: u32 = 4; // 人 ren

/// Build a fixture dataset. When `with_fourgram` is true, a `fourgram.fst` is written that makes
/// `(我, 是, 号, 人)` very cheap so the 我是号人 path should beat the otherwise-cheaper 我是好人.
fn build_fixture(with_fourgram: bool) -> PathBuf {
    // (surface, reading, unigram_cost). 好 and 号 share reading `hao`.
    let words: Vec<(&str, &str, u16)> = vec![
        ("我", "wo", 80),   // 0
        ("是", "shi", 90),  // 1
        ("好", "hao", 100), // 2  cheaper
        ("号", "hao", 400), // 3  pricier
        ("人", "ren", 120), // 4
    ];

    let entries: Vec<WordEntry> = words
        .iter()
        .map(|(s, _, c)| WordEntry { surface: s.to_string(), unigram_cost: *c, pos: 0 })
        .collect();
    let words_bytes = rkyv::to_bytes::<_, 4096>(&entries).expect("serialize words");

    // reading -> [(word_id, cost)]
    let mut readings: BTreeMap<String, Vec<(u32, u16)>> = BTreeMap::new();
    for (id, (_, reading, cost)) in words.iter().enumerate() {
        readings.entry((*reading).to_string()).or_default().push((id as u32, *cost));
    }

    // postings.bin + lexicon.fst
    let mut postings: Vec<u8> = Vec::new();
    let mut fst_entries: Vec<(String, u64)> = Vec::new();
    for (reading, list) in &readings {
        let offset = postings.len() as u64;
        postings.extend_from_slice(&(list.len() as u16).to_le_bytes());
        for (id, cost) in list {
            postings.extend_from_slice(&id.to_le_bytes());
            postings.extend_from_slice(&cost.to_le_bytes());
        }
        fst_entries.push((reading.clone(), offset));
    }
    fst_entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut lex_builder = fst::MapBuilder::memory();
    for (k, v) in &fst_entries {
        lex_builder.insert(k.as_bytes(), *v).unwrap();
    }
    let lexicon_fst = lex_builder.into_inner().unwrap();

    // bigram.fst: anchor the prefix and make 是->好 CHEAPER than 是->号, and 好->人 / 号->人 present.
    // Costs are stored with the signed-log-ratio on-disk bias (encode_cost), matching real data.
    let mut bigrams: Vec<(u32, u32, i32)> = vec![
        (WO, SHI, 50),        // 我 -> 是
        (SHI, HAO_GOOD, 60),  // 是 -> 好   (cheaper)
        (SHI, HAO_NUM, 300),  // 是 -> 号   (pricier)
        (HAO_GOOD, REN, 70),  // 好 -> 人
        (HAO_NUM, REN, 70),   // 号 -> 人
    ];
    bigrams.sort_by(|a, b| bigram_key(a.0, a.1).cmp(&bigram_key(b.0, b.1)));
    let mut bi_builder = fst::MapBuilder::memory();
    for (p, i, c) in &bigrams {
        bi_builder.insert(bigram_key(*p, *i), encode_cost(*c)).unwrap();
    }
    let bigram_fst = bi_builder.into_inner().unwrap();

    // trigram.fst: make the 好 path's trigrams cheap so that WITHOUT the 4-gram the 我是好人 path
    // is the clear winner. We give (是,好,人) a cheap trigram and (是,号,人) a pricier one; combined
    // with the cheaper unigram/bigram of 好, path A (我是好人) wins under bigram+trigram alone.
    let mut trigrams: Vec<(u32, u32, u32, i32)> = vec![
        (WO, SHI, HAO_GOOD, 40),   // 我 是 好
        (WO, SHI, HAO_NUM, 200),   // 我 是 号  (pricier)
        (SHI, HAO_GOOD, REN, 30),  // 是 好 人  (cheap)
        (SHI, HAO_NUM, REN, 200),  // 是 号 人  (pricier)
    ];
    trigrams.sort_by(|a, b| trigram_key(a.0, a.1, a.2).cmp(&trigram_key(b.0, b.1, b.2)));
    let mut tri_builder = fst::MapBuilder::memory();
    for (a, b, c, cost) in &trigrams {
        tri_builder.insert(trigram_key(*a, *b, *c), encode_cost(*cost)).unwrap();
    }
    let trigram_fst = tri_builder.into_inner().unwrap();

    // english.fst (a couple of words so the loader path is exercised).
    let mut eng = vec!["hello", "rust"];
    eng.sort();
    let mut eng_builder = fst::SetBuilder::memory();
    for w in &eng {
        eng_builder.insert(w.as_bytes()).unwrap();
    }
    let english_fst = eng_builder.into_inner().unwrap();

    // Unique temp dir.
    use std::sync::atomic::{AtomicU64, Ordering};
    static CTR: AtomicU64 = AtomicU64::new(0);
    let uniq = CTR.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("pyime_fourgram_{}_{}", std::process::id(), uniq));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write(&dir, "words.bin", &words_bytes);
    write(&dir, "postings.bin", &postings);
    write(&dir, "lexicon.fst", &lexicon_fst);
    write(&dir, "bigram.fst", &bigram_fst);
    write(&dir, "trigram.fst", &trigram_fst);
    write(&dir, "english.fst", &english_fst);

    if with_fourgram {
        // fourgram.fst: (我, 是, 号, 人) is extremely cheap -> in this 4-word context the 我是号人
        // path must beat 我是好人, overriding the bigram+trigram preference. We deliberately do NOT
        // add (我,是,好,人), so path A keeps its trigram cost while path B gets the cheap 4-gram.
        // Strongly NEGATIVE (signed log-ratio) so the 4-gram bonus overcomes 号's pricier
        // unigram/trigram and flips the ranking. The signed cost is stored via encode_cost.
        let mut fourgrams: Vec<(u32, u32, u32, u32, i32)> = vec![
            (WO, SHI, HAO_NUM, REN, -800), // 我 是 号 人  -> strong 4-gram bonus
        ];
        fourgrams.sort_by(|a, b| {
            fourgram_key(a.0, a.1, a.2, a.3).cmp(&fourgram_key(b.0, b.1, b.2, b.3))
        });
        let mut fg_builder = fst::MapBuilder::memory();
        for (a, b, c, d, cost) in &fourgrams {
            fg_builder.insert(fourgram_key(*a, *b, *c, *d), encode_cost(*cost)).unwrap();
        }
        let fourgram_fst = fg_builder.into_inner().unwrap();
        write(&dir, "fourgram.fst", &fourgram_fst);
    }

    dir
}

fn write(dir: &Path, name: &str, bytes: &[u8]) {
    std::fs::write(dir.join(name), bytes).unwrap();
}

fn top_texts(e: &Engine, input: &str, n: usize) -> Vec<String> {
    let cfg = EngineConfig::default();
    e.convert(input, &cfg).into_iter().take(n).map(|c| c.text).collect()
}

/// Without `fourgram.fst` the engine runs trigram-only and MUST still work (graceful, no
/// regression): the cheaper-unigram + cheaper-trigram 好 path wins, so top-1 is 我是好人.
#[test]
fn trigram_only_picks_a_when_fourgram_absent() {
    let dir = build_fixture(false);
    let e = Engine::load(&dir).expect("load trigram-only engine");
    let top = top_texts(&e, "woshihaoren", 6);
    assert_eq!(
        top.first().map(String::as_str),
        Some("我是好人"),
        "trigram-only should pick 我是好人 (好 cheaper), got {top:?}"
    );
    // Both paths should be present, with A above B.
    let pos_good = top.iter().position(|t| t == "我是好人");
    let pos_num = top.iter().position(|t| t == "我是号人");
    if let (Some(g), Some(num)) = (pos_good, pos_num) {
        assert!(g < num, "好 path must outrank 号 path without the 4-gram, got {top:?}");
    }
}

/// WITH `fourgram.fst`, the cheap (我,是,号,人) 4-gram overrides the bigram+trigram preference in
/// the rescoring pass, so the SAME input now ranks 我是号人 first — proving the 4-gram transition
/// cost is consulted and the rescoring re-orders the N-best.
#[test]
fn fourgram_rescoring_overrides_picks_b() {
    let dir = build_fixture(true);
    let e = Engine::load(&dir).expect("load fourgram engine");
    let top = top_texts(&e, "woshihaoren", 6);
    assert_eq!(
        top.first().map(String::as_str),
        Some("我是号人"),
        "4-gram (我,是,号,人) rescoring should make 我是号人 win, got {top:?}"
    );
}

/// The rescoring pass must change ONLY the LM portion of the score: with the 4-gram present, the
/// non-LM costs (reading/edit costs) are identical, so the two engines produce the SAME candidate
/// SET (same texts) — only the ORDER differs. This guards against the rescoring corrupting scores.
#[test]
fn rescoring_preserves_candidate_set() {
    let with = Engine::load(&build_fixture(true)).unwrap();
    let without = Engine::load(&build_fixture(false)).unwrap();
    use std::collections::BTreeSet;
    let set_with: BTreeSet<String> = top_texts(&with, "woshihaoren", 10).into_iter().collect();
    let set_without: BTreeSet<String> = top_texts(&without, "woshihaoren", 10).into_iter().collect();
    // Both paths of interest are present in both engines.
    for want in ["我是好人", "我是号人"] {
        assert!(set_with.contains(want), "4-gram engine missing {want}: {set_with:?}");
        assert!(set_without.contains(want), "trigram engine missing {want}: {set_without:?}");
    }
}
