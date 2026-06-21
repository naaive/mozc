//! Trigram (stupid-backoff) decoder tests.
//!
//! These build a tiny self-contained dataset (matching the on-disk format contract in DESIGN.md)
//! into a temp dir, OPTIONALLY including a `trigram.fst`, then load it via `Engine::load` and
//! assert:
//!   1. With NO `trigram.fst`, the decoder runs bigram-only and picks the bigram-best word (A).
//!   2. With a `trigram.fst` that makes a competing word (B) cheaper in the (w1,w2,_) context,
//!      the SAME input now ranks B first — proving the trigram path is actually used.
//!
//! The two engines share an identical lexicon / bigram model; only the presence of the trigram
//! file differs, so any ranking change is attributable to the trigram transition cost.

use pyime_core::format::{bigram_key, trigram_key, WordEntry};
use pyime_core::{Engine, EngineConfig};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

// Word ids (kept explicit so the bigram/trigram keys below are readable).
const WO: u32 = 0; // 我  wo
const SHI: u32 = 1; // 是 shi
const HAO_GOOD: u32 = 2; // 好 hao  (cheaper unigram -> bigram-best for reading `hao`)
const HAO_NUM: u32 = 3; // 号 hao  (pricier unigram; only a cheap TRIGRAM makes it win)

/// Build a fixture dataset. When `with_trigram` is true, a `trigram.fst` is written that makes
/// `(我, 是, 号)` very cheap so 号 should beat the otherwise-cheaper 好.
fn build_fixture(with_trigram: bool) -> PathBuf {
    // (surface, reading, unigram_cost). 好 and 号 share reading `hao`.
    let words: Vec<(&str, &str, u16)> = vec![
        ("我", "wo", 80),   // 0
        ("是", "shi", 90),  // 1
        ("好", "hao", 100), // 2  cheaper
        ("号", "hao", 400), // 3  pricier
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

    // bigram.fst: make BOTH 是->好 and 是->号 present so the bigram model has an opinion, with
    // 是->好 CHEAPER. Bigram-only must therefore prefer 好. (我->是 cheap to anchor the prefix.)
    let mut bigrams: Vec<(u32, u32, u64)> = vec![
        (WO, SHI, 50),       // 我 -> 是
        (SHI, HAO_GOOD, 60), // 是 -> 好   (cheaper)
        (SHI, HAO_NUM, 300), // 是 -> 号   (pricier)
    ];
    bigrams.sort_by(|a, b| bigram_key(a.0, a.1).cmp(&bigram_key(b.0, b.1)));
    let mut bi_builder = fst::MapBuilder::memory();
    for (p, i, c) in &bigrams {
        bi_builder.insert(bigram_key(*p, *i), *c).unwrap();
    }
    let bigram_fst = bi_builder.into_inner().unwrap();

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
    let dir = std::env::temp_dir().join(format!("pyime_trigram_{}_{}", std::process::id(), uniq));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write(&dir, "words.bin", &words_bytes);
    write(&dir, "postings.bin", &postings);
    write(&dir, "lexicon.fst", &lexicon_fst);
    write(&dir, "bigram.fst", &bigram_fst);
    write(&dir, "english.fst", &english_fst);

    if with_trigram {
        // trigram.fst: (我, 是, 号) is extremely cheap -> in this 3-word context 号 must beat 好,
        // overriding the bigram preference. We deliberately do NOT add (我,是,好), so 好 falls back
        // to TRIGRAM_BACKOFF + bigram(是->好); the cheap 号 trigram still wins.
        let mut trigrams: Vec<(u32, u32, u32, u64)> = vec![
            (WO, SHI, HAO_NUM, 1), // 我 是 号  -> basically free
        ];
        trigrams.sort_by(|a, b| {
            trigram_key(a.0, a.1, a.2).cmp(&trigram_key(b.0, b.1, b.2))
        });
        let mut tri_builder = fst::MapBuilder::memory();
        for (a, b, c, cost) in &trigrams {
            tri_builder.insert(trigram_key(*a, *b, *c), *cost).unwrap();
        }
        let trigram_fst = tri_builder.into_inner().unwrap();
        write(&dir, "trigram.fst", &trigram_fst);
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

/// Without `trigram.fst` the engine runs bigram-only and MUST still work: 是->好 is the cheaper
/// bigram, so the cheaper-unigram 好 wins; top-1 ends in 好, not 号.
#[test]
fn bigram_only_picks_a_when_trigram_absent() {
    let dir = build_fixture(false);
    let e = Engine::load(&dir).expect("load bigram-only engine");
    let top = top_texts(&e, "woshihao", 5);
    assert_eq!(
        top.first().map(String::as_str),
        Some("我是好"),
        "bigram-only should pick 我是好 (好 cheaper), got {top:?}"
    );
    // 我是号 should be present but ranked below 我是好.
    let pos_good = top.iter().position(|t| t == "我是好");
    let pos_num = top.iter().position(|t| t == "我是号");
    if let (Some(g), Some(num)) = (pos_good, pos_num) {
        assert!(g < num, "好 must outrank 号 in bigram-only mode, got {top:?}");
    }
}

/// WITH `trigram.fst`, the cheap (我,是,号) trigram overrides the bigram preference, so the SAME
/// input now ranks 我是号 first — proving the trigram transition cost is actually consulted.
#[test]
fn trigram_overrides_bigram_picks_b() {
    let dir = build_fixture(true);
    let e = Engine::load(&dir).expect("load trigram engine");
    let top = top_texts(&e, "woshihao", 5);
    assert_eq!(
        top.first().map(String::as_str),
        Some("我是号"),
        "trigram (我,是,号) should make 我是号 win, got {top:?}"
    );
}
