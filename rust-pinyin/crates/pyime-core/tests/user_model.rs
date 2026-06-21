//! Online personalization (user model) integration tests against the REAL built `data/` directory.
//!
//! Each test SKIPS gracefully when the data dir is absent (fresh checkout without the data build),
//! so `cargo test` still passes. A fresh temp file backs the user model in each test so they neither
//! interfere with each other nor leave state behind.

use pyime_core::{Engine, EngineConfig};
use std::path::{Path, PathBuf};

/// Locate the workspace `data/` directory, or `None` if the built lexicon isn't present.
fn data_dir() -> Option<PathBuf> {
    for cand in ["data", "../../data"] {
        let p = Path::new(cand);
        if p.join("lexicon.fst").exists() {
            return Some(p.to_path_buf());
        }
    }
    None
}

fn engine() -> Option<Engine> {
    Some(Engine::load(&data_dir()?).expect("load real data engine"))
}

/// Engine with a user model backed by a unique temp file (created lazily on save).
fn engine_with_user(tag: &str) -> Option<(Engine, PathBuf)> {
    let e = engine()?;
    let path = std::env::temp_dir().join(format!(
        "pyime_user_{}_{}_{}.json",
        std::process::id(),
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_file(&path);
    let e = e.with_user_model(Some(path.clone()));
    Some((e, path))
}

fn top_texts(e: &Engine, input: &str, cfg: &EngineConfig, n: usize) -> Vec<String> {
    e.convert(input, cfg).into_iter().take(n).map(|c| c.text).collect()
}

fn rank_of(e: &Engine, input: &str, cfg: &EngineConfig, want: &str) -> Option<usize> {
    e.convert(input, cfg).iter().position(|c| c.text == want)
}

/// ADAPTATION: an input whose default top-1 is X but a valid candidate Y exists lower in the list;
/// committing Y a few times must promote it to #1 (and at least strictly above its original rank).
#[test]
fn adaptation_promotes_committed_candidate() {
    let Some((e, _path)) = engine_with_user("adapt") else {
        eprintln!("skip adaptation_promotes_committed_candidate: no data/");
        return;
    };
    let cfg = EngineConfig::default();

    // `beijing` defaults to 北京 #1, with 背景 a valid lower-ranked candidate.
    let input = "beijing";
    let target = "背景";
    let before = top_texts(&e, input, &cfg, 8);
    let before_rank = rank_of(&e, input, &cfg, target);
    assert_eq!(before.first().map(String::as_str), Some("北京"), "default top1 {before:?}");
    assert!(
        before_rank.map(|r| r >= 1).unwrap_or(false),
        "{target} must exist below #1 by default, before={before:?}"
    );

    // The user repeatedly commits 背景 for this input.
    for _ in 0..3 {
        e.commit(input, target);
    }

    let after = top_texts(&e, input, &cfg, 8);
    let after_rank = rank_of(&e, input, &cfg, target);
    assert_eq!(
        after.first().map(String::as_str),
        Some(target),
        "after committing {target} it should be #1, after={after:?}"
    );
    assert!(
        after_rank < before_rank,
        "{target} rank must strictly improve: {before_rank:?} -> {after_rank:?}"
    );
}

/// AUTO-LEARNED PHRASE / NEW WORD: commit an input → a surface that is NOT a normal top candidate
/// (here a name/neologism the lattice never produces as #1); it must appear near #1 next time, and a
/// FRESH engine with no user file must NOT show it.
#[test]
fn auto_learned_phrase_resurfaces() {
    let Some((e, path)) = engine_with_user("phrase") else {
        eprintln!("skip auto_learned_phrase_resurfaces: no data/");
        return;
    };
    let cfg = EngineConfig::default();

    // A made-up multi-char surface that the engine would never produce as the top candidate.
    let input = "wodemingzi";
    let learned = "我的名字叫小赵同学"; // an auto-learned unit (phrase)
    let before = top_texts(&e, input, &cfg, 10);
    assert!(
        !before.iter().any(|t| t == learned),
        "the learned phrase must not appear before committing, before={before:?}"
    );

    e.commit(input, learned);

    let after = top_texts(&e, input, &cfg, 3);
    assert_eq!(
        after.first().map(String::as_str),
        Some(learned),
        "auto-learned phrase should rank #1 on re-entry, after={after:?}"
    );

    // A FRESH engine that never loaded this user file must not show the learned phrase.
    let fresh = engine().unwrap(); // no user model attached at all
    let fresh_top = top_texts(&fresh, input, &cfg, 10);
    assert!(
        !fresh_top.iter().any(|t| t == learned),
        "a fresh engine (no user file) must not surface the learned phrase, got {fresh_top:?}"
    );
    let _ = std::fs::remove_file(&path);
}

/// PERSISTENCE: commit, `save_user`, reload via `with_user_model(same path)`, and the learned
/// ranking must persist into the new engine.
#[test]
fn persistence_round_trip() {
    let Some((e, path)) = engine_with_user("persist") else {
        eprintln!("skip persistence_round_trip: no data/");
        return;
    };
    let cfg = EngineConfig::default();

    let input = "shijian";
    let target = "事件"; // default #1 is 时间; 事件 is a valid lower candidate
    assert_eq!(
        top_texts(&e, input, &cfg, 1).first().map(String::as_str),
        Some("时间"),
        "default top1 must be 时间"
    );
    for _ in 0..3 {
        e.commit(input, target);
    }
    assert_eq!(
        top_texts(&e, input, &cfg, 1).first().map(String::as_str),
        Some(target),
        "{target} should be #1 after commits"
    );
    e.save_user().expect("save user model");
    drop(e);

    // Reload a brand-new engine attaching the SAME user file: the learned ranking must persist.
    let e2 = engine().unwrap().with_user_model(Some(path.clone()));
    assert_eq!(
        top_texts(&e2, input, &cfg, 1).first().map(String::as_str),
        Some(target),
        "learned ranking must persist across reload"
    );
    let _ = std::fs::remove_file(&path);
}

/// NO-OP SAFETY: with NO user model attached, OR with `user_weight = 0`, the results are
/// byte-identical to the plain engine for showcase cases.
#[test]
fn no_user_model_is_byte_identical() {
    let Some(plain) = engine() else {
        eprintln!("skip no_user_model_is_byte_identical: no data/");
        return;
    };
    let cfg = EngineConfig::default();

    // Baseline (no user model attached).
    let cases = ["nihao", "zhongguo", "github", "beijing", "shijian", "wo用github"];
    let baseline: Vec<Vec<(String, f32)>> = cases
        .iter()
        .map(|inp| plain.convert(inp, &cfg).into_iter().map(|c| (c.text, c.score)).collect())
        .collect();

    // (1) Engine WITH an (empty) user model attached + default weight: must be identical.
    let with_empty = engine().unwrap().with_user_model(None);
    for (inp, base) in cases.iter().zip(&baseline) {
        let got: Vec<(String, f32)> =
            with_empty.convert(inp, &cfg).into_iter().map(|c| (c.text, c.score)).collect();
        assert_eq!(&got, base, "empty user model changed results for {inp}");
    }

    // (2) Even after committing, `user_weight = 0` must reproduce the baseline exactly.
    let mut cfg0 = EngineConfig::default();
    cfg0.user_weight = 0;
    let primed = engine().unwrap().with_user_model(None);
    primed.commit("beijing", "背景");
    primed.commit("shijian", "事件");
    primed.commit("nihao", "你好");
    for (inp, base) in cases.iter().zip(&baseline) {
        let got: Vec<(String, f32)> =
            primed.convert(inp, &cfg0).into_iter().map(|c| (c.text, c.score)).collect();
        assert_eq!(&got, base, "user_weight=0 changed results for {inp}");
    }

    // And the canonical showcase top-1s are intact under the empty model.
    assert_eq!(top_texts(&with_empty, "nihao", &cfg, 1), vec!["你好".to_string()]);
    assert_eq!(top_texts(&with_empty, "zhongguo", &cfg, 1), vec!["中国".to_string()]);
    assert_eq!(top_texts(&with_empty, "github", &cfg, 1), vec!["github".to_string()]);
}
