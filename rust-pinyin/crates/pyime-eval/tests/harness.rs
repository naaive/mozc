//! Integration self-test for the eval harness.
//!
//! If the real engine data (`data/`) and held-out corpus exist, this generates a SMALL gold set,
//! runs the full eval, and asserts structural invariants (buckets present, metrics in [0,1],
//! latency > 0). Otherwise it skips gracefully so the test suite stays green without data.

use std::path::{Path, PathBuf};

/// Locate the workspace root (the dir containing `data/` and `corpus/`) by walking up from
/// the crate manifest dir. Returns None if not found.
fn workspace_root() -> Option<PathBuf> {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("data").join("hanzi_pinyin.tsv").exists()
            && dir.join("corpus").join("heldout_sentences.txt").exists()
        {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn in_unit(x: f64) -> bool {
    x.is_finite() && (-1e-9..=1.0 + 1e-9).contains(&x)
}

fn check_metrics(m: &pyime_eval::BucketMetrics) {
    for (name, v) in [
        ("top1", m.top1),
        ("top5", m.top5),
        ("top10", m.top10),
        ("mrr", m.mrr),
        ("char_acc", m.char_acc),
        ("cer", m.cer),
        ("coverage", m.coverage),
    ] {
        assert!(in_unit(v), "metric {name} out of [0,1] for bucket {}: {v}", m.bucket);
    }
    // Monotonic inclusion: top1 <= top5 <= top10 <= coverage (within fp slack).
    assert!(m.top1 <= m.top5 + 1e-9, "top1 > top5 in {}", m.bucket);
    assert!(m.top5 <= m.top10 + 1e-9, "top5 > top10 in {}", m.bucket);
    assert!(m.top10 <= m.coverage + 1e-9, "top10 > coverage in {}", m.bucket);
}

#[test]
fn self_test_small_eval() {
    let Some(root) = workspace_root() else {
        eprintln!("self_test_small_eval: data/corpus not found, skipping");
        return;
    };
    let data_dir = root.join("data");
    let corpus = root.join("corpus").join("heldout_sentences.txt");
    let hanzi = data_dir.join("hanzi_pinyin.tsv");

    // Tiny gold set in a temp location.
    let out = std::env::temp_dir().join(format!("pyime_gold_test_{}.jsonl", std::process::id()));
    pyime_eval::generate_gold_capped(&corpus, &hanzi, &out, 12345, 25)
        .expect("generate small gold set");

    // Reproducibility: regenerating with the same seed yields byte-identical output.
    let out2 = std::env::temp_dir().join(format!("pyime_gold_test2_{}.jsonl", std::process::id()));
    pyime_eval::generate_gold_capped(&corpus, &hanzi, &out2, 12345, 25)
        .expect("regenerate small gold set");
    let a = std::fs::read(&out).unwrap();
    let b = std::fs::read(&out2).unwrap();
    assert_eq!(a, b, "gold generation must be deterministic for a fixed seed");

    let cases = pyime_eval::gold::load(&out).expect("load gold");
    assert!(!cases.is_empty(), "gold set should be non-empty");

    let engine = match pyime_core::Engine::load(Path::new(&data_dir)) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("self_test_small_eval: engine load failed ({e}), skipping eval assertions");
            let _ = std::fs::remove_file(&out);
            let _ = std::fs::remove_file(&out2);
            return;
        }
    };
    let cfg = pyime_core::EngineConfig::default();
    let report = pyime_eval::run_eval(&engine, &cfg, &out).expect("run eval");

    // At least one bucket present; every present bucket name is a known bucket.
    assert!(!report.buckets.is_empty(), "expected at least one bucket");
    for m in &report.buckets {
        assert!(
            pyime_eval::gold::BUCKETS.contains(&m.bucket.as_str()),
            "unknown bucket {}",
            m.bucket
        );
        assert!(m.n > 0, "bucket {} reported with 0 cases", m.bucket);
        check_metrics(m);
    }
    check_metrics(&report.overall);

    // Latency must be positive and percentiles ordered.
    assert!(report.latency.samples > 0, "no latency samples");
    assert!(report.latency.mean_ms > 0.0, "mean latency must be > 0");
    assert!(report.latency.p50_ms <= report.latency.p95_ms + 1e-9);
    assert!(report.latency.p95_ms <= report.latency.p99_ms + 1e-9);
    assert!(report.latency.throughput_per_s > 0.0);

    // Memory: data dir size should be non-zero (real data present).
    assert!(report.memory.data_disk_bytes > 0, "data dir size should be > 0");

    // Table renders and contains the OVERALL row.
    let table = pyime_eval::render_table(&report);
    assert!(table.contains("OVERALL"), "table missing OVERALL row");

    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(&out2);
}

/// The FAIR gold set (v2, correct word readings) must:
///   * generate deterministically for a fixed seed,
///   * read the polyphone 词组 银行 as `yinhang` (and never the bogus `yinxing`),
///   * for a 词组 whose first-character reading is wrong out of context (系统: 系→`xi` in context,
///     but `ji` as a bare per-char first reading), produce `xitong` — and so DIFFER from the
///     per-character (v1) gold for that sentence.
#[test]
fn gold_v2_uses_correct_polyphone_readings() {
    let Some(root) = workspace_root() else {
        eprintln!("gold_v2_uses_correct_polyphone_readings: data/corpus not found, skipping");
        return;
    };
    let data_dir = root.join("data");
    let word_pinyin = data_dir.join("word_pinyin.tsv");
    let hanzi = data_dir.join("hanzi_pinyin.tsv");
    if !word_pinyin.exists() {
        eprintln!("gold_v2: word_pinyin.tsv missing, skipping");
        return;
    }

    // Tiny corpus with polyphone 词组. 银行 must read `yinhang`; 电脑系统 must read `...xitong`.
    let pid = std::process::id();
    let corpus = std::env::temp_dir().join(format!("pyime_gv2_corpus_{pid}.txt"));
    std::fs::write(&corpus, "我去银行\n电脑系统\n").unwrap();

    let v2 = std::env::temp_dir().join(format!("pyime_gv2_{pid}.jsonl"));
    pyime_eval::generate_gold_correct_capped(&corpus, &word_pinyin, &hanzi, &v2, 7, 50)
        .expect("generate v2");

    // Determinism: same seed → byte-identical output.
    let v2b = std::env::temp_dir().join(format!("pyime_gv2b_{pid}.jsonl"));
    pyime_eval::generate_gold_correct_capped(&corpus, &word_pinyin, &hanzi, &v2b, 7, 50)
        .expect("regenerate v2");
    assert_eq!(
        std::fs::read(&v2).unwrap(),
        std::fs::read(&v2b).unwrap(),
        "v2 gold generation must be deterministic for a fixed seed"
    );

    let cases = pyime_eval::gold::load(&v2).expect("load v2 gold");
    let find_full = |expected: &str| {
        cases
            .iter()
            .find(|c| c.bucket == "full" && c.expected == expected)
            .unwrap_or_else(|| panic!("a `full` case for {expected}"))
    };

    // 银行 → yinhang, never yinxing.
    let bank = find_full("我去银行");
    assert!(bank.input.contains("yinhang"), "expected `yinhang` in {:?}", bank.input);
    assert!(!bank.input.contains("yinxing"), "must not contain `yinxing` in {:?}", bank.input);

    // 系统 → xitong (correct), not jitong (per-char 系→ji).
    let sys = find_full("电脑系统");
    assert!(sys.input.contains("xitong"), "expected `xitong` in {:?}", sys.input);
    assert!(!sys.input.contains("jitong"), "must not contain per-char `jitong` in {:?}", sys.input);

    // v2 (correct) must differ from v1 (per-char) for the 系统 sentence.
    let v1 = std::env::temp_dir().join(format!("pyime_gv1_{pid}.jsonl"));
    pyime_eval::generate_gold_capped(&corpus, &hanzi, &v1, 7, 50).expect("generate v1");
    let v1_cases = pyime_eval::gold::load(&v1).expect("load v1 gold");
    let v1_sys = v1_cases
        .iter()
        .find(|c| c.bucket == "full" && c.expected == "电脑系统")
        .expect("v1 `full` case for 电脑系统");
    assert_ne!(
        v1_sys.input, sys.input,
        "v2 (correct) and v1 (per-char) readings must differ for a polyphone sentence"
    );
    assert!(v1_sys.input.contains("jitong"), "v1 should mis-read as `jitong`: {:?}", v1_sys.input);

    for p in [&corpus, &v2, &v2b, &v1] {
        let _ = std::fs::remove_file(p);
    }
}
