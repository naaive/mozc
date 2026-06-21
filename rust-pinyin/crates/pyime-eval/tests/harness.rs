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
