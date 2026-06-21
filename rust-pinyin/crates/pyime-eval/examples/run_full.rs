//! Throwaway driver: generate the full gold set, run eval against real data, print the table,
//! and write report.json. Run with `cargo run --release -p pyime-eval --example run_full`.
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|p| p.join("data").join("hanzi_pinyin.tsv").exists())
        .expect("workspace root with data/")
        .to_path_buf();
    let data = root.join("data");
    let corpus = root.join("corpus").join("heldout_sentences.txt");
    let hanzi = data.join("hanzi_pinyin.tsv");
    let gold = data.join("gold.jsonl");

    pyime_eval::generate_gold(&corpus, &hanzi, &gold, 0xC0FFEE)?;
    let n = pyime_eval::gold::load(&gold)?.len();
    eprintln!("gold cases: {n}");

    let engine = pyime_core::Engine::load(&data)?;
    let cfg = pyime_core::EngineConfig::default();
    let report = pyime_eval::run_eval(&engine, &cfg, &gold)?;

    println!("{}", pyime_eval::render_table(&report));
    std::fs::write(root.join("report.json"), serde_json::to_string_pretty(&report)?)?;
    eprintln!("wrote {}", root.join("report.json").display());
    Ok(())
}
