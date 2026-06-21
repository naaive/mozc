//! Generate the FAIR gold set (`data/gold_v2.jsonl`) from the held-out corpus using CORRECT word
//! readings (longest-match tokenization over `data/word_pinyin.tsv`, per-char fallback via
//! `data/hanzi_pinyin.tsv`). Does NOT touch `data/gold.jsonl`.
//!
//! Run with: `cargo run --release -p pyime-eval --example gen_goldv2`
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|p| p.join("data").join("word_pinyin.tsv").exists())
        .expect("workspace root with data/word_pinyin.tsv")
        .to_path_buf();
    let data = root.join("data");
    let corpus = root.join("corpus").join("heldout_sentences.txt");
    let word_pinyin = data.join("word_pinyin.tsv");
    let hanzi = data.join("hanzi_pinyin.tsv");
    let out = data.join("gold_v2.jsonl");

    pyime_eval::generate_gold_correct(&corpus, &word_pinyin, &hanzi, &out, 0xC0FFEE)?;
    let n = pyime_eval::gold::load(&out)?.len();
    eprintln!("wrote {} ({} cases)", out.display(), n);
    Ok(())
}
