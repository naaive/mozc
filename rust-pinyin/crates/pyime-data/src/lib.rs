//! pyime-data — data build pipeline. DATA agent: implement download + normalization +
//! emit data/* per DESIGN.md contract. Re-export a `build_all(out_dir, corpus_dir)` entry.
pub fn build_all(_out_dir: &std::path::Path, _corpus_dir: &std::path::Path) -> anyhow::Result<()> {
    anyhow::bail!("not yet implemented")
}
