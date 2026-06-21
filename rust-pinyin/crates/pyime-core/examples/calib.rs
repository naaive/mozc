use pyime_core::{Engine, EngineConfig};
use std::path::Path;
fn main() -> anyhow::Result<()> {
    let eng = Engine::load(Path::new("data"))?;
    let cfg = EngineConfig::default();
    for inp in ["nihao","beijing","zhongguo","github","woaizhongguo"] {
        let c = eng.convert(inp, &cfg);
        println!("== {} ==", inp);
        for cand in c.iter().take(6) {
            println!("  {:?} score={} kind={:?} nseg={}", cand.text, cand.score, cand.kind, cand.segments.len());
        }
    }
    Ok(())
}
