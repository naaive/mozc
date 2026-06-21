use pyime_core::{Engine, EngineConfig};
use std::path::Path;
fn main() -> anyhow::Result<()> {
    let eng = Engine::load(Path::new("data"))?;
    let cfg = EngineConfig::default();
    for inp in ["woaizhongguo","nihao","zhongguo"] {
        let c = eng.convert(inp, &cfg);
        let r = c.iter().position(|x| x.text=="我爱中国" || x.text=="你好" || x.text=="中国");
        let top: Vec<String> = c.iter().take(6).map(|x| x.text.clone()).collect();
        println!("{:>14} rank_target={:?} top={:?}", inp, r, top);
    }
    Ok(())
}
