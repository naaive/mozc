use pyime_core::{Engine, EngineConfig};
use std::path::Path;
fn main() -> anyhow::Result<()> {
    let eng = Engine::load(Path::new("data"))?;
    let cfg = EngineConfig::default();
    for inp in ["nihao","beijing","zhongguo","zongguo","woaizhongguo","bj","nh","zhonghuarenmingongheguo","wodiannao","github","wo用github","xiexie","mingtianjian","womendoushihaohaizi"] {
        let t = std::time::Instant::now();
        let c = eng.convert(inp, &cfg);
        let dt = t.elapsed().as_secs_f64()*1000.0;
        let top: Vec<String> = c.iter().take(5).map(|x| x.text.clone()).collect();
        println!("{:>26} -> {:?}  ({:.2}ms, {} cands)", inp, top, dt, c.len());
    }
    Ok(())
}
