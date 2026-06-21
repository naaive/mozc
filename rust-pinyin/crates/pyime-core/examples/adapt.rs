use pyime_core::{Engine, EngineConfig};
use std::path::{Path, PathBuf};
fn top(e:&Engine,i:&str,c:&EngineConfig,n:usize)->Vec<String>{e.convert(i,c).into_iter().take(n).map(|x|x.text).collect()}
fn main()->anyhow::Result<()>{
    let up = PathBuf::from("/tmp/pyime_user_demo.json"); let _=std::fs::remove_file(&up);
    let e = Engine::load(Path::new("data"))?.with_user_model(Some(up.clone()));
    let c = EngineConfig::default();
    println!("BEFORE beijing    : {:?}", top(&e,"beijing",&c,4));
    println!("BEFORE wodemingzi : {:?}", top(&e,"wodemingzi",&c,3));
    for _ in 0..3 { e.commit("beijing","背景"); }
    e.commit("wodemingzi","小明同学");   // 自学新词
    e.save_user()?;
    println!("AFTER  beijing    : {:?}", top(&e,"beijing",&c,4));
    println!("AFTER  wodemingzi : {:?}", top(&e,"wodemingzi",&c,3));
    // reload from disk -> persisted?
    let e2 = Engine::load(Path::new("data"))?.with_user_model(Some(up));
    println!("RELOAD beijing    : {:?}", top(&e2,"beijing",&c,4));
    Ok(())
}
