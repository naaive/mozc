//! pyime-eval — comprehensive, quantified evaluation harness. EVAL agent: implement
//! gold-set loading + metrics (Top-1/5/10, MRR, CER, latency p50/p95, memory) bucketed by
//! scenario, plus a report table + report.json. See DESIGN.md "Evaluation system".
use pyime_core::{Engine, EngineConfig};
use std::path::Path;

pub fn run_eval(_engine: &Engine, _cfg: &EngineConfig, _gold: &Path) -> anyhow::Result<String> {
    Ok("eval: not yet implemented".to_string())
}
