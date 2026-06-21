//! Lexicon loader (lexicon.fst + postings.bin + words.bin) and PinyinAutomaton.
//! CORE agent: implement against DESIGN.md data contract.
use std::path::Path;

pub struct Lexicon {
    // CORE agent: hold mmap'd fst::Map, postings, words.
}

impl Lexicon {
    pub fn load(_data_dir: &Path) -> anyhow::Result<Lexicon> {
        Ok(Lexicon {})
    }
}
