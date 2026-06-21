//! Word unigram + bigram language model (bigram.fst, backoff to unigram).
//! CORE agent: implement.
use std::path::Path;

pub struct LanguageModel {
    // CORE agent: hold mmap'd bigram.fst.
}

impl LanguageModel {
    pub fn load(_data_dir: &Path) -> anyhow::Result<LanguageModel> {
        Ok(LanguageModel {})
    }
}
