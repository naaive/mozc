//! Canonical pinyin syllable table + max-munch segmentation primitives.
//!
//! The canonical inventory is standard Hanyu Pinyin without tone marks, with `ü` written
//! as `v`. We also treat standalone initials (b, p, m, ... zh, ch, sh) as valid *abbreviation
//! tokens* so that simplified-pinyin input (`bj` → 北京) can be segmented; these are flagged
//! separately so the decoder can apply an abbreviation penalty and an "initial-match" lexicon
//! lookup rather than an exact reading match.

use rustc_hash::FxHashSet;
use std::sync::OnceLock;

/// The ~410 canonical Hanyu Pinyin syllables (no tones, ü → v).
pub const SYLLABLES: &[&str] = &[
    // a / o / e group
    "a", "ai", "an", "ang", "ao",
    "o", "ou",
    "e", "ei", "en", "eng", "er",
    // b
    "ba", "bo", "bai", "bei", "bao", "ban", "ben", "bang", "beng", "bi", "bie", "biao",
    "bian", "bin", "bing", "bu",
    // p
    "pa", "po", "pai", "pei", "pao", "pou", "pan", "pen", "pang", "peng", "pi", "pie", "piao",
    "pian", "pin", "ping", "pu",
    // m
    "ma", "mo", "me", "mai", "mei", "mao", "mou", "man", "men", "mang", "meng", "mi", "mie",
    "miao", "miu", "mian", "min", "ming", "mu",
    // f
    "fa", "fo", "fei", "fou", "fan", "fen", "fang", "feng", "fu",
    // d
    "da", "de", "dai", "dei", "dao", "dou", "dan", "den", "dang", "deng", "dong", "di", "die",
    "diao", "diu", "dian", "ding", "du", "duo", "dui", "duan", "dun",
    // t
    "ta", "te", "tai", "tao", "tou", "tan", "tang", "teng", "tong", "ti", "tie", "tiao", "tian",
    "ting", "tu", "tuo", "tui", "tuan", "tun",
    // n
    "na", "ne", "nai", "nei", "nao", "nou", "nan", "nen", "nang", "neng", "nong", "ni", "nie",
    "niao", "niu", "nian", "nin", "niang", "ning", "nu", "nuo", "nuan", "nv", "nve",
    // l
    "la", "le", "lo", "lai", "lei", "lao", "lou", "lan", "lang", "leng", "long", "li", "lia",
    "lie", "liao", "liu", "lian", "lin", "liang", "ling", "lu", "luo", "luan", "lun", "lv",
    "lve", "lue",
    // g
    "ga", "ge", "gai", "gei", "gao", "gou", "gan", "gen", "gang", "geng", "gong", "gu", "gua",
    "guo", "guai", "gui", "guan", "gun", "guang",
    // k
    "ka", "ke", "kai", "kei", "kao", "kou", "kan", "ken", "kang", "keng", "kong", "ku", "kua",
    "kuo", "kuai", "kui", "kuan", "kun", "kuang",
    // h
    "ha", "he", "hai", "hei", "hao", "hou", "han", "hen", "hang", "heng", "hong", "hu", "hua",
    "huo", "huai", "hui", "huan", "hun", "huang",
    // j
    "ji", "jia", "jie", "jiao", "jiu", "jian", "jin", "jiang", "jing", "jiong", "ju", "jue",
    "juan", "jun",
    // q
    "qi", "qia", "qie", "qiao", "qiu", "qian", "qin", "qiang", "qing", "qiong", "qu", "que",
    "quan", "qun",
    // x
    "xi", "xia", "xie", "xiao", "xiu", "xian", "xin", "xiang", "xing", "xiong", "xu", "xue",
    "xuan", "xun",
    // zh
    "zha", "zhe", "zhi", "zhai", "zhei", "zhao", "zhou", "zhan", "zhen", "zhang", "zheng",
    "zhong", "zhu", "zhua", "zhuo", "zhuai", "zhui", "zhuan", "zhun", "zhuang",
    // ch
    "cha", "che", "chi", "chai", "chao", "chou", "chan", "chen", "chang", "cheng", "chong",
    "chu", "chua", "chuo", "chuai", "chui", "chuan", "chun", "chuang",
    // sh
    "sha", "she", "shi", "shai", "shei", "shao", "shou", "shan", "shen", "shang", "sheng",
    "shu", "shua", "shuo", "shuai", "shui", "shuan", "shun", "shuang",
    // r
    "re", "ri", "rao", "rou", "ran", "ren", "rang", "reng", "rong", "ru", "rua", "ruo", "rui",
    "ruan", "run",
    // z
    "za", "ze", "zi", "zai", "zei", "zao", "zou", "zan", "zen", "zang", "zeng", "zong", "zu",
    "zuo", "zui", "zuan", "zun",
    // c
    "ca", "ce", "ci", "cai", "cao", "cou", "can", "cen", "cang", "ceng", "cong", "cu", "cuo",
    "cui", "cuan", "cun",
    // s
    "sa", "se", "si", "sai", "sao", "sou", "san", "sen", "sang", "seng", "song", "su", "suo",
    "sui", "suan", "sun",
    // y
    "ya", "yo", "ye", "yai", "yao", "you", "yan", "yang", "yin", "ying", "yong",
    "yu", "yue", "yuan", "yun", "yi",
    // w
    "wa", "wo", "wai", "wei", "wan", "wen", "wang", "weng", "wu",
];

/// Standalone initials usable as abbreviation tokens.
pub const INITIALS: &[&str] = &[
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "zh", "ch", "sh", "r",
    "z", "c", "s", "y", "w",
];

fn syllable_set() -> &'static FxHashSet<&'static str> {
    static SET: OnceLock<FxHashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| SYLLABLES.iter().copied().collect())
}

fn initial_set() -> &'static FxHashSet<&'static str> {
    static SET: OnceLock<FxHashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| INITIALS.iter().copied().collect())
}

/// True if `s` is an exact canonical syllable.
#[inline]
pub fn is_syllable(s: &str) -> bool {
    syllable_set().contains(s)
}

/// True if `s` is a standalone initial (abbreviation token).
#[inline]
pub fn is_initial(s: &str) -> bool {
    initial_set().contains(s)
}

/// Maximum canonical syllable length (in bytes/ascii chars).
pub const MAX_SYL_LEN: usize = 6; // e.g. "zhuang", "shuang"

/// Enumerate every way to split a *prefix* of `s` (starting at byte 0) into a single valid
/// syllable. Returns `(syllable_str_slice, consumed_len)` for each, longest-first (max munch
/// ordering preserved by descending length).
///
/// `s` must be lowercase ASCII letters only (caller normalizes / splits on separators).
pub fn prefix_syllables(s: &str) -> Vec<(&str, usize)> {
    let bytes = s.as_bytes();
    let n = bytes.len();
    let mut out = Vec::new();
    let max = MAX_SYL_LEN.min(n);
    // longest first for max-munch preference
    for len in (1..=max).rev() {
        let cand = &s[..len];
        if is_syllable(cand) {
            out.push((cand, len));
        }
    }
    out
}

/// Enumerate prefix initials (1- or 2-char) of `s` for abbreviation tokens.
/// Returns `(initial_str, consumed_len)`. 2-char initials (zh/ch/sh) preferred (longest first).
pub fn prefix_initials(s: &str) -> Vec<(&str, usize)> {
    let mut out = Vec::new();
    let n = s.len();
    if n >= 2 {
        let two = &s[..2];
        if is_initial(two) {
            out.push((two, 2));
        }
    }
    if n >= 1 {
        let one = &s[..1];
        if is_initial(one) {
            out.push((one, 1));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_syllables() {
        assert!(is_syllable("ni"));
        assert!(is_syllable("hao"));
        assert!(is_syllable("zhong"));
        assert!(is_syllable("wo"));
        assert!(is_syllable("yong"));
        assert!(is_syllable("wen"));
        assert!(!is_syllable("xyz"));
    }

    #[test]
    fn max_munch_prefix() {
        // "nihao" -> first syllable options: "ni" (and that's it, "nih" not valid)
        let p = prefix_syllables("nihao");
        assert!(p.iter().any(|(s, l)| *s == "ni" && *l == 2));
        // "xian" should munch the whole thing AND offer "xi"
        let p = prefix_syllables("xian");
        assert!(p.iter().any(|(s, _)| *s == "xian"));
        assert!(p.iter().any(|(s, _)| *s == "xi"));
        // longest-first
        assert_eq!(p[0].0, "xian");
    }

    #[test]
    fn initials_work() {
        assert!(is_initial("b"));
        assert!(is_initial("zh"));
        let p = prefix_initials("zhang");
        assert_eq!(p[0], ("zh", 2));
    }
}
