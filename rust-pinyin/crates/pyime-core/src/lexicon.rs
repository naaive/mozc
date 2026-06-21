//! Lexicon loader (lexicon.fst + postings.bin + words.bin) and lattice→FST matching.
//!
//! Data contract (DESIGN.md v1):
//!   * `lexicon.fst` : `fst::Map`, key = canonical syllables joined by `'` (e.g. `ni'hao`),
//!     value = u64 offset into `postings.bin`.
//!   * `postings.bin`: at each offset, `u16 n` then `n × (u32 word_id, u16 cost)` little-endian.
//!   * `words.bin`   : rkyv-archived `Vec<WordEntry>` indexed by word id.
//!
//! Matching: we walk the FST node graph alongside the syllable lattice. Each lattice edge
//! consumes input letters and emits canonical syllable bytes that we follow through the FST,
//! inserting the `'` separator between syllables. Abbreviation edges (initial-only tokens)
//! follow the initial bytes and then any continuation up to the next separator — i.e. they
//! match *any* dictionary syllable beginning with that initial.

use crate::format::{ArchivedWordEntry, WordEntry};
use crate::segment::Edge;
use fst::raw::{Fst, Node, Output};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// A dictionary match covering a lattice path: word(s) + their costs + input span + edit cost.
#[derive(Debug, Clone)]
pub struct WordMatch {
    /// (word_id, reading-specific unigram cost) pairs sharing this reading.
    pub words: Vec<(u32, u16)>,
    /// Number of syllables (lattice edges) consumed by this reading.
    pub n_syllables: usize,
    /// Lattice letter index where this match starts.
    pub start: usize,
    /// Lattice letter index where this match ends (exclusive).
    pub end: usize,
    /// Accumulated edit cost from the lattice edges on this path.
    pub edit_cost: i32,
}

pub struct Lexicon {
    _fst_mmap: Mmap,
    fst: Fst<&'static [u8]>,
    postings: Mmap,
    _words_mmap: Mmap,
    words_bytes: &'static [u8],
    pub num_words: usize,
    /// Optional English vocabulary (fst::Set) for passthrough ranking.
    _english_mmap: Option<Mmap>,
    english: Option<Fst<&'static [u8]>>,
}

impl Lexicon {
    pub fn load(data_dir: &Path) -> anyhow::Result<Lexicon> {
        let fst_mmap = mmap(&data_dir.join("lexicon.fst"))?;
        // SAFETY: the mmap is owned by this struct and never moved/unmapped while `fst` lives.
        let fst_slice: &'static [u8] = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(&fst_mmap[..]) };
        let fst = Fst::new(fst_slice)
            .map_err(|e| anyhow::anyhow!("invalid lexicon.fst: {e}"))?;

        let postings = mmap(&data_dir.join("postings.bin"))?;

        let words_mmap = mmap(&data_dir.join("words.bin"))?;
        let words_bytes: &'static [u8] =
            unsafe { std::mem::transmute::<&[u8], &'static [u8]>(&words_mmap[..]) };
        // validate archive once
        let archived = rkyv::check_archived_root::<Vec<WordEntry>>(words_bytes)
            .map_err(|e| anyhow::anyhow!("invalid words.bin archive: {e}"))?;
        let num_words = archived.len();

        // english.fst is optional.
        let eng_path = data_dir.join("english.fst");
        let (english_mmap, english) = if eng_path.exists() {
            let m = mmap(&eng_path)?;
            let slice: &'static [u8] =
                unsafe { std::mem::transmute::<&[u8], &'static [u8]>(&m[..]) };
            let f = Fst::new(slice)
                .map_err(|e| anyhow::anyhow!("invalid english.fst: {e}"))?;
            (Some(m), Some(f))
        } else {
            (None, None)
        };

        Ok(Lexicon {
            _fst_mmap: fst_mmap,
            fst,
            postings,
            _words_mmap: words_mmap,
            words_bytes,
            num_words,
            _english_mmap: english_mmap,
            english,
        })
    }

    /// True if `w` (lowercased) is in the English vocabulary.
    pub fn is_english(&self, w: &str) -> bool {
        match &self.english {
            Some(f) => f.get(w.as_bytes()).is_some(),
            None => false,
        }
    }

    #[inline]
    fn archived_words(&self) -> &ArchivedVec {
        // Safe: validated in `load`.
        unsafe { rkyv::archived_root::<Vec<WordEntry>>(self.words_bytes) }
    }

    /// Surface string for a word id.
    pub fn surface(&self, id: u32) -> Option<String> {
        let words = self.archived_words();
        words.get(id as usize).map(|w: &ArchivedWordEntry| w.surface.as_str().to_string())
    }

    /// Global unigram cost for a word id.
    pub fn unigram_cost(&self, id: u32) -> Option<u16> {
        let words = self.archived_words();
        words.get(id as usize).map(|w: &ArchivedWordEntry| w.unigram_cost.into())
    }

    /// Read a postings list at `offset`: returns (word_id, cost) pairs.
    fn read_postings(&self, offset: u64) -> Vec<(u32, u16)> {
        let buf = &self.postings[..];
        let off = offset as usize;
        if off + 2 > buf.len() {
            return Vec::new();
        }
        let n = u16::from_le_bytes([buf[off], buf[off + 1]]) as usize;
        let mut out = Vec::with_capacity(n);
        let mut p = off + 2;
        for _ in 0..n {
            if p + 6 > buf.len() {
                break;
            }
            let id = u32::from_le_bytes([buf[p], buf[p + 1], buf[p + 2], buf[p + 3]]);
            let cost = u16::from_le_bytes([buf[p + 4], buf[p + 5]]);
            out.push((id, cost));
            p += 6;
        }
        out
    }

    /// Walk the FST from `node` following `byte`. Returns the child node and accumulated output.
    fn step<'a>(&'a self, node: &Node<'a>, byte: u8, out_acc: Output) -> Option<(Node<'a>, Output)> {
        let i = node.find_input(byte)?;
        let t = node.transition(i);
        let child = self.fst.node(t.addr);
        Some((child, out_acc.cat(t.out)))
    }

    /// Try to follow the byte string `s` from `node`, returning the resulting node + output.
    fn follow_str<'a>(
        &'a self,
        node: Node<'a>,
        s: &str,
        out_acc: Output,
    ) -> Option<(Node<'a>, Output)> {
        let mut cur = node;
        let mut acc = out_acc;
        for &b in s.as_bytes() {
            let (c, o) = self.step(&cur, b, acc)?;
            cur = c;
            acc = o;
        }
        Some((cur, acc))
    }

    /// Find all dictionary words whose canonical reading matches a path through the lattice
    /// starting at letter index `start`. Recursively walks edges, threading the FST node.
    ///
    /// `edges_from[i]` are lattice edges leaving letter index `i`. Returns one `WordMatch` per
    /// FST key (reading) reachable. Bounded by max syllables and total path expansion.
    pub fn match_from(
        &self,
        edges_from: &[Vec<Edge>],
        start: usize,
    ) -> Vec<WordMatch> {
        let root = self.fst.root();
        let mut results = Vec::new();
        let mut budget: u32 = 20_000; // node-visit budget to bound blowups
        self.walk(
            edges_from,
            start,
            start,
            root,
            Output::zero(),
            0,
            0,
            &mut results,
            &mut budget,
        );
        results
    }

    #[allow(clippy::too_many_arguments)]
    fn walk(
        &self,
        edges_from: &[Vec<Edge>],
        start: usize,
        pos: usize,
        node: Node<'_>,
        out_acc: Output,
        edit_cost: i32,
        depth: usize,
        results: &mut Vec<WordMatch>,
        budget: &mut u32,
    ) {
        if depth > 12 || *budget == 0 {
            return;
        }

        for edge in &edges_from[pos] {
            *budget = budget.saturating_sub(1);
            if *budget == 0 {
                return;
            }
            let new_edit = edit_cost + edge.cost;

            if edge.abbrev {
                // Initial-only token: follow the initial bytes, then any continuation up to the
                // next separator (or word end). Enumerate via DFS over FST transitions.
                if let Some((after_init, acc1)) = self.follow_str(node, &edge.syllable, out_acc) {
                    self.expand_abbrev(
                        edges_from,
                        start,
                        edge.end,
                        after_init,
                        acc1,
                        new_edit,
                        depth,
                        results,
                        budget,
                    );
                }
            } else {
                // Exact syllable: follow its bytes.
                if let Some((after_syl, acc1)) = self.follow_str(node, &edge.syllable, out_acc) {
                    self.emit_and_continue(
                        edges_from, start, edge.end, after_syl, acc1, new_edit, depth, results,
                        budget,
                    );
                }
            }
        }
    }

    /// After consuming a syllable (FST positioned just after its bytes), (1) if a key terminates
    /// here, emit a match; (2) follow a `'` separator and recurse for multi-word readings.
    #[allow(clippy::too_many_arguments)]
    fn emit_and_continue(
        &self,
        edges_from: &[Vec<Edge>],
        start: usize,
        pos: usize,
        node: Node<'_>,
        out_acc: Output,
        edit_cost: i32,
        depth: usize,
        results: &mut Vec<WordMatch>,
        budget: &mut u32,
    ) {
        // (1) terminal key here?
        if node.is_final() {
            let final_out = out_acc.cat(node.final_output());
            let postings = self.read_postings(final_out.value());
            if !postings.is_empty() {
                results.push(WordMatch {
                    words: postings,
                    n_syllables: 0, // (informational; not used downstream)
                    start,
                    end: pos,
                    edit_cost,
                });
            }
        }
        // (2) continue to next syllable across a `'` separator.
        if pos < edges_from.len() {
            if let Some((after_sep, acc_sep)) = self.step(&node, b'\'', out_acc) {
                self.walk(
                    edges_from, start, pos, after_sep, acc_sep, edit_cost, depth + 1, results,
                    budget,
                );
            }
        }
    }

    /// Expand an abbreviation token: from `node` (positioned after the initial), follow any
    /// number of additional reading bytes until a `'` separator or word end, treating each as a
    /// possible syllable completion. We DFS over FST transitions, stopping at `'`.
    #[allow(clippy::too_many_arguments)]
    fn expand_abbrev(
        &self,
        edges_from: &[Vec<Edge>],
        start: usize,
        pos: usize,
        node: Node<'_>,
        out_acc: Output,
        edit_cost: i32,
        depth: usize,
        results: &mut Vec<WordMatch>,
        budget: &mut u32,
    ) {
        // The current node may already complete a syllable (e.g. initial "a"/"e" cases, or single
        // letter readings). Treat node as a syllable end here too.
        self.emit_and_continue(
            edges_from, start, pos, node, out_acc, edit_cost, depth, results, budget,
        );

        // Follow every non-separator transition deeper (still the *same* abbreviated syllable).
        for ti in 0..node.len() {
            *budget = budget.saturating_sub(1);
            if *budget == 0 {
                return;
            }
            let t = node.transition(ti);
            if t.inp == b'\'' {
                continue; // separator handled inside emit_and_continue
            }
            let child = self.fst.node(t.addr);
            let acc = out_acc.cat(t.out);
            self.expand_abbrev(
                edges_from, start, pos, child, acc, edit_cost, depth, results, budget,
            );
        }
    }

    /// Exact whole-key lookup (used by tests / probing).
    pub fn lookup_exact(&self, key: &str) -> Option<Vec<(u32, u16)>> {
        let out = self.fst.get(key.as_bytes())?;
        Some(self.read_postings(out.value()))
    }
}

/// rkyv archived Vec alias for readability.
type ArchivedVec = rkyv::vec::ArchivedVec<ArchivedWordEntry>;

fn mmap(path: &Path) -> anyhow::Result<Mmap> {
    let f = File::open(path)
        .map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))?;
    // SAFETY: file is opened read-only; we treat the map as immutable.
    let m = unsafe { Mmap::map(&f) }
        .map_err(|e| anyhow::anyhow!("mmap {}: {e}", path.display()))?;
    Ok(m)
}
