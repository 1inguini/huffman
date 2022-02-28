use std::cmp::Ord;
use std::collections::HashMap;
use std::io::{BufRead, BufWriter, Read, Write};
use std::str::FromStr;
use std::*;

#[derive(Debug)]
enum Error {
    /// this would never happen
    Unreachable(&'static str),

    /// just relaying io::Error
    Io(io::Error),

    /// there is no input from stdin
    NoStdin,

    /// something is wrong with encoding definition part of input
    InvalidEncodingDefinition(IsAt<EncodingDefifnitionError>),

    /// something is wrong with encoded string part of input
    InvalidCodeString(IsAt<CodeStringError>),
}
#[derive(Debug)]
struct IsAt<T> {
    /// number of lines from the top
    line: usize,
    /// number of character from the left
    character: usize,
    is: T,
}

#[derive(Debug)]
enum EncodingDefifnitionError {
    /// definition is not in "word TAB encoding" style
    MisformattedDefinition,

    /// same word has been associated with another word
    DuplicateDefinitions,

    /// same code has been associated with another word
    DuplicateEncodings,

    /// there is word missing definition, guessing from defined codes
    InsufficientDefinition,
    /// code part has something wrong
    InvalidEncoding(ParseHuffCodeError),
}
use EncodingDefifnitionError::*;

#[derive(Debug)]
enum CodeStringError {
    /// there is something other than 0s and 1s in the string
    NonBinary,

    /// string of binary is not in meaningful format
    MalformedBinary,
}

// represent the Huffman encoding
#[derive(Debug, Clone, Default)]
struct Encoding {
    /// words with encoding prefixed with index-of-vector replication of 1s and ends in 0
    common: Vec<String>,
    /// word with encoding of only 1s
    rarest: String,
}

mod huffman {

    use crate::util;

    use std::{
        collections::{HashMap, VecDeque},
        hash::Hash,
    };
    /// represents the Huffman coding for symbols
    #[derive(Debug)]
    pub struct Tree<Symbol: Ord + Hash>(Node<Symbol>);
    impl<Symbol> Tree<Symbol>
    where
        Symbol: Ord + Hash,
    {
        /// creates Huffman tree from HashMap of symbol to it's occurence
        /// returns None at empty input
        pub fn new(symbols_and_their_occurence: HashMap<Symbol, usize>) -> Option<Self> {
            if symbols_and_their_occurence.is_empty() {
                None
            } else {
                // create nodes and sort them by occurence
                let mut leaves = symbols_and_their_occurence
                    .into_iter()
                    .map(|(symbol, occurence)| Root::new(occurence, symbol))
                    .collect::<Vec<Root<Symbol>>>();
                leaves.sort_unstable_by(|r0, r1| Ord::cmp(&r0.occurence, &r1.occurence));

                // prepare two queue, one filled with sorted nodes
                let mut leaves: VecDeque<Root<Symbol>> = leaves.into();
                let mut branches: VecDeque<Root<Symbol>> = VecDeque::new();

                // pop the rarer one from the front of two queues
                while let Some(r0) = Root::pop_rarer(&mut leaves, &mut branches) {
                    match Root::pop_rarer(&mut leaves, &mut branches) {
                        None => {
                            break;
                        }
                        Some(r1) => branches.push_back(Root::merge(r0, r1)),
                    }
                }
                Some(Tree(Root::pop_rarer(&mut leaves, &mut branches)?.inner))
            }
        }
    }
    #[derive(Debug)]
    /// associates a symbol and it's occurence
    struct Root<Symbol> {
        /// occurence of the symbol in given sequence, or a sum of childrens
        occurence: usize,

        /// depth of the tree
        depth: usize,

        /// a symbol to be encoded
        inner: Node<Symbol>,
    }
    #[derive(Debug)]
    ///  used at construction of encoding
    enum Node<Symbol> {
        Symbol(Symbol),
        Merged {
            shallower: Box<Node<Symbol>>,
            deeper: Box<Node<Symbol>>,
        },
    }
    impl<Symbol> Root<Symbol> {
        fn new(occurence: usize, symbol: Symbol) -> Self {
            Root {
                occurence,
                depth: 0,
                inner: Node::Symbol(symbol),
            }
        }

        fn merge(root0: Self, root1: Self) -> Self {
            let (shallower, deeper, depth) = if root0.depth < root1.depth {
                (root0.inner, root1.inner, root1.depth)
            } else {
                (root1.inner, root0.inner, root0.depth)
            };
            Root {
                occurence: root0.occurence + root1.occurence,
                depth: depth + 1,
                inner: Node::Merged {
                    shallower: Box::new(shallower),
                    deeper: Box::new(deeper),
                },
            }
        }
        fn pop_rarer<'a>(
            leaves: &mut VecDeque<Self>,
            branches: &mut VecDeque<Self>,
        ) -> Option<Self> {
            match leaves.front() {
                None => branches.pop_front(),
                Some(leaf) => {
                    if leaf.occurence < branches.front()?.occurence {
                        leaves.pop_front()
                    } else {
                        branches.pop_front()
                    }
                }
            }
        }
    }
}

impl Encoding {
    /// Create a new HuffTree, with the given Iterator of words
    fn new<I>(words: &mut I) -> Option<Self>
    where
        I: Iterator<Item = String>,
    {
        // sort by occurences, from less to more
        let mut words_occurrences = util::count_occurrences(words)
            .into_iter()
            .collect::<Vec<(String, usize)>>();
        words_occurrences.sort_unstable_by(|(_, v0), (_, v1)| Ord::cmp(v0, v1));
        let mut words_occurrences = words_occurrences.into_iter().map(|(word, _)| word);
        words_occurrences.next_back().map(|rarest| Encoding {
            common: words_occurrences.collect(),
            rarest,
        })
    }

    /// format Hufftree to string with each word and corresponding encoding
    /// Each word-encoding relation is newwline seperated,
    /// and each word-encoding relation is represented by tab seperated pair of word and encoding.
    fn format_encodings(self) -> String {
        let mut result = String::new();
        let mut line_sep = false;
        for (word, code) in self.into_iter() {
            if line_sep {
                result.push('\n');
            };
            line_sep = true;
            result.push_str(&format!("{}\t{}", word, code));
        }
        result
    }
    /// encode words
    fn encode_words<I>(self, words: &mut I) -> Result<Vec<HuffCode>, usize>
    where
        I: Iterator<Item = String>,
    {
        let dict = self.into_iter().collect::<HashMap<String, HuffCode>>();
        let mut result: Vec<HuffCode> = Vec::new();
        for (i, word) in words.enumerate() {
            result.push(dict.get(&word).ok_or(i)?.clone());
        }
        Ok(result)
    }
}

#[derive(Debug)]
struct EncodingsIterator {
    index: usize,
    inner: Encoding,
}
impl IntoIterator for Encoding {
    type Item = (String, HuffCode);
    type IntoIter = EncodingsIterator;
    fn into_iter(self) -> Self::IntoIter {
        EncodingsIterator {
            index: 0,
            inner: self,
        }
    }
}
impl Iterator for EncodingsIterator {
    type Item = (String, HuffCode);
    fn next(&mut self) -> Option<Self::Item> {
        let result = if self.index == self.inner.common.len() {
            Some((
                mem::take(&mut self.inner.rarest),
                HuffCode {
                    init_length: if self.index < 1 { 0 } else { self.index - 1 },
                    last: true,
                },
            ))
        } else {
            match self.inner.common.get(self.index) {
                None => None,
                Some(word) => Some((
                    word.clone(),
                    HuffCode {
                        init_length: self.index,
                        last: false,
                    },
                )),
            }
        };
        self.index += 1;
        result
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct HuffCode {
    init_length: usize,
    last: bool,
}
impl fmt::Display for HuffCode {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        for _ in 0..self.init_length {
            write!(f, "1")?;
        }
        write!(f, "{}", if self.last { 1 } else { 0 })
    }
}
#[derive(Debug)]
enum ParseHuffCodeError {
    Empty,
    NonBinary,
    InvalidBinary,
}
impl str::FromStr for HuffCode {
    type Err = ParseHuffCodeError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use ParseHuffCodeError::*;
        let s = s.as_bytes();
        if !s.iter().all(|byte| *byte == b'0' || *byte == b'1') {
            return Err(NonBinary);
        }
        return match s.split_last() {
            None => Err(Empty),
            Some((last, init)) => Ok(HuffCode {
                init_length: if init.iter().all(|b| *b == b'1') {
                    init.len()
                } else {
                    return Err(InvalidBinary);
                },
                last: *last == b'1',
            }),
        };
    }
}

// options
use clap::{Parser, Subcommand};

/// represent all acceptable arguments
#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    mode: Mode,
}
#[derive(Subcommand)]
enum Mode {
    /// encodes words string
    Encode,

    /// decodes words string
    Decode,
}

fn main() -> Result<(), Error> {
    // prepare stdout with buffering
    let stdout = io::stdout();
    let mut stdout = BufWriter::new(stdout.lock());
    macro_rules! println {
        ($($arg:tt)*) => ({
            $crate::writeln!(stdout, $($arg)*).map_err(Error::Io)?;
        })
    }
    // abort when there is no input from stdin
    if atty::is(atty::Stream::Stdin) {
        println!("Huffman only accepts string from stdin.");
        return Err(Error::NoStdin);
    }

    // get arguments
    let args = Args::parse();

    // run each subcommands
    return match args.mode {
        Mode::Decode => {
            // get lines from stdin, one line at a time
            let stdin = io::stdin();
            let stdin = stdin.lock();
            let mut lines = stdin.lines().enumerate();

            // skip newlines before encoding definitions
            let mut trailing = false;

            // record encodings
            let mut dict: Vec<(usize, HuffCode, String)> = Vec::new();
            for (linenum, line) in &mut lines {
                let line = line.map_err(Error::Io)?;
                // encoding definition part starts after and ends before empty line
                if line.is_empty() {
                    if trailing {
                        break;
                    } else {
                        continue;
                    }
                } else {
                    trailing = true
                };

                // each encoding definition is tab seperated pair of word and encoding
                match line.split_once('\t') {
                    None => {
                        return Err(Error::InvalidEncodingDefinition(IsAt {
                            line: linenum,
                            character: 0,
                            is: MisformattedDefinition,
                        }));
                    }
                    Some((word, encoding)) => {
                        let word = word.to_string();
                        // check for duplicate word
                        if dict.iter().any(|(_, _, w)| w == &word) {
                            return Err(Error::InvalidEncodingDefinition(IsAt {
                                line: linenum,
                                character: 0,
                                is: DuplicateDefinitions,
                            }));
                        }
                        dict.push((
                            linenum,
                            (HuffCode::from_str(encoding).map_err(|err| {
                                Error::InvalidEncodingDefinition(IsAt {
                                    line: linenum,
                                    character: word.len() + 1,
                                    is: InvalidEncoding(err),
                                })
                            }))?,
                            word,
                        ))
                    }
                }
            }
            // validate and convert Hashmap to Encodings
            dict.sort_unstable_by(|(_, code0, _), (_, code1, _)| Ord::cmp(code0, code1));
            let mut dict = dict.into_iter();

            let mut expected: HuffCode = HuffCode {
                init_length: 0,
                last: false,
            };
            let mut encodings: Encoding = match dict.next() {
                None => {
                    return Err(Error::NoStdin);
                }
                Some((linenum, encoding, word)) => {
                    if encoding == expected {
                        Encoding {
                            common: Vec::new(),
                            rarest: word,
                        }
                    } else {
                        return Err(Error::InvalidEncodingDefinition(IsAt {
                            line: linenum,
                            character: 0,
                            is: InsufficientDefinition,
                        }));
                    }
                }
            };
            for (linenum, encoding, word) in dict {
                expected.init_length += if encoding.last { 0 } else { 1 };
                expected.last = encoding.last;
                if encoding == expected {
                    encodings.common.push(encodings.rarest);
                    encodings.rarest = word;
                } else {
                    return Err(Error::InvalidEncodingDefinition(IsAt {
                        line: linenum,
                        character: 0,
                        is: InsufficientDefinition,
                    }));
                }
            }

            // decode string
            for (linenum, line) in &mut lines {
                let line = line.map_err(Error::Io)?;
                if line.is_empty() {
                    continue;
                };
                let mut decoded: String = String::new();
                let mut ones: usize = 0;
                for (pos, &byte) in line.as_bytes().iter().enumerate() {
                    if encodings.common.len() <= ones {
                        decoded.push_str(&encodings.rarest);
                        decoded.push('\n');
                        ones = 0;
                    } else {
                        match byte {
                            b'1' => ones += 1,
                            b'0' => {
                                decoded
                                    .push_str(encodings.common.get(ones).ok_or(Error::Unreachable(
                                    "ones should have been resetted before it gets out of bounds",
                                ))?);
                                decoded.push('\n');
                                ones = 0;
                            }
                            _ => {
                                return Err(Error::InvalidCodeString(IsAt {
                                    line: linenum,
                                    character: pos,
                                    is: CodeStringError::NonBinary,
                                }));
                            }
                        }
                    }
                }

                // check trailing bits
                if 0 < ones {
                    return Err(Error::InvalidCodeString(IsAt {
                        line: linenum,
                        character: line.as_bytes().len() - ones,
                        is: CodeStringError::MalformedBinary,
                    }));
                }
                println!("{}", decoded);
            }
            Ok(())
        }
        Mode::Encode => {
            // get words from stdin, waits until EOF
            let mut input = String::new();
            io::stdin()
                .lock()
                .read_to_string(&mut input)
                .map_err(Error::Io)?;
            let words = input.split_whitespace().map(str::to_string);
            // derive the huffman encodings of words as a tree
            let huffman_encodings: Encoding =
                Encoding::new(&mut words.clone()).ok_or(Error::NoStdin)?;
            // print each word and corresponding encoding
            println!("{}", huffman_encodings.clone().format_encodings());
            println!();
            // print encoded string
            let encoded_string = {
                let mut encoded_string = String::new();
                huffman_encodings
                    .encode_words(&mut words.clone())
                    .map_err(|_| {
                        Error::Unreachable("there shouldn't be words that has no encodingXX")
                    })?
                    .into_iter()
                    .for_each(|code| encoded_string.push_str(&format!("{}", code)));
                encoded_string
            };
            println!("{}", encoded_string);
            Ok(())
        }
    };
}

mod util {
    use std::{collections::HashMap, hash::Hash};

    /// count occurrences of each word
    pub fn count_occurrences<I, T>(words: &mut I) -> HashMap<T, usize>
    where
        I: Iterator<Item = T>,
        T: Eq + Hash,
    {
        let mut words_occurrences: HashMap<T, usize> = HashMap::new();
        for word in words {
            match &words_occurrences.get(&word) {
                None => words_occurrences.insert(word, 0),
                Some(&occurrences) => words_occurrences.insert(word, occurrences + 1),
            };
        }
        words_occurrences
    }
}
