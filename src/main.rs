use std::cmp::Ord;
use std::collections::HashMap;
use std::io::{BufRead, BufWriter, Read, Write};
use std::str::FromStr;
use std::*;

#[derive(Debug)]
enum Error {
    Unreachable(&'static str),
    Unimplemented(&'static str),
    IoError(io::Error),
    NoInput,
    InvalidEncodingDefinition {
        linenum: usize,
        line: String,
        err: EncodingDefifnitionError,
    },
}
#[derive(Debug)]
enum EncodingDefifnitionError {
    MisformattedDefinition,
    DuplicateEncodings,
    InvalidEncoding(ParseHuffCodeError),
}
use EncodingDefifnitionError::*;

// represent the Huffman encoding
#[derive(Debug, Clone)]
enum HuffTree<'a> {
    End(&'a str),
    Branch(&'a str, Box<HuffTree<'a>>),
}

impl<'a> HuffTree<'a> {
    /// Create a new HuffTree, with the given Iterator of words
    fn new<I>(words: &mut I) -> Option<HuffTree<'a>>
    where
        I: Iterator<Item = &'a str>,
    {
        fn helper<'a, I>(words_sorted_by_occurences: &mut I, rarest: HuffTree<'a>) -> HuffTree<'a>
        where
            I: Iterator<Item = &'a str>,
        {
            match words_sorted_by_occurences.next() {
                None => rarest,
                Some(rare) => helper(
                    words_sorted_by_occurences,
                    HuffTree::Branch(rare, Box::new(rarest)),
                ),
            }
        }
        // sort by occurences, from less to more
        let mut words_occurrences = count_occurrences(words)
            .into_iter()
            .collect::<Vec<(&str, usize)>>();
        words_occurrences.sort_unstable_by(|(_, v0), (_, v1)| Ord::cmp(v0, v1));
        let mut words_occurrences = words_occurrences.into_iter().map(|(word, _)| word);
        words_occurrences
            .next()
            .map(|rarest| helper(&mut words_occurrences, HuffTree::End(rarest)))
    }

    /// format Hufftree to string with each word and corresponding encoding
    /// Each word-encoding relation is newwline seperated,
    /// and each word-encoding relation is represented by tab seperated pair of word and encoding.
    fn format_encodings(&self) -> String {
        let mut result = String::new();
        let mut line_sep = false;
        for (word, code) in self.into_iter() {
            if line_sep {
                result.push_str("\n");
            };
            line_sep = true;
            result.push_str(&format!("{}\t{}", word, code));
        }
        result
    }
    /// encode words
    fn encode_words<I>(&self, words: &mut I) -> Result<Vec<HuffCode>, usize>
    where
        I: Iterator<Item = &'a str>,
    {
        let dict = self.into_iter().collect::<HashMap<&str, HuffCode>>();
        let mut result: Vec<HuffCode> = Vec::new();
        for (i, word) in words.enumerate() {
            result.push(dict.get(word).ok_or(i)?.clone());
        }
        Ok(result)
    }
}

impl<'a> IntoIterator for &'a HuffTree<'a> {
    type Item = (&'a str, HuffCode);
    type IntoIter = HuffTreeIter<'a>;
    fn into_iter(self) -> HuffTreeIter<'a> {
        HuffTreeIter::Some {
            init_length: 0,
            tail: self,
        }
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
        if !s.into_iter().all(|byte| *byte == b'0' || *byte == b'1') {
            return Err(NonBinary);
        }
        return match s.split_last() {
            None => Err(Empty),
            Some((last, init)) => Ok(HuffCode {
                init_length: if init.into_iter().all(|b| *b == b'1') {
                    init.len()
                } else {
                    return Err(InvalidBinary);
                },
                last: *last == b'1',
            }),
        };
    }
}

#[derive(Debug)]
enum HuffTreeIter<'a> {
    None,
    Some {
        init_length: usize,
        tail: &'a HuffTree<'a>,
    },
}

impl<'a> Iterator for HuffTreeIter<'a> {
    type Item = (&'a str, HuffCode);
    fn next(&mut self) -> Option<(&'a str, HuffCode)> {
        match self {
            &mut HuffTreeIter::None => None,
            &mut HuffTreeIter::Some {
                init_length,
                tail: HuffTree::End(word),
            } => {
                *self = HuffTreeIter::None;
                Some((
                    word,
                    HuffCode {
                        init_length: if init_length == 0 { 0 } else { init_length - 1 },
                        last: true,
                    },
                ))
            }
            &mut HuffTreeIter::Some {
                init_length,
                tail: HuffTree::Branch(word, rest),
            } => {
                *self = HuffTreeIter::Some {
                    init_length: init_length + 1,
                    tail: rest,
                };
                Some((
                    word,
                    HuffCode {
                        init_length: init_length,
                        last: false,
                    },
                ))
            }
        }
    }
}

/// count occurrences of each word
fn count_occurrences<'a, I>(words: &mut I) -> HashMap<&'a str, usize>
where
    I: Iterator<Item = &'a str>,
{
    let mut words_occurrences: HashMap<&str, usize> = HashMap::new();
    for word in words {
        match &words_occurrences.get(&word) {
            None => words_occurrences.insert(word, 0),
            Some(&occurrences) => words_occurrences.insert(word, occurrences + 1),
        };
    }
    words_occurrences
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
            $crate::writeln!(stdout, $($arg)*).map_err(Error::IoError)?;
        })
    }
    // abort when there is no input from stdin
    if atty::is(atty::Stream::Stdin) {
        println!("Huffman only accepts string from stdin.");
        return Err(Error::NoInput);
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
                let line = line.map_err(Error::IoError)?;
                // encoding definition part starts after and ends before empty line
                if line == "" {
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
                        return Err(Error::InvalidEncodingDefinition {
                            linenum: linenum,
                            line: line,
                            err: MisformattedDefinition,
                        });
                    }
                    Some((word, encoding)) => {
                        let _ = dict.push((
                            linenum,
                            (HuffCode::from_str(encoding).map_err(|err| {
                                Error::InvalidEncodingDefinition {
                                    linenum: linenum,
                                    line: line.clone(),
                                    err: InvalidEncoding(err),
                                }
                            }))?,
                            word.to_string(),
                        ));
                    }
                }
            }
            // convert Hashmap to HuffTree
            dict.sort_unstable_by(|(_, code0, _), (_, code1, _)| Ord::cmp(code0, code1));
            let mut dict = dict.into_iter();
            let hufftree: HuffTree = match dict.next() {
                None => {
                    return Err(Error::NoInput);
                }
                Some((
                    _,
                    HuffCode {
                        init_length: 0,
                        last: false,
                    },
                    word,
                )) => HuffTree::End(word.to_string()),
            };
            for (linenum, encoding, word) in dict {}

            for (linenum, line) in &mut lines {
                let line = line.map_err(Error::IoError)?;
                println!("{}", line);
                if line == "" {
                    println!("\nempty line!\n");
                };
            }
            Err(Error::Unimplemented("decoding"))
        }
        Mode::Encode => {
            // get words from stdin, waits until EOF
            let mut input = String::new();
            io::stdin()
                .lock()
                .read_to_string(&mut input)
                .map_err(Error::IoError)?;
            let input = input;
            let words = input.split_whitespace();
            // derive the huffman encodings of words as a tree
            let huffman_encoding: HuffTree =
                HuffTree::new(&mut words.clone()).ok_or(Error::NoInput)?;
            // print each word and corresponding encoding
            println!("{}", huffman_encoding.format_encodings());
            println!("");
            // print encoded string
            let encoded_string = {
                let mut encoded_string = String::new();
                huffman_encoding
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
