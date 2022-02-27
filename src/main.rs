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
    IoError(io::Error),

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
#[derive(Debug, Clone)]
struct Encodings {
    /// words with encoding prefixed with index-of-vector replication of 1s and ends in 0
    common: Vec<String>,
    /// word with encoding of only 1s
    rarest: String,
}

impl Encodings {
    /// Create a new HuffTree, with the given Iterator of words
    fn new<I>(words: &mut I) -> Option<Self>
    where
        I: Iterator<Item = String>,
    {
        // sort by occurences, from less to more
        let mut words_occurrences = count_occurrences(words)
            .into_iter()
            .collect::<Vec<(String, usize)>>();
        words_occurrences.sort_unstable_by(|(_, v0), (_, v1)| Ord::cmp(v0, v1));
        let mut words_occurrences = words_occurrences.into_iter().map(|(word, _)| word);
        words_occurrences.next_back().map(|rarest| Encodings {
            common: words_occurrences.collect(),
            rarest: rarest,
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
                result.push_str("\n");
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
impl Default for Encodings {
    fn default() -> Self {
        Encodings {
            common: Vec::new(),
            rarest: String::new(),
        }
    }
}

#[derive(Debug)]
struct EncodingsIterator {
    index: usize,
    inner: Encodings,
}
impl IntoIterator for Encodings {
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

/// count occurrences of each word
fn count_occurrences<I>(words: &mut I) -> HashMap<String, usize>
where
    I: Iterator<Item = String>,
{
    let mut words_occurrences: HashMap<String, usize> = HashMap::new();
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
            let mut encodings: Encodings = match dict.next() {
                None => {
                    return Err(Error::NoStdin);
                }
                Some((linenum, encoding, word)) => {
                    if encoding == expected {
                        Encodings {
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
                let line = line.map_err(Error::IoError)?;
                if line == "" {
                    continue;
                };
                let mut decoded: String = String::new();
                let mut ones: usize = 0;
                for (pos, &byte) in line.as_bytes().into_iter().enumerate() {
                    if encodings.common.len() <= ones {
                        decoded.push_str(&encodings.rarest);
                        decoded.push_str("\n");
                        ones = 0;
                    } else {
                        match byte {
                            b'1' => ones += 1,
                            b'0' => {
                                decoded
                                    .push_str(encodings.common.get(ones).ok_or(Error::Unreachable(
                                    "ones should have been resetted before it gets out of bounds",
                                ))?);
                                decoded.push_str("\n");
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
                .map_err(Error::IoError)?;
            let words = input.split_whitespace().map(str::to_string);
            // derive the huffman encodings of words as a tree
            let huffman_encodings: Encodings =
                Encodings::new(&mut words.clone()).ok_or(Error::NoStdin)?;
            // print each word and corresponding encoding
            println!("{}", huffman_encodings.clone().format_encodings());
            println!("");
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
