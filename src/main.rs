use std::cmp::Ord;
use std::collections::HashMap;
use std::io::Read;
use std::*;

#[derive(Debug)]
enum Error {
    Unreachable(&'static str),
    IoError(io::Error),
    NoInput,
}

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
        words_occurrences.sort_by(|(_, v0), (_, v1)| Ord::cmp(v0, v1));
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
            prefix_length: 0,
            tail: self,
        }
    }
}

#[derive(Clone, Debug)]
struct HuffCode {
    prefix_length: usize,
    end: bool,
}
impl fmt::Display for HuffCode {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        for _ in 0..self.prefix_length {
            write!(f, "1")?;
        }
        write!(f, "{}", if self.end { 1 } else { 0 })
    }
}

#[derive(Debug)]
enum HuffTreeIter<'a> {
    None,
    Some {
        prefix_length: usize,
        tail: &'a HuffTree<'a>,
    },
}

impl<'a> Iterator for HuffTreeIter<'a> {
    type Item = (&'a str, HuffCode);
    fn next(&mut self) -> Option<(&'a str, HuffCode)> {
        match self {
            &mut HuffTreeIter::None => None,
            &mut HuffTreeIter::Some {
                prefix_length,
                tail: HuffTree::End(word),
            } => {
                *self = HuffTreeIter::None;
                Some((
                    word,
                    HuffCode {
                        prefix_length: if prefix_length == 0 {
                            0
                        } else {
                            prefix_length - 1
                        },
                        end: true,
                    },
                ))
            }
            &mut HuffTreeIter::Some {
                prefix_length,
                tail: HuffTree::Branch(word, rest),
            } => {
                *self = HuffTreeIter::Some {
                    prefix_length: prefix_length + 1,
                    tail: rest,
                };
                Some((
                    word,
                    HuffCode {
                        prefix_length: prefix_length,
                        end: false,
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

fn main() -> Result<(), Error> {
    // options
    use clap::{Parser, Subcommand};

    #[derive(Parser)]
    #[clap(author, version, about, long_about = None)]
    struct Cli {
        #[clap(subcommand)]
        mode: Mode,
    }
    #[derive(Subcommand)]
    enum Mode {
        /// encodes words string
        Encode,

        /// encodes words string
        Decode,
    }
    let args = Cli::parse();
    match args.mode {
        Mode::Decode => println!("Decoding is not implemented yet."),
        Mode::Encode => (),
    }

    if atty::is(atty::Stream::Stdin) {
        println!("Huffman only accepts string from stdin.");
        return Err(Error::NoInput);
    }

    // get words from stdin
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(Error::IoError)?;
    let words = input.split_whitespace();

    // derive the huffman encodings of words as a tree
    let huffman_encoding: HuffTree = HuffTree::new(&mut words.clone()).ok_or(Error::NoInput)?;

    // print each word and corresponding encoding
    println!("{}", huffman_encoding.format_encodings());
    println!("");

    // print encoded string
    let mut encoded_string = String::new();
    huffman_encoding
        .encode_words(&mut words.clone())
        .map_err(|_| Error::Unreachable("there shouldn't be words that has no encodingXX"))?
        .into_iter()
        .for_each(|code| encoded_string.push_str(&format!("{}", code)));
    println!("{}", encoded_string);

    return Ok(());
}
