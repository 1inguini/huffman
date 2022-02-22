use std::cmp::Ord;
use std::collections::HashMap;
use std::io::Read;
use std::*;

#[derive(Debug)]
enum Error {
    Unreachable,
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
    fn format_encodings(&self) -> String {
        // fn helper(next: String, encodings: &HuffTree) -> String {
        //     match encodings {
        //         HuffTree::End(word) => word.to_string() + "\t" + &next + "\n",
        //         HuffTree::Branch(rest, word) => {
        //             word.to_string() + "\t" + &next + "0" + "\n" + &helper(next + "1", rest)
        //         }
        //     }
        // }
        // helper(String::new(), self)
        let mut result = String::new();
        for (word, code) in self.into_iter() {
            result = result + &format!("{}\t{}\n", word, code)
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

    fn encode_string(&self, string: &str) -> Option<String> {
        let mut result = String::new();
        for code in self.encode_words(&mut string.split_whitespace()).ok()? {
            result = result + &format!("{}", code);
        }
        Some(result)
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
    // get input from stdin
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(Error::IoError)?;
    // derive the huffman encodings of words as a tree
    let huffman_encoding: HuffTree =
        HuffTree::new(&mut input.split_whitespace()).ok_or(Error::NoInput)?;

    // print each word and corresponding encoding
    println!("{}", huffman_encoding.format_encodings());

    // print encoded string
    println!(
        "{}",
        huffman_encoding
            .encode_string(&input)
            .ok_or(Error::Unreachable)?
    );

    return Ok(());
}
