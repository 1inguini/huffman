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

// utilities
#[derive(Debug, Clone)]
enum HuffTree<'a> {
    Leaf(&'a str),
    Node(Box<HuffTree<'a>>, &'a str),
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
                    HuffTree::Node(Box::new(rarest), rare),
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
            .map(|rarest| helper(&mut words_occurrences, HuffTree::Leaf(rarest)))
    }

    /// format Hufftree to string with each word and corresponding encoding
    fn format_encodings(&self) -> String {
        fn helper(next: String, encodings: &HuffTree) -> String {
            match encodings {
                HuffTree::Leaf(word) => word.to_string() + "\t" + &next + "\n",
                HuffTree::Node(rest, word) => {
                    word.to_string() + "\t" + &next + "0" + "\n" + &helper(next + "1", rest)
                }
            }
        }
        helper("".to_string(), self)
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

    return Ok(());
}
