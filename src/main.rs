use std::cmp::Ord;
use std::collections::HashMap;
use std::io;
use std::io::Read;

#[derive(Debug)]
enum Error {
    IoError(io::Error),
    NoInput,
}

fn main() -> Result<(), Error> {
    // utilities
    #[derive(Debug)]
    enum HuffTree<'a> {
        Leaf(&'a str),
        Node(Box<HuffTree<'a>>, &'a str),
    }
    fn into_huffman_coding<'a>(
        words_occurrences: &mut Vec<&'a str>,
        rarest: HuffTree<'a>,
    ) -> HuffTree<'a> {
        match words_occurrences.pop() {
            None => rarest,
            Some(rare) => {
                into_huffman_coding(words_occurrences, HuffTree::Node(Box::new(rarest), rare))
            }
        }
    }

    // get words from stdin
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(Error::IoError)?;
    let words = input.split_whitespace();
    println!(
        "get words from stdin:\n\t{:?}",
        &words.clone().collect::<Vec<&str>>()
    );

    // count occurrences of each word
    let mut words_occurrences: HashMap<&str, usize> = HashMap::new();
    for word in words {
        match &words_occurrences.get(&word) {
            None => words_occurrences.insert(word, 0),
            Some(&occurrences) => words_occurrences.insert(word, occurrences + 1),
        };
    }
    println!(
        "count occurrences of each word:\n\t{:?}",
        &words_occurrences
    );

    // sort by occurences, from more to less
    let mut words_occurrences: Vec<(&str, usize)> = words_occurrences
        .into_iter()
        .collect::<Vec<(&str, usize)>>();
    words_occurrences.sort_by(|(_, v0), (_, v1)| Ord::cmp(v1, v0));
    let mut words_occurrences: Vec<&str> = words_occurrences
        .into_iter()
        .map(|(word, _)| word)
        .collect();
    println!(
        "sort by occurences, from more to less:\n\t{:?}",
        &words_occurrences
    );

    // derive the huffman encodings of words as a tree
    let huffman_encoding = words_occurrences
        .pop()
        .map(|rarest| into_huffman_coding(&mut words_occurrences, HuffTree::Leaf(rarest)))
        .ok_or(Error::NoInput)?;
    println!(
        "derive the huffman encodings of words as a tree:\n\t{:?}",
        &huffman_encoding
    );

    return Ok(());
}
