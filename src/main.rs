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

fn main() -> Result<(), Error> {
    // utilities
    #[derive(Debug, Clone)]
    enum HuffTree<'a> {
        Leaf(&'a str),
        Node(Box<HuffTree<'a>>, &'a str),
    }
    impl<'a> From<HuffTree<'a>> for HashMap<String, usize> {
        fn from(tree: HuffTree) -> HashMap<String, usize> {
            fn helper<'a>(
                next: usize,
                encodings: HuffTree<'a>,
                map: &'a mut HashMap<String, usize>,
            ) -> () {
                match encodings {
                    HuffTree::Leaf(word) => {
                        map.insert(word.to_string(), next);
                    }
                    HuffTree::Node(rest, word) => {
                        let prefix = next << 1;
                        map.insert(word.to_string(), prefix);
                        helper(prefix + 1, *rest, map);
                    }
                }
            }
            let mut map = HashMap::new();
            helper(0, tree, &mut map);
            map
        }
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
    // println!(
    //     "get words from stdin:\n\t{:?}",
    //     words.clone().collect::<Vec<&str>>()
    // );

    // count occurrences of each word
    let mut words_occurrences: HashMap<&str, usize> = HashMap::new();
    for word in words.clone() {
        match &words_occurrences.get(&word) {
            None => words_occurrences.insert(word, 0),
            Some(&occurrences) => words_occurrences.insert(word, occurrences + 1),
        };
    }
    // println!(
    //     "count occurrences of each word:\n\t{:?}",
    //     &words_occurrences
    // );

    // sort by occurences, from more to less
    let mut words_occurrences: Vec<(&str, usize)> = words_occurrences
        .into_iter()
        .collect::<Vec<(&str, usize)>>();
    words_occurrences.sort_by(|(_, v0), (_, v1)| Ord::cmp(v1, v0));
    let mut words_occurrences: Vec<&str> = words_occurrences
        .into_iter()
        .map(|(word, _)| word)
        .collect();
    // println!(
    //     "sort by occurences, from more to less:\n\t{:?}",
    //     &words_occurrences
    // );

    // derive the huffman encodings of words as a tree
    let huffman_encoding: HuffTree = words_occurrences
        .pop()
        .map(|rarest| into_huffman_coding(&mut words_occurrences, HuffTree::Leaf(rarest)))
        .ok_or(Error::NoInput)?;
    // println!(
    //     "derive the huffman encodings of words as a tree:\n\t{:?}",
    //     &huffman_encoding
    // );

    // print each word and corresponding encoding
    let huffman_encoding: HashMap<String, usize> = huffman_encoding.into();
    for (word, encoding) in huffman_encoding.into_iter() {
        println!("{}\n\t{:b}", &word, &encoding);
    }

    return Ok(());
}
