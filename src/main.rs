use std::cmp::Ord;
use std::collections::HashMap;
use std::io;
use std::io::Read;

fn main() -> Result<(), std::io::Error> {
    // get words from stdin
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let words = input.split_whitespace();

    // count occurrences of each word
    let mut words_occurrences: HashMap<&str, usize> = HashMap::new();
    for word in words {
        match &words_occurrences.get(&word) {
            None => words_occurrences.insert(word, 0),
            Some(&occurrences) => words_occurrences.insert(word, occurrences + 1),
        };
    }

    // sort by occurences, from more to less
    let mut words_occurrences: Vec<(&str, usize)> = words_occurrences
        .into_iter()
        .collect::<Vec<(&str, usize)>>();
    words_occurrences.sort_by(|(_, v0), (_, v1)| Ord::cmp(v1, v0));

    println!("{:?}", words_occurrences);
    return Ok(());
}
