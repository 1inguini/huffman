use std::collections::HashMap;
use std::io;
use std::io::Read;

fn main() {
    let mut input = String::new();
    match io::stdin().read_to_string(&mut input) {
        Ok(_) => {
            let words = input.split_whitespace();
            let mut words_occurrences: HashMap<&str, usize> = HashMap::new();
            for word in words {
                match &words_occurrences.get(&word) {
                    None => words_occurrences.insert(word, 1),
                    Some(&occurrences) => words_occurrences.insert(word, occurrences + 1),
                };
            }
            println!("{:?}", words_occurrences);
        }
        Err(err) => println!("{}", err),
    }
}
