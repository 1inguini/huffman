use bitvec::vec::BitVec;
use std::cmp::Ord;
use std::collections::HashMap;
use std::io::{BufRead, BufWriter, Read, Write};
use std::*;
use util::IsAt;

mod util {
    use bitvec::{slice::BitSlice, vec::BitVec};
    use std::{cmp::Ordering, collections::HashMap, hash::Hash, str::FromStr};

    /// count occurrences of each word
    pub fn count_occurrences<I, T>(words: &mut I) -> HashMap<T, usize>
    where
        I: Iterator<Item = T>,
        T: Eq + Hash,
    {
        let mut words_occurrences: HashMap<T, usize> = HashMap::new();
        for word in words {
            match &words_occurrences.get(&word) {
                None => words_occurrences.insert(word, 1),
                Some(&occurrences) => words_occurrences.insert(word, occurrences + 1),
            };
        }
        words_occurrences
    }

    /// BitVec to String of '0's and '1's
    pub fn format_bits(bits: &BitSlice) -> String {
        let mut result = String::new();
        for bit in bits.iter() {
            result.push(if *bit { '1' } else { '0' });
        }
        result
    }

    /// Just wrap other types to impliment some trait.
    pub struct Wrap<T> {
        pub inner: T,
    }
    impl<T> Wrap<T> {
        pub fn new(inner: T) -> Self {
            Wrap { inner }
        }
    }
    impl FromStr for Wrap<BitVec> {
        type Err = usize;
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            let mut bits: BitVec = BitVec::new();
            for (position, byte) in s.bytes().enumerate() {
                match byte {
                    b'0' => bits.push(false),
                    b'1' => bits.push(true),
                    _ => {
                        return Err(position);
                    }
                }
            }
            Ok(Wrap { inner: bits })
        }
    }

    #[derive(Debug)]
    /// add positional information to some type
    pub struct IsAt<T> {
        /// number of lines from the top
        pub line: usize,
        /// number of character from the left
        pub character: usize,
        pub is: T,
    }
    impl<T> Ord for IsAt<T>
    where
        T: Ord,
    {
        fn cmp(&self, other: &Self) -> Ordering {
            Ord::cmp(&self.is, &other.is)
        }
    }
    impl<Symbol> PartialOrd for IsAt<Symbol>
    where
        Symbol: Ord,
    {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl<Symbol> PartialEq for IsAt<Symbol>
    where
        Symbol: Ord,
    {
        fn eq(&self, other: &Self) -> bool {
            matches!(self.cmp(other), Ordering::Equal)
        }
    }
    impl<Symbol> Eq for IsAt<Symbol> where Symbol: Ord {}
}

#[derive(Debug)]
enum Error {
    /// this would never happen
    Unreachable(&'static str),

    /// just relaying io::Error
    Io(io::Error),

    /// there is no input from stdin
    NoStdin,

    /// something is wrong with encoding definition part of input
    InvalidCodeDefinition(Vec<IsAt<CodeDefinitionError>>),
    // InvalidCodeDefinition(IsAt<CodeDefinitionError>),
    /// something is wrong with encoded string part of input
    InvalidCodeString(IsAt<CodeStringError>),
}

#[derive(Debug)]
enum CodeDefinitionError {
    /// definition is not in "word TAB encoding" style
    MisformattedDefinition,

    /// same word has been associated with another word
    DuplicateDefinitions,

    /// same code has been associated with another word
    DuplicateCodes,

    /// there is word missing definition, guessing from defined codes
    InsufficientDefinitions(Vec<BitVec>),
    /// code part has something wrong
    InvalidCode(CodeStringError),
}

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
    use bitvec::{ptr::BitRef, slice::BitSlice, vec::BitVec};
    use std::{fmt::Debug, iter, mem, ops::Index};
    /// represents the Huffman Code for symbols
    #[derive(Debug)]
    pub struct Tree<Symbol: Ord> {
        inner: Box<Node<Symbol>>,
    }
    impl<Symbol> Tree<Symbol>
    where
        Symbol: Ord,
    {
        pub fn new(symbol: Symbol) -> Self {
            Tree {
                inner: Box::new(Node::Symbol(symbol)),
            }
        }

        pub fn extend(zero: Self, one: Self) -> Self {
            Tree {
                inner: Box::new(Node::Branch {
                    zero: zero.inner,
                    one: one.inner,
                }),
            }
        }
        /// encode a single symbol
        pub fn encode(&self, symbol: &Symbol) -> Option<BitVec> {
            self.inner.index_owned(symbol)
        }

        /// encode symbol sequence
        pub fn encode_sequence<I>(&self, symbols: &mut I) -> Result<BitVec, usize>
        where
            I: Iterator<Item = Symbol>,
        {
            let mut result: BitVec = BitVec::new();
            for (i, symbol) in symbols.enumerate() {
                let mut code = self.encode(&symbol).ok_or(i)?;
                result.append(&mut code);
            }
            Ok(result)
        }

        // decode bits
        pub fn decode(&self, bits: &BitSlice) -> Result<impl Iterator<Item = &Symbol>, usize> {
            // self.inner.decode(bits)
            match &self.inner.as_ref() {
                Node::Symbol(..) => Err(0),
                Node::Branch { zero, one } => {
                    let mut result: Vec<&Symbol> = Vec::new();
                    let mut position = 0;
                    let mut punctuation = true;
                    let (mut on_false, mut on_true) = (zero, one);
                    for (pos, bit) in bits.iter().enumerate() {
                        position = pos;
                        let walk = if *bit { on_true } else { on_false };
                        match walk.as_ref() {
                            Node::Symbol(symbol) => {
                                result.push(symbol);
                                on_false = zero;
                                on_true = one;
                                punctuation = true;
                            }
                            Node::Branch { one, zero } => {
                                on_false = zero;
                                on_true = one;
                                punctuation = false;
                            }
                        }
                    }
                    if punctuation {
                        Ok(result.into_iter())
                    } else {
                        Err(position)
                    }
                }
            }
        }
    }

    impl<Symbol> Index<&BitSlice> for Tree<Symbol>
    where
        Symbol: Ord,
    {
        type Output = Symbol;
        fn index(&self, index: &BitSlice) -> &Self::Output {
            &self.inner[index]
        }
    }

    #[derive(Debug)]
    pub struct Intermediate<Symbol: Ord> {
        node: Box<Node<Option<Symbol>>>,
    }

    impl<Symbol> Intermediate<Symbol>
    where
        Symbol: Ord,
    {
        pub fn new() -> Self {
            Intermediate {
                node: Box::new(Node::Symbol(None)),
            }
        }
        /// updates symbol to node in tree representing the given code
        /// # Returns
        /// * `None` - successfully updated and the updated node was `None`
        /// * `Some(foo)` - successfully updated and the updated node was `Some(foo)`
        /// * `Some(symbol)` - the given code was a prefix of other code
        /// * `Some(symbol)` - other code was a prefix of the given code
        pub fn update(&mut self, code: &BitSlice, symbol: Symbol) -> Option<Symbol> {
            self.node.update(code, symbol)
        }

        /// Converts Node<Option<Symbol>> to Node<Symbol>.
        /// Returns the given node unchanged if there is any None inside.
        pub fn harden(self) -> Result<Tree<Symbol>, Self> {
            Ok(Tree {
                inner: Box::new(self.node.harden().map_err(|node| Intermediate {
                    node: Box::new(node),
                })?),
            })
        }

        /// collect code that has no symbol associated
        pub fn missing(&self) -> impl Iterator<Item = BitVec> {
            self.node.missing()
        }
    }

    /// internal representationof Huffman tree
    #[derive(Debug)]
    enum Node<Symbol>
    where
        Symbol: Ord,
    {
        Symbol(Symbol),
        Branch {
            zero: Box<Node<Symbol>>,
            one: Box<Node<Symbol>>,
        },
    }
    impl<Symbol> Node<Symbol>
    where
        Symbol: Ord,
    {
        /// Returns the path to shallowest occurence of given symbol in bits
        fn index_owned(&self, index: &Symbol) -> Option<BitVec> {
            // let mut code = self.index_owned_reversed(index)?;
            // code.reverse();
            // Some(code)
            let mut stack: Vec<(BitVec, &Node<Symbol>)> = vec![(BitVec::new(), self)];
            while let Some((code, node)) = stack.pop() {
                match node {
                    Node::Symbol(symbol) => {
                        if symbol == index {
                            return Some(code);
                        } else {
                            continue;
                        }
                    }
                    Node::Branch { one, zero } => {
                        let mut code_zero = code.clone();
                        code_zero.push(false);
                        stack.push((code_zero, zero));
                        let mut code_one = code;
                        code_one.push(true);
                        stack.push((code_one, one));
                    }
                }
            }
            None
        }

        // fn index_owned_reversed(&self, index: &Symbol) -> Option<BitVec> {
        //     match self {
        //         Node::Symbol(symbol) => {
        //             if symbol == index {
        //                 Some(BitVec::new())
        //             } else {
        //                 None
        //             }
        //         }
        //         Node::Branch { one, zero } => zero
        //             .as_ref()
        //             .index_owned_reversed(index)
        //             .map(|bits| {
        //                 let mut bits = bits;
        //                 bits.push(false);
        //                 bits
        //             })
        //             .or_else(|| {
        //                 one.as_ref().index_owned_reversed(index).map(|bits| {
        //                     let mut bits = bits;
        //                     bits.push(true);
        //                     bits
        //                 })
        //             }),
        //     }
        // }
        // decode bits
        fn decode(&self, bits: &BitSlice) -> Result<impl Iterator<Item = &Symbol>, usize> {
            let mut result: Vec<&Symbol> = Vec::new();
            self.decode_helper(self, &mut bits.iter(), &mut result)?;
            Ok(result.into_iter())
        }
        fn decode_helper<'a, 'b, Bits>(
            &'a self,
            walk: &'a Self,
            bits: &mut Bits,
            reversed_symbols: &mut Vec<&'a Symbol>,
        ) -> Result<(), usize>
        where
            Bits: Iterator<Item = BitRef<'b>> + ExactSizeIterator,
        {
            match (walk, bits.len()) {
                (Node::Symbol(symbol), 0) => {
                    reversed_symbols.push(symbol);
                    Ok(())
                }
                (Node::Symbol(symbol), ..) => {
                    reversed_symbols.push(symbol);
                    self.decode_helper(self, bits, reversed_symbols)
                }
                (Node::Branch { .. }, 0) => Err(0),
                (Node::Branch { zero, one }, ..) => match bits.next() {
                    Some(bit) => self
                        .decode_helper(if *bit { one } else { zero }, bits, reversed_symbols)
                        .map_err(|pos| pos + 1),
                    None => Err(0),
                },
            }
        }
    }
    impl<Symbol> Node<Option<Symbol>>
    where
        Symbol: Ord,
    {
        /// updates symbol to node in tree representing the given code
        /// # Returns
        /// * `None` - successfully updated and the updated node was `None`
        /// * `Some(foo)` - successfully updated and the updated node was `Some(foo)`
        /// * `Some(symbol)` - the given code was a prefix of other code
        /// * `Some(symbol)` - other code was a prefix of the given code
        fn update(&mut self, code: &BitSlice, symbol: Symbol) -> Option<Symbol> {
            match (self, code.split_first()) {
                (Node::Symbol(location), None) => {
                    let old = mem::take(location);
                    *location = Some(symbol);
                    old
                }
                (location @ Node::Symbol(None), Some(..)) => {
                    *location = Node::Branch {
                        one: Box::new(Node::Symbol(None)),
                        zero: Box::new(Node::Symbol(None)),
                    };
                    location.update(code, symbol)
                }
                (Node::Symbol(Some(..)), Some(..)) | (Node::Branch { .. }, None) => Some(symbol),
                (Node::Branch { zero, one }, Some((bit, index))) => {
                    if *bit {
                        one.update(index, symbol)
                    } else {
                        zero.update(index, symbol)
                    }
                }
            }
        }

        /// Converts Node<Option<Symbol>> to Node<Symbol>.
        /// Returns the given node unchanged if there is any None inside.
        fn harden(self) -> Result<Node<Symbol>, Self> {
            match self {
                Node::Symbol(Some(symbol)) => Ok(Node::Symbol(symbol)),
                node @ Node::Symbol(None) => Err(node),
                Node::Branch { zero, one } => Ok(Node::Branch {
                    zero: Box::new((*zero).harden()?),
                    one: Box::new((*one).harden()?),
                }),
            }
        }

        /// collect code that has no symbol associated
        fn missing(&self) -> impl Iterator<Item = BitVec> {
            let mut result: Vec<BitVec> = Vec::new();
            self.missing_helper(&mut result, BitVec::new());
            result.into_iter()
        }
        fn missing_helper(&self, vacant_codes: &mut Vec<BitVec>, code: BitVec) {
            match self {
                Node::Symbol(None) => {
                    vacant_codes.push(code);
                }
                Node::Symbol(Some(..)) => mem::drop(code),
                Node::Branch { zero, one } => {
                    {
                        let mut code = code.clone();
                        code.push(false);
                        zero.missing_helper(vacant_codes, code);
                    }
                    {
                        let mut code = code;
                        code.push(true);
                        one.missing_helper(vacant_codes, code);
                    }
                }
            }
        }
    }
    impl<Symbol> Index<&BitSlice> for Node<Symbol>
    where
        Symbol: Ord,
    {
        type Output = Symbol;

        /// Returns shallowest of corresponding symbols.
        fn index(&self, index: &BitSlice) -> &Self::Output {
            match (self, index.split_first()) {
                (Node::Symbol(symbol), None) => symbol,
                (Node::Branch { zero, one }, Some((bit, index))) => {
                    if *bit {
                        one.index(index)
                    } else {
                        zero.index(index)
                    }
                }
                (..) => panic!("Out of bounds access"),
            }
        }
    }
    /// Canonical Huffman
    pub mod canonical {
        use super::Node;
        use crate::util;
        use bitvec::{slice::BitSlice, vec::BitVec};
        use std::{
            cmp::{Ord, Ordering, PartialEq, PartialOrd},
            collections::VecDeque,
            fmt::Display,
            hash::Hash,
            ops::Index,
        };
        /// represents the Canonical Huffman Code for symbols
        #[derive(Debug)]
        pub struct Tree<Symbol: Ord> {
            inner: Box<Node<Symbol>>,
        }
        impl<Symbol> Tree<Symbol>
        where
            Symbol: Ord,
        {
            /// creates Canonical Huffman tree from HashMap of symbol to it's occurence
            /// returns None at empty input
            pub fn new<Map>(symbols_and_their_occurence: Map) -> Option<Self>
            where
                Map: Iterator<Item = (Symbol, usize)>,
            {
                // create nodes and sort them by occurence
                let mut leaves = symbols_and_their_occurence
                    .map(|(symbol, occurence)| Root::new(occurence, symbol))
                    .collect::<Vec<Root<Symbol>>>();
                leaves.sort_unstable();
                // prepare two queue, one filled with sorted nodes
                let mut leaves: VecDeque<Root<Symbol>> = leaves.into();
                let mut branches: VecDeque<Root<Symbol>> = VecDeque::new();

                loop {
                    match (
                        Root::pop_rarer(&mut leaves, &mut branches),
                        Root::pop_rarer(&mut leaves, &mut branches),
                    ) {
                        // pop the rarest two roots from the front of two queues and merge them ...
                        (Some(node0), Some(node1)) => {
                            branches.push_front(Root::merge(node0, node1))
                        }
                        // ... until there is only one root
                        (Some(node), None) | (None, Some(node)) => {
                            return Some(Tree { inner: node.inner })
                        }
                        (None, None) => return None,
                    }
                }
            }

            /// count occurence of each symbol in givin symbol sequence and construct Huffman tree
            pub fn from_sequence<I>(sequence: &mut I) -> Option<Self>
            where
                I: Iterator<Item = Symbol>,
                Symbol: Hash,
            {
                Self::new(util::count_occurrences(sequence).into_iter())
            }
            /// format Hufftree to string with each word and corresponding encoding
            /// Each word-encoding relation is newwline seperated,
            /// and each word-encoding relation is represented by tab seperated pair of word and encoding.
            pub fn codebook_tsv(&self) -> String
            where
                Symbol: Display,
            {
                fn records<Symbol>(code: String, current: &Node<Symbol>) -> String
                where
                    Symbol: Display + Ord,
                {
                    match current {
                        Node::Symbol(symbol) => format!("{}\t{}", symbol, code),
                        Node::Branch { zero, one } => {
                            format!(
                                "{}\n{}",
                                records(format!("{}0", code), zero),
                                records(format!("{}1", code), one)
                            )
                        }
                    }
                }
                format!("Symbol\tCode\n{}", records(String::new(), &self.inner))
            }

            /// encode a single symbol
            pub fn encode(&self, symbol: &Symbol) -> Option<BitVec> {
                self.inner.index_owned(symbol)
            }

            /// encode symbol sequence
            pub fn encode_sequence<I>(&self, symbols: &mut I) -> Result<BitVec, usize>
            where
                I: Iterator<Item = Symbol>,
            {
                let mut result: BitVec = BitVec::new();
                for (i, symbol) in symbols.enumerate() {
                    let mut code = self.encode(&symbol).ok_or(i)?;
                    result.append(&mut code);
                }
                Ok(result)
            }
        }
        impl<Symbol> Index<&BitSlice> for Tree<Symbol>
        where
            Symbol: Ord,
        {
            type Output = Symbol;
            fn index(&self, index: &BitSlice) -> &Self::Output {
                &self.inner[index]
            }
        }

        /// associates a symbol and it's occurence
        #[derive(Debug)]
        struct Root<Symbol>
        where
            Symbol: Ord,
        {
            /// occurence of the symbol in given sequence, or a sum of childrens
            occurence: usize,

            /// depth of the tree
            depth: usize,

            /// a symbol to be encoded
            inner: Box<Node<Symbol>>,
        }
        impl<Symbol> Root<Symbol>
        where
            Symbol: Ord,
        {
            pub fn new(occurence: usize, symbol: Symbol) -> Self {
                Root {
                    occurence,
                    depth: 0,
                    inner: Box::new(Node::Symbol(symbol)),
                }
            }

            /// merge two roots, combinging their occurences
            /// if the depth of two root are the same,
            /// then move the root with smaller deepest node to the `one` field
            pub fn merge(root0: Self, root1: Self) -> Self {
                let (zero, one) = match Ord::cmp(&root0.depth, &root1.depth) {
                    Ordering::Less => (root0, root1),
                    Ordering::Greater => (root1, root0),
                    Ordering::Equal => {
                        if root0.deepest() <= root1.deepest() {
                            (root0, root1)
                        } else {
                            (root1, root0)
                        }
                    }
                };
                Root {
                    occurence: zero.occurence + one.occurence,
                    depth: one.depth + 1,
                    inner: Box::new(Node::Branch {
                        zero: zero.inner,
                        one: one.inner,
                    }),
                }
            }

            /// pop the rarer element at the front of two queues
            fn pop_rarer(queue0: &mut VecDeque<Self>, queue1: &mut VecDeque<Self>) -> Option<Self> {
                match (queue0.front(), queue1.front()) {
                    (Some(node0), Some(node1)) => {
                        if node0.occurence <= node1.occurence {
                            queue0.pop_front()
                        } else {
                            queue1.pop_front()
                        }
                    }
                    (Some(..), None) => queue0.pop_front(),
                    (..) => queue1.pop_front(),
                }
            }

            /// view smallest symbol in deepest nodes
            fn deepest(&self) -> &Symbol {
                // fn helper<Symbol>(node: &Node<Symbol>) -> &Symbol
                // where
                //     Symbol: Ord,
                // {
                //     match &node {
                //         Node::Symbol(symbol) => symbol,
                //         Node::Branch { one, .. } => helper(one),
                //     }
                // }
                // helper(&self.inner);
                let mut node = &self.inner;
                loop {
                    match node.as_ref() {
                        Node::Symbol(symbol) => return symbol,
                        Node::Branch { one, .. } => node = one,
                    }
                }
            }
        }
        impl<Symbol> Ord for Root<Symbol>
        where
            Symbol: Ord,
        {
            fn cmp(&self, other: &Self) -> Ordering {
                match Ord::cmp(&self.occurence, &other.occurence) {
                    Ordering::Equal => Ord::cmp(self.deepest(), other.deepest()),
                    ord => ord,
                }
            }
        }
        impl<Symbol> PartialOrd for Root<Symbol>
        where
            Symbol: Ord,
        {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl<Symbol> PartialEq for Root<Symbol>
        where
            Symbol: Ord,
        {
            fn eq(&self, other: &Self) -> bool {
                matches!(self.cmp(other), Ordering::Equal)
            }
        }
        impl<Symbol> Eq for Root<Symbol> where Symbol: Ord {}
    }
}

// options
use clap::{ArgEnum, CommandFactory, Parser, Subcommand};

/// represent all acceptable arguments
#[derive(Debug, Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    mode: Mode,
}
#[derive(Debug, Subcommand)]
enum Mode {
    /// encodes words string
    Encode(Encode),

    /// decodes words string
    Decode(Decode),
}
#[derive(Debug, Parser)]
struct Encode {
    #[clap(long, short = 'b', arg_enum)]
    /// codebook output format
    codebook_format: CodebookFormat,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ArgEnum)]
enum CodebookFormat {
    /// don't output
    Hide,
    /// tab separated values
    Tsv,
}

#[derive(Debug, Parser)]
struct Decode {}

impl Cli {
    fn init() -> Cli {
        // abort when there is no input from stdin
        if atty::is(atty::Stream::Stdin) {
            Cli::command()
                .error(
                    clap::ErrorKind::MissingRequiredArgument,
                    "huffman only accepts string from stdin.",
                )
                .exit();
            // return Err(Error::NoStdin);
        }
        // get arguments
        Cli::parse()
    }
}

fn main() -> Result<(), Error> {
    let args = Cli::init();

    // prepare stdout with buffering
    let stdout = io::stdout();
    let mut stdout = BufWriter::new(stdout.lock());

    // prepare stdin with buffering
    let stdin = io::stdin();
    let mut stdin = stdin.lock();

    macro_rules! println {
        ($($arg:tt)*) => ({
            writeln!(stdout, $($arg)*).map_err(Error::Io)?;
        })
    }

    // run each subcommands
    return match args.mode {
        Mode::Encode(Encode { codebook_format }) => {
            // get words from stdin, waits until EOF
            let input = {
                let mut input = String::new();
                stdin.read_to_string(&mut input).map_err(Error::Io)?;
                input
            };
            let words = || input.split_whitespace();
            // derive the huffman encodings of words as a tree
            let huffman_encodings: huffman::canonical::Tree<&str> =
                huffman::canonical::Tree::from_sequence(&mut words()).ok_or(Error::NoStdin)?;

            // print codebook
            match codebook_format {
                CodebookFormat::Hide => {}
                CodebookFormat::Tsv => {
                    println!("{}", huffman_encodings.codebook_tsv());
                    println!();
                }
            }
            // print encoded string
            println!(
                "{}",
                util::format_bits(&huffman_encodings.encode_sequence(&mut words()).map_err(
                    |_| Error::Unreachable("there shouldn't be words that has no encoding")
                )?,)
            );
            Ok(())
        }
        Mode::Decode(Decode { .. }) => {
            // get lines from stdin, one line at a time
            let mut lines = stdin.lines().enumerate();

            // skip newlines before encoding definitions
            let mut trailing = false;

            // record codes
            // let mut dict: Vec<(usize, BitVec, String)> = Vec::new();
            let mut codebook: huffman::Intermediate<String> = huffman::Intermediate::new();
            // record errors
            let mut errors: Vec<IsAt<CodeDefinitionError>> = Vec::new();

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
                    trailing = true;
                }

                // each encoding definition is tab seperated pair of word and encoding
                match line.split_once('\t') {
                    None => {
                        errors.push(IsAt {
                            line: linenum,
                            character: 0,
                            is: CodeDefinitionError::MisformattedDefinition,
                        });
                        continue;
                    }
                    Some((word, code)) => {
                        if word == "Symbol" && code == "Code" {
                            continue;
                        }
                        let code = match code.parse::<util::Wrap<BitVec>>() {
                            Err(position) => {
                                errors.push(IsAt {
                                    line: linenum,
                                    character: word.len() + 1 + position,
                                    is: CodeDefinitionError::InvalidCode(
                                        CodeStringError::NonBinary,
                                    ),
                                });
                                continue;
                            }
                            Ok(bits) => bits.inner,
                        };
                        if let Some(old) = codebook.update(&code, word.to_string()) {
                            errors.push(IsAt {
                                line: linenum,
                                character: 0,
                                is: if word == old {
                                    // check for duplicate word
                                    CodeDefinitionError::DuplicateDefinitions
                                } else {
                                    CodeDefinitionError::InvalidCode(
                                        CodeStringError::MalformedBinary,
                                    )
                                },
                            });
                            continue;
                        };
                    }
                }
            }
            // validate and convert Hashmap to Encodings
            let codebook = codebook.harden().map_err(|codebook| {
                errors.push(IsAt {
                    line: 0,
                    character: 0,
                    is: CodeDefinitionError::InsufficientDefinitions(
                        codebook.missing().collect::<Vec<BitVec>>(),
                    ),
                });
                Error::InvalidCodeDefinition(errors)
            })?;

            // decode string
            for (linenum, line) in &mut lines {
                let line = line.map_err(Error::Io)?;
                if line.is_empty() {
                    continue;
                };
                // let decoded = {
                //     let mut decoded: String = String::new();
                //     let mut ones: usize = 0;
                //     for (pos, &byte) in line.as_bytes().iter().enumerate() {
                //         if encodings.common.len() <= ones {
                //             decoded.push_str(&encodings.rarest);
                //             decoded.push('\n');
                //             ones = 0;
                //         } else {
                //             match byte {
                //                 b'1' => ones += 1,
                //                 b'0' => {
                //                     decoded
                //                     .push_str(encodings.common.get(ones).ok_or(Error::Unreachable(
                //                     "ones should have been resetted before it gets out of bounds",
                //                 ))?);
                //                     decoded.push('\n');
                //                     ones = 0;
                //                 }
                //                 _ => {
                //                     return Err(Error::InvalidCodeString(IsAt {
                //                         line: linenum,
                //                         character: pos,
                //                         is: CodeStringError::NonBinary,
                //                     }));
                //                 }
                //             }
                //         }
                //     }
                // };

                // // check trailing bits
                // if 0 < ones {
                //     return Err(Error::InvalidCodeString(IsAt {
                //         line: linenum,
                //         character: line.as_bytes().len() - ones,
                //         is: CodeStringError::MalformedBinary,
                //     }));
                // }
                let decoded = codebook
                    .decode(
                        line.parse::<util::Wrap<BitVec>>()
                            .map_err(|position| {
                                Error::InvalidCodeString(IsAt {
                                    line: linenum,
                                    character: position,
                                    is: CodeStringError::NonBinary,
                                })
                            })?
                            .inner
                            .as_bitslice(),
                    )
                    .map_err(|position| {
                        Error::InvalidCodeString(IsAt {
                            line: linenum,
                            character: position,
                            is: CodeStringError::MalformedBinary,
                        })
                    });
                println!(
                    "{}",
                    decoded?.fold(String::new(), |accm, symbol| accm + " " + symbol)
                );
            }
            Ok(())
        }
    };
}
