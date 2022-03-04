mod util {
    use bitvec::{slice::BitSlice, vec::BitVec};
    use std::{cmp::Ordering, collections::HashMap, hash::Hash, str::FromStr};

    /// count occurrences of each word
    pub fn count_occurrences<T>(words: &mut dyn Iterator<Item = T>) -> HashMap<T, usize>
    where
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

mod huffman {
    use crate::util;
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
        pub fn encode_sequence(
            &self,
            symbols: &mut dyn Iterator<Item = Symbol>,
        ) -> Result<BitVec, usize> {
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
            // self.node.update(code, symbol)
            match (self.node.as_mut(), code.split_first()) {
                (Node::Symbol(location), None) => {
                    let old = mem::take(location);
                    *location = Some(symbol);
                    old
                }
                (node @ Node::Symbol(None), Some(..)) => {
                    *node = Node::Branch {
                        one: Box::new(Node::Symbol(None)),
                        zero: Box::new(Node::Symbol(None)),
                    };
                    self.update(code, symbol)
                }
                // other code was a prefix of the given code
                (Node::Symbol(Some(..)), Some(..)) => Some(symbol),
                // the given code was a prefix of other code
                (Node::Branch { .. }, None) => Some(symbol),
                (Node::Branch { zero, one }, Some((bit, code))) => {
                    let dummy = Box::new(Node::Symbol(None));
                    let node: &mut Box<Node<Option<Symbol>>> = if *bit { one } else { zero };
                    let mut temp: Intermediate<Symbol> = Intermediate {
                        node: mem::replace(node, dummy),
                    };
                    let result = temp.update(code, symbol);
                    mem::swap(&mut temp.node, node);
                    result
                }
            }
        }

        /// Converts `Node<Option<Symbol>>` to `Node<Symbol>`.
        /// Returns the given node unchanged if there is any `None` inside.
        pub fn harden(&self) -> Option<Tree<&Symbol>> {
            Some(Tree {
                inner: self.node.harden()?,
            })
        }

        /// collect codes that has no symbol associated
        pub fn missing(&self) -> impl Iterator<Item = BitVec> {
            // self.node.missing()
            let mut vacant_codes = Vec::new();
            let mut stack = vec![(BitVec::new(), self.node.as_ref())];
            while let Some((code, node)) = stack.pop() {
                match node {
                    Node::Symbol(None) => {
                        vacant_codes.push(code);
                    }
                    Node::Symbol(Some(..)) => {}
                    Node::Branch { zero, one } => {
                        {
                            let mut code = code.clone();
                            code.push(false);
                            stack.push((code, zero));
                        }
                        {
                            let mut code = code;
                            code.push(true);
                            stack.push((code, one));
                        }
                    }
                }
            }
            vacant_codes.into_iter()
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
    }
    impl<Symbol> Node<Option<Symbol>>
    where
        Symbol: Ord,
    {
        /// Converts Node<Option<Symbol>> to Node<Symbol>.
        /// Returns the given node unchanged if there is any None inside.
        fn harden(&self) -> Option<Box<Node<&Symbol>>> {
            Some(Box::new(match self {
                Node::Symbol(Some(symbol)) => Node::Symbol(symbol),
                Node::Symbol(None) => {
                    return None;
                }
                Node::Branch { zero, one } => Node::Branch {
                    zero: ((zero).harden()?),
                    one: ((one).harden()?),
                },
            }))
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
            pub fn from_sequence(sequence: &mut dyn Iterator<Item = Symbol>) -> Option<Self>
            where
                Symbol: Hash,
            {
                Self::new(util::count_occurrences(sequence).into_iter())
            }
            /// format Hufftree to string with each word and corresponding encoding
            /// Each word-encoding relation is newwline seperated,
            /// and each word-encoding relation is represented by tab seperated pair of word and encoding.
            pub fn codebook_tsv<D>(&self, display: &D) -> String
            where
                D: Fn(&Symbol) -> String,
            {
                fn records<Symbol, D>(display: &D, code: String, current: &Node<Symbol>) -> String
                where
                    Symbol: Ord,
                    D: Fn(&Symbol) -> String,
                {
                    match current {
                        Node::Symbol(symbol) => format!("{}\t{}", display(symbol), code),
                        Node::Branch { zero, one } => {
                            format!(
                                "{}\n{}",
                                records(display, format!("{}0", code), zero),
                                records(display, format!("{}1", code), one)
                            )
                        }
                    }
                }
                format!(
                    "Symbol\tCode\n{}",
                    records(display, String::new(), &self.inner)
                )
            }

            /// encode a single symbol
            pub fn encode(&self, symbol: &Symbol) -> Option<BitVec> {
                self.inner.index_owned(symbol)
            }

            /// encode symbol sequence
            pub fn encode_sequence(
                &self,
                symbols: &mut dyn Iterator<Item = Symbol>,
            ) -> Result<BitVec, usize> {
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

use bitvec::vec::BitVec;
use std::{
    cmp::Ord,
    collections::HashMap,
    fmt::Display,
    hash::Hash,
    io::{self, BufRead, BufWriter, Write},
    marker::PhantomData,
};
use util::IsAt;

/// options
use clap::{ArgEnum, CommandFactory, Parser, Subcommand};
/// represent all acceptable arguments
#[derive(Debug, Parser)]
#[clap(author, version, about, long_about = None)]
struct Config {
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
    /// codebook output format
    #[clap(long, short = 'b', arg_enum)]
    codebook_format: CodebookFormat,

    /// read input as words or binary
    #[clap(long, short = 's', arg_enum)]
    symbol_type: SymbolType,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ArgEnum)]
enum CodebookFormat {
    /// don't output
    Hide,
    /// tab separated values
    Tsv,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ArgEnum)]
enum SymbolType {
    Bytes,
    Words,
}

#[derive(Debug, Parser)]
struct Decode {
    /// read input as words or binary
    #[clap(long, short = 's', arg_enum)]
    symbol_type: SymbolType,
}

#[derive(Debug)]
struct Cli<'i, 'o, In, Out> {
    stdin: &'i mut In,
    stdout: &'o mut Out,
}

impl<'i, 'o, In, Out> Cli<'i, 'o, In, Out>
where
    In: BufRead,
    Out: Write,
{
    fn init() -> Config {
        // abort when there is no input from stdin
        if atty::is(atty::Stream::Stdin) {
            Config::command()
                .error(
                    clap::ErrorKind::MissingRequiredArgument,
                    "huffman only accepts string from stdin.",
                )
                .exit();
            // return Err(Error::NoStdin);
        } else {
            // get arguments
            Config::parse()
        }
    }

    fn encode(self, config: Encode) -> Result<(), Error> {
        // get words from stdin, waits until EOF
        let input = {
            let mut input = String::new();
            self.stdin.read_to_string(&mut input).map_err(Error::Io)?;
            input
        };

        match config.symbol_type {
            SymbolType::Words => {
                let symbols = || input.split_whitespace();
                self.encode_generic(
                    config,
                    &|s: &&str| s.to_string(),
                    &mut symbols(),
                    &mut symbols(),
                )
            }
            SymbolType::Bytes => {
                let symbols = || input.bytes();
                self.encode_generic(
                    config,
                    &|byte| format!("0x{:>02X}", byte),
                    &mut symbols(),
                    &mut symbols(),
                )
            }
        }
    }

    fn encode_generic<Symbol: Ord + Hash>(
        self,
        config: Encode,
        display: &dyn Fn(&Symbol) -> String,
        symbols0: &mut dyn Iterator<Item = Symbol>,
        symbols1: &mut dyn Iterator<Item = Symbol>,
    ) -> Result<(), Error> {
        let Cli { stdout, .. } = self;
        // derive the huffman encodings of words as a tree
        let huffman_encodings =
            huffman::canonical::Tree::from_sequence(symbols0).ok_or(Error::NoStdin)?;

        // print codebook
        match config.codebook_format {
            CodebookFormat::Hide => {}
            CodebookFormat::Tsv => {
                writeln!(stdout, "{}", huffman_encodings.codebook_tsv(&display))
                    .map_err(Error::Io)?;
                writeln!(stdout).map_err(Error::Io)?;
            }
        }
        // print encoded string
        writeln!(
            stdout,
            "{}",
            util::format_bits(&huffman_encodings.encode_sequence(symbols1).map_err(|_| {
                Error::Unreachable("there shouldn't be words that has no encoding")
            })?,)
        )
        .map_err(Error::Io)?;
        Ok(())
    }

    fn decode(self, Decode { .. }: Decode) -> Result<(), Error>
    where
        In: BufRead,
        Out: Write,
    {
        let Cli { stdin, stdout, .. } = self;
        macro_rules! println {
            ($($arg:tt)*) => ({
                writeln!(stdout, $($arg)*).map_err(Error::Io)?;
            })
        }
        // get lines from stdin, one line at a time
        let mut lines = stdin.lines().enumerate();

        // skip newlines before encoding definitions
        let mut trailing = false;

        // record codes
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
                                is: CodeDefinitionError::InvalidCode(CodeStringError::NonBinary),
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
                                CodeDefinitionError::InvalidCode(CodeStringError::MalformedBinary)
                            },
                        });
                        continue;
                    };
                }
            }
        }
        // validate and convert Hashmap to Encodings
        let codebook = codebook.harden().ok_or_else(|| {
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
}

fn main() -> Result<(), Error> {
    let args = Cli::<io::StdinLock, BufWriter<io::Stdout>>::init();

    // prepare stdout with buffering
    let stdout = io::stdout();
    let stdout = &mut BufWriter::new(stdout.lock());

    // prepare stdin with buffering
    let stdin = io::stdin();
    let stdin = &mut stdin.lock();

    let cli = Cli { stdin, stdout };

    // run each subcommands
    let result = match args.mode {
        Mode::Encode(config) => cli.encode(config),
        Mode::Decode(config) => cli.decode(config),
    };

    stdout.flush().map_err(Error::Io)?;
    result
}
