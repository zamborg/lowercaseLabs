import collections
import re

class BPE:
    def __init__(self, vocabulary = None) -> None:
        self.vocabulary = vocabulary if vocabulary else self._default_vocab()
        self.str_split = lambda x : "".join([" ".join(word) + " </w>" for word in x.split(" ")])

    def build_vocabulary(self, text, vocab_size = 50):
        """
        Algorithm is as follows:
            - vocab is seeded at _default_vocab() -- all letters
            - space seperate the string and add end of word token " </w>" 
            - compute frequency counts for each pair of "tokens"
            - find most common pair and merge them
            - add that to the vocabulary
            - apply that merge to the space seperated string
            - iterate at frequency computation
        """
        tokens = self.str_split(text)
        self.merge_set = set()
        
        for i in range(vocab_size):
            pair, freq = self._compute_pair_frequency(tokens)
            self.vocabulary["".join(pair)] = freq
            self.merge_set.add(pair) # pair is a tuple
            tokens = re.sub(pattern=" ".join(pair), repl="".join(pair), string=tokens)

    def tokenize(self, text):
        """
        tokenize splits the text into base characters then applies the merge_set iteratively until it can no longer merge.
        returns a list of text tokens
        """
        tokens = self.str_split(text).split(" ")
        update = True
        while update:
            update=False
            i = 0
            while (i < len(tokens)-2):
                a,b = tokens[i], tokens[i+1]
                if (a,b) in self.merge_set:
                    tokens.pop(i+1)
                    tokens[i] = a+b
                    update=True
                i += 1
        return tokens

    def _compute_pair_frequency(self, text, return_max=True):
        """
        Text is space "tokenized"
        """
        tokens = text.split(" ")
        pairs = collections.defaultdict(int) # tuple : int
        for i in range(len(tokens)-1):
            pairs[(tokens[i], tokens[i+1])] += 1
        # pairs computed now we add the most common one to the vocabulary.
        if return_max:
            return max(pairs.items(), key=lambda x:x[1])
        sort = dict(sorted(pairs.items(), key= lambda x: x[1])) # sort by value
        return sort

    def _default_vocab(self):
        default = "".join("the quick brown fox jumped over the lazy dog".split())
        vocab = collections.defaultdict(int)
        for c in default:
            vocab[c] += 1
        vocab["</w>"] = len(default.split())
        return vocab
