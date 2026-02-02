import re

class Tokenizer:

    def __init__(self, vocab, special_tokens, merges):
        self.vocab = vocab
        self.special_tokens = special_tokens
        self.merges = merges
        self.str_to_token = {}

        for token_id, s in vocab.items():
            self.str_to_token[s] = token_id

    def encode(self, text) -> [int]:

        tokens = []

        # 1. Split by special tokens
        pattern = '|'.join(re.escape(token) for token in self.special_tokens)
        parts = re.split(f"({pattern})", text)

        # turn each part from str -> token ids
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        for part in parts:
            if part in self.special_tokens:
                tokens.append([self.str_to_token[part]])
                continue

            # pre tokenize
            for word in re.finditer(PAT, part):
                word_byte_arr = [bytes(b) for b in word.group().encode('utf-8')]
                l = 0
                while l < len(word_byte_arr):
                    r = l + 1
                    while r <= len(word_byte_arr) and tuple(word_byte_arr[l, r]) in self.str_to_token:
                        r += 1

                    tokens.append(self.str_to_token[tuple(word_byte_arr[l, r + 1])])
                    l = r

        return tokens

    def decode(self, tokens: [int]) -> str:
        text = ''

        for t in tokens:
            text += self.vocab[t]

        return text