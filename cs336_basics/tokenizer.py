import os
from typing import BinaryIO
import regex

class Tokenizer:

    def __init__(self, vocab, special_tokens, merges):
        self.vocab = vocab
        self.special_tokens = special_tokens
        self.str_to_token = {}
        self.merge_ranks = {}

        for token_id, s in vocab.items():
            self.str_to_token[s] = token_id

        # Store merges as (bytes1, bytes2) -> rank
        for i, m in enumerate(merges):
            self.merge_ranks[(m[0], m[1])] = i

    def _bpe_encode_word(self, word_bytes):
        """Apply BPE merges to a word (bytes) and return list of token IDs."""
        # Start with individual bytes as tokens
        tokens = [bytes([b]) for b in word_bytes]
        
        # Iteratively merge pairs until no more merges possible
        while len(tokens) > 1:
            # Find the pair with the lowest merge rank
            best_pair_idx = None
            best_rank = float('inf')
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair, float('inf'))
                if rank < best_rank:
                    best_rank = rank
                    best_pair_idx = i
            
            # No more merges possible
            if best_pair_idx is None or best_rank == float('inf'):
                break
            
            # Merge the best pair
            new_token = tokens[best_pair_idx] + tokens[best_pair_idx + 1]
            tokens = tokens[:best_pair_idx] + [new_token] + tokens[best_pair_idx + 2:]
        
        # Convert to token IDs
        return [self.str_to_token[t] for t in tokens]

    def encode(self, text) -> [int]:
        tokens = []

        # 1. Split by special tokens
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = '|'.join(regex.escape(token) for token in sorted_tokens)
            parts = regex.split(f"({pattern})", text)
        else:
            parts = [text]

        # turn each part from str -> token ids
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        for part in parts:
            if self.special_tokens and part in self.special_tokens:
                tokens.append(self.str_to_token[part.encode('utf-8')])
                continue

            # pre tokenize
            matches = regex.findall(PAT, part) or ([part] if part else [])
            for word_str in matches:
                word_bytes = word_str.encode('utf-8')
                tokens.extend(self._bpe_encode_word(word_bytes))

        return tokens

    def decode(self, tokens: [int]) -> str:
        text = b''

        for t in tokens:
            # print(self.vocab)
            # print('t', t, self.vocab[t])
            # print('ò', 'ò'.encode('utf-8'))
            # print('in', self.str_to_token[b'\xc3\xb2'])
            text += self.vocab[t]

        return text.decode('utf-8', errors='replace')

    def _find_chunk_boundaries(
        self,
        file: BinaryIO,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(self.special_tokens, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = 128000

        desired_num_chunks = file_size // chunk_size

        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            chunk = file.read(chunk_size)  # Read a mini chunk

            found_at = chunk.find(self.special_tokens)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
            else:
                chunk_boundaries[bi] = file_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def encode_iterable(self, f):

        for line in f:
            if self.special_tokens:
                sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
                pattern = '|'.join(regex.escape(token) for token in sorted_tokens)
                parts = regex.split(f"({pattern})", line)
            else:
                parts = [line]

            # turn each part from str -> token ids
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

            for part in parts:
                if self.special_tokens and part in self.special_tokens:
                    yield self.str_to_token[part.encode('utf-8')]
                    continue

                # pre tokenize
                matches = regex.findall(PAT, part) or ([part] if part else [])
                for word_str in matches:
                    word_bytes = word_str.encode('utf-8')
                    for token_id in self._bpe_encode_word(word_bytes):
                        yield token_id