import os
from typing import BinaryIO
import regex as re
from collections import defaultdict
import time
from multiprocessing import Pool

class PreTokenizer:
    def __init__(self, file_path: str, special_tokens: list[str]):

        assert special_tokens and len(special_tokens) >= 1, "Must provide one or more special tokens"
        self.file_path = file_path
        self.special_tokens = special_tokens

    def _find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            chunk = file.read(chunk_size)  # Read a mini chunk

            found_at = chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
            elif chunk == b'':
                chunk_boundaries[bi] = file_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def _process_chunk(self, args):
        file_path, start, end = args
        pre_tokens = defaultdict(int)

        with open(file_path, 'rb') as f:
            f.seek(start)
        
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token

            # 1. split by special token
            pattern = '|'.join(re.escape(token) for token in self.special_tokens)
            parts = re.split(f"({pattern})", chunk)

            # 2. find pre tokens and encode
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

            for p in parts:
                if p in self.special_tokens:
                    continue

                for w in re.finditer(PAT, p):
                    
                    k = w.group().encode('utf-8')
                    t = tuple(bytes([b]) for b in k)
                    pre_tokens[t] += 1

            return pre_tokens

    def get_pre_tokens(self):

        NUM_PROCESSES = 12
        # Working memory required to file size // k. 
        # owt dataset is around 11GB and my mac has 6 GB free memory
        NUM_CHUNKS = NUM_PROCESSES * 2

        with open(self.file_path, "rb") as f:
            boundaries = self._find_chunk_boundaries(f, NUM_CHUNKS, self.special_tokens[0].encode('utf-8'))

        # multi - processing
        pre_tokens = defaultdict(int)
        start_time = time.time()
        chunk_args = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            chunk_args.append((self.file_path, start, end))

        with Pool(processes=NUM_PROCESSES) as pool:
            # - Results are aggregated as they complete (not all at once)
            for result in pool.imap_unordered(self._process_chunk, chunk_args):
                for k, v in result.items():
                    pre_tokens[k] += v

        end_time = time.time()

        # print('total tokens', len(pre_tokens.keys()))
        # print('total count', sum(pre_tokens.values()))
        print("pre tokenization time: ", end_time - start_time)

        return pre_tokens