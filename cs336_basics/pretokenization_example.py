import os
from typing import BinaryIO
import re
import regex as re
from collections import defaultdict
from multiprocessing import Pool
import time

def find_chunk_boundaries(
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

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(args):
    file_path, start, end = args
    pre_tokens = defaultdict(int)

    with open(file_path, 'rb') as f:
        f.seek(start)
    
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token

        # 1. split by special token
        pattern = '|'.join(re.escape(token) for token in ["<|endoftext|>"])
        parts = re.split(f"({pattern})", chunk)

        # 2. find pre tokens and encode
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        for p in parts:
            for w in re.finditer(PAT, p):
                k = w.group().encode('utf-8')
                t = tuple(bytes([b]) for b in k)
                pre_tokens[t] += 1

        return pre_tokens


if __name__ == '__main__':
## Usage
    FILE_PATH = 'data/TinyStoriesV2-GPT4-train.txt'
    NUM_PROCESSES = 12

    with open(FILE_PATH, "rb") as f:
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES, b"<|endoftext|>")

    pre_tokens = defaultdict(int)

    # single thread
    start_time = time.time()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        result = process_chunk((FILE_PATH, start, end))
        for k, v in result.items():
            pre_tokens[k] += v

    end_time = time.time()
    print('total tokens', len(pre_tokens.keys()))
    print('total count', sum(pre_tokens.values()))
    print('single process elapsed time:', end_time - start_time)


    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.

    # multi - processing
    pre_tokens = defaultdict(int)
    start_time = time.time()
    chunk_args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk_args.append((FILE_PATH, start, end))

    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(process_chunk, chunk_args)

    for result in results:
        for k, v in result.items():
            pre_tokens[k] += v

    end_time = time.time()

    print('total tokens', len(pre_tokens.keys()))
    print('total count', sum(pre_tokens.values()))
    print("multi processing elapsed time: ", end_time - start_time)

    



