from typing import BinaryIO, List, Dict, Tuple, Iterator, Set
from collections import Counter, defaultdict
import os
import multiprocessing as mp
from functools import partial
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def split_document(
        file: BinaryIO,
        n_splits: int,
        end_token: bytes,
) -> List[int]:
    '''
    split the document and ensure the middle parts end before end_token
    '''
    assert isinstance(end_token, bytes), "end_token must be a bytestring"
    
    # count the bytes size
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)

    # caculate the average size
    split_size = size // n_splits

    # Initialize the split
    split_idxs = [split_size * i for i in range(n_splits+1)]
    split_idxs[-1] = size

    # Define the cache size
    cache_size = min(2048, size // (n_splits * 3)) # adapt cache size to balance the split size

    # Refine the split idxs to end with end_token
    for i in range(1, n_splits - 1): # the first and last idx remains
        start_idx = split_idxs[i]
        file.seek(start_idx)
        
        # seek the end token
        while True:
            cache_bytes = file.read(cache_size)

            # When reaching the end, return empty bytes
            if cache_bytes == b'':
                split_idxs[i] = size
                break

            # Find the end token in cache bytes, return the idx before the first byte of end_token
            cut_idx = cache_bytes.find(end_token)
            if cut_idx != -1:
                split_idxs[i] = start_idx + cut_idx
                break
            else:
                start_idx += cache_size
        
    # The idxs may overlap with long context. 
    # If overlapping happens, there are same idxs.
    # sort to make the idx increase.
    return sorted(set(split_idxs))

def bytes_split(
        byt: bytes,
) -> Tuple[bytes]:
    return tuple([bytes([b]) for b in byt])

def text2pre_tokens(
        chunk_text: bytes,
        pattern: re.Pattern,
) -> Iterator[bytes]:
    '''
    yields pre_tokens from chunk_text
    '''
    for matching in pattern.finditer(chunk_text):
        pre_token = matching.group(0)
        # Only consider the pre_tokens whose length are larger than 1
        if len(pre_token) >= 2:
            yield pre_token

def chunk_pre_tokenization(
        start: int,
        end: int, 
        file_path: str,
        extract_PAT: re.Pattern,
        special_PAT: re.Pattern,
) -> Dict[Tuple[bytes], int]:
    '''
    Given a file path, start idx, end idx (parallel reading):
    Do
    (1) Split the cache with the special tokens;
    (2) Use pattern to extract pre_tokens;
    (3) Count the pre_tokens under the form of bytes;
    '''
    # Initialize the counter
    pre_tokens_cnt = Counter()

    # Read the chunk
    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    # Extract pre_tokens
    chunk_bytes = [subchunk for subchunk in special_PAT.split(chunk_bytes) if subchunk] # split returns ''
        
    for subchunk in chunk_bytes:
        for pre_token in text2pre_tokens(subchunk, extract_PAT):
            pre_tokens_cnt[bytes_split(pre_token)] += 1
    
    return pre_tokens_cnt

def file_pre_tokenization(
        file_path: str,
        special_tokens: str,
        extract_pattern: str = PAT,
        n_processes: int = 8,
) -> Dict[Tuple[bytes], int]:
    '''
    Given a file path, the number of processes, 
    special_tokens(assume the end_token is the first token in special_tokens),
    return the counter of pre_tokens under the type of bytes
    '''
    # check if end_token is in special_tokens
    assert len(special_tokens) >= 1, 'at least one special token is needed'

    # Initialize the patterns
    extract_PAT = re.compile(extract_pattern.encode('utf-8')) # faster processing with compiling
    # Sort with descending length, so that the longer tokens are matched first
    special_tokens = sorted(set(special_tokens), key=len, reverse=True)
    special_PAT = b'|'.join(re.escape(t).encode('utf-8') for t in special_tokens)
    special_PAT = re.compile(special_PAT)

    # chunk split
    with open(file_path, 'rb') as f:
        # Get the split idxs
        split_idxs = split_document(
            file= f, 
            n_splits = n_processes,
            end_token = special_tokens[0].encode('utf-8')
        )

    # the start end pair
    idx_pair = zip(split_idxs[:-1], split_idxs[1:])
    with mp.Pool(processes=n_processes) as pool:
        # fix the patterns
        task = partial(
            chunk_pre_tokenization,
            file_path = file_path,
            extract_PAT = extract_PAT,
            special_PAT = special_PAT,
        )

        # map the file chunks
        partial_cnts = pool.starmap(task, idx_pair)
    
    # combine the partial counters
    total_cnts = Counter()
    for cnt in partial_cnts:
        total_cnts.update(cnt)
    
    return total_cnts

def count_pairs(
        start: int,
        end: int,
        pre_tokens_list: List[Tuple[Tuple[bytes], int]]
) -> Tuple[Dict[Tuple[bytes, bytes], int], Dict[Tuple[bytes, bytes], Set[Tuple[bytes]]]]:
    '''
    After the converting the counter to list,
    calculate the pairs within a chunk,
    and log the correlated pre_token
    '''
    # initialize the pair counter and the pairs-pre_token connection
    pairs_cnt = Counter()
    pairs2pt = defaultdict(set)

    # scan the pre_tokens
    for i in range(start, end):
        pre_token, freq = pre_tokens_list[i]
        # calculate the 
        for j in range(len(pre_token)-1):
            pair = (pre_token[j], pre_token[j+1])
            pairs_cnt[pair] += freq
            pairs2pt[pair].add(pre_token)
        
    return pairs_cnt, pairs2pt


def pre_tokens2pairs(
    pre_tokens_cnt: Dict[Tuple[bytes], int],
    n_processes: int = 8,
) -> Tuple[Dict[Tuple[bytes, bytes], int], Dict[Tuple[bytes, bytes], Set[Tuple[bytes]]]]:
    '''
    count the byte pairs in a pre_token counter (parallel)
    '''
    # initialize the pair counter and the pairs-pre_token connection
    pairs_cnt = Counter()
    pairs2pt = defaultdict(set)

    # parallelize the pairs couting
    # list for process
    pre_tokens_cnt = list(pre_tokens_cnt.items())
    # caculate the chunk idx
    n = len(pre_tokens_cnt)
    chunk_n = n // n_processes
    split_idxs = [i * chunk_n for i in range(n_processes)]
    split_idxs.append(n)
    split_pair = zip(split_idxs[:-1], split_idxs[1:])

    # multiprocess
    with mp.Pool(processes=n_processes) as pool:
        task = partial(
            count_pairs,
            pre_tokens_list = pre_tokens_cnt
        )

        results = pool.starmap(task, split_pair)

    # combine the counters and mapping
    for cnt, mapper in results:
        pairs_cnt.update(cnt)
        # update sets
        for pair, pre_tokens in mapper.items():
                pairs2pt[pair].update(pre_tokens)

    return pairs_cnt, pairs2pt


if __name__ == '__main__':

    # Check split function
    with open('data/TinyStoriesV2-GPT4-valid.txt', 'rb') as f:
        split_idxs = split_document(
            f, 5, b'<|endoftext|>'
        )
    print(split_idxs)

    # check the pre_tokenization function
    pre_tokens_cnt = file_pre_tokenization(
        'data/TinyStoriesV2-GPT4-valid.txt',
        ['<|endoftext|>'],
    )
    print(list(pre_tokens_cnt.items())[:5])

    # check the pair count function
    pairs_cnt, pairs2pt = pre_tokens2pairs(pre_tokens_cnt, 5)
    print(list(pairs_cnt.items())[:5])
    print(list(pairs2pt.items())[:1])