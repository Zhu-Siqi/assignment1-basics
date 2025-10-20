from typing import BinaryIO, List, Dict, Tuple, Iterator
from collections import Counter
import os
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

def str2bytes(string: str) -> List[bytes]:
    '''
    convert a string to a list of bytes
    (Direct iteration on a bytes object yields integers
    which should be converted to bytes
    Why the integer type is invalid?
        bytes objects are the symbolic units that can be
        concatenated during BPE merges, whereas integers cannot
    )
    '''
    encoded = string.encode('utf-8')
    return [encoded[i:i+1] for i in range(len(encoded))]

def text2pre_tokens(
        chunk_text: str,
        pattern: re.Pattern,
) -> Iterator[Tuple[bytes, ...]]:
    '''
    yields pre_tokens from chunk_text
    '''
    for matching in pattern.finditer(chunk_text):
        pre_token = matching.group(0)
        yield tuple(str2bytes(pre_token))

def pre_tokenization(
        file: BinaryIO,
        split_idxs: List[int],
        pattern: str,
        special_tokens: List[str],
) -> Dict[Tuple[bytes, ...], int]:
    '''
    For a binary read file and the split idxs:
    Do
    (1) Cache each split; (2) Decode cache as string;
    (3) Split the cache with the special tokens;
    (4) Use pattern to extract pre_tokens;
    (5) Count the pre_tokens under the form of bytes;
    '''
    # Initialize the indicator/the counter/patterns
    file.seek(0)
    pre_tokens_cnt = Counter()
    extract_PAT = re.compile(pattern) # faster processing with compiling
    
    # Sort with descending length, so that the longer tokens are matched first
    # deal with null special_tokens
    special_PAT = None
    if special_tokens:
        special_tokens = sorted(set(special_tokens), key=len, reverse=True)
        special_PAT = '|'.join(re.escape(t) for t in special_tokens)
        special_PAT = re.compile(special_PAT)

    # Split
    for idx in range(len(split_idxs)-1):
        # Convert bytes to string
        cache_file = file.read(split_idxs[idx+1] - split_idxs[idx])
        cache_file = cache_file.decode('utf-8', errors = 'ignore')

        # deal with null special_tokens
        if not special_PAT:
            cache_file = [cache_file]
        else:
            cache_file = [t for t in special_PAT.split(cache_file) if t] # split returns ''
        
        for t in cache_file:
            for pre_token in text2pre_tokens(t, extract_PAT):
                pre_tokens_cnt[pre_token] += 1
    
    return pre_tokens_cnt

if __name__ == '__main__':
    
    # Use valid dataset to test the functions

    # Check convert function
    print(type(str2bytes('abcd')[0]))
    print(str2bytes('abcd'))

    # Check split function
    with open('data/TinyStoriesV2-GPT4-valid.txt', 'rb') as f:
        split_idxs = split_document(
            f, 4, b'<|endoftext|>'
        )
        pre_token_dict = pre_tokenization(
            f, split_idxs, PAT, ['<|endoftext|>']
        )
    print(split_idxs)
    print(pre_token_dict)