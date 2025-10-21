from typing import BinaryIO, List, Dict, Tuple, Iterator, Set
import os
import multiprocessing as mp
from functools import partial
from .utils import file_pre_tokenization, pre_tokens2pairs
import heapq
from collections import Counter, defaultdict
from tqdm import tqdm

def init_vocab(
        special_tokens: List[str]
) -> Dict[int, bytes]:
    '''
    Initialize the vocabulary, special tokens follow the original tokens
    '''
    vocab = {i:bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')
    return vocab

class ComparablePair:
    """
    make the heap be the max heap
    """
    def __init__(self, freq: int, pair: Tuple[bytes, bytes]):
        self.freq = freq
        self.pair = pair

    def __lt__(self, other: 'ComparablePair') -> bool:
        if self.freq != other.freq:
            return self.freq > other.freq
        return self.pair > other.pair

    def __eq__(self, other: 'ComparablePair') -> bool:
        return self.freq == other.freq and self.pair == other.pair

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # intialize the final return
    vocab = init_vocab(special_tokens)
    merge_list = []
    n_processes=kwargs.get('n_processes', 4)

    # get the pre_tokens and pairs_counter
    pre_tokens_cnt = file_pre_tokenization(
        file_path=input_path,
        special_tokens=special_tokens,
        n_processes=n_processes
    )
    pairs_cnt, pair2pre_tokens = pre_tokens2pairs(
        pre_tokens_cnt=pre_tokens_cnt,
        n_processes=n_processes
    )

    # Auxiliary variables
    # store the current merge state of the pre_tokens
    pre_tokens_state = {key: list(key) for key in pre_tokens_cnt.keys()}
    # heapify the pairs counter
    pairs_heap = [ComparablePair(val, key) for key, val in pairs_cnt.items()]
    heapq.heapify(pairs_heap)

    # iterative merge
    for _ in tqdm(range(vocab_size - len(vocab))):
        # find the most frequent pairs
        comparable_item = heapq.heappop(pairs_heap)
        val, pair = comparable_item.freq, comparable_item.pair
        # It may be out of date
        while val != pairs_cnt.get(pair, 0):
            if not pairs_heap: # check if null heap
                return vocab, merge_list 
            comparable_item = heapq.heappop(pairs_heap)
            val, pair = comparable_item.freq, comparable_item.pair

        # Store the merge rule
        merge_list.append(pair)
        # Store the new vocab
        vocab[len(vocab)] = pair[0] + pair[1]
        # log the pair that changed this loop and push them later
        change_pair_set = set()
        
        # find the pre_tokens that contain the pair
        related_pre_tokens = pair2pre_tokens[pair]

        # initialize the bytes to be compared
        first, second = pair

        # iterate each pre_token
        # If we do multiprocessing here,
        # then we will suffer the cost of initialization of pool and serialization hundred of times!
        # Thus, we just do a for loop!
        for pre_token in related_pre_tokens:
            freq = pre_tokens_cnt[pre_token]
            # Get the current state
            raw_state = pre_tokens_state[pre_token]
            if len(raw_state) < 2:
                continue
            new_state = []

            # Merge all the pairs
            j = 0
            # log the change idx
            raw_left_idxs = []
            new_idxs = []
            while j < len(raw_state): # at least two tokens are needed
                # check the pair before reaching the end
                if j < len(raw_state)-1 and raw_state[j] == first and raw_state[j+1] == second:
                    # log the change idx
                    raw_left_idxs.append(j)
                    new_idxs.append(len(new_state))
                    # merge
                    new_state.append(first + second)
                    j += 2
                else:
                    # not merge
                    new_state.append(raw_state[j])
                    j += 1

            # !!! As we don't remove the pre_token when it doesn't contain some pair after merging,
            # !!! It is possible that the pre_token doesn't contain the pair!

            # store the new state
            if new_idxs:
                pre_tokens_state[pre_token] = new_state
            
                # count the changes // store the new pair mapper
                # minus raw pair
                for idx in raw_left_idxs:
                    # (A,[B],C,D)
                    # (B,C) which could be omitted as we pop this pair at last
                    # new_pair = (raw_state[idx], raw_state[idx+1])
                    # pairs_cnt[new_pair] -= freq
                    # change_pair_set.add(new_pair)
                    # (A,B)
                    if idx >= 1:
                        new_pair = (raw_state[idx-1], raw_state[idx])
                        pairs_cnt[new_pair] -= freq
                        change_pair_set.add(new_pair)
                    # (C,D)
                    if idx <= len(raw_state) - 3:
                        new_pair = (raw_state[idx+1], raw_state[idx+2])
                        pairs_cnt[new_pair] -= freq
                        change_pair_set.add(new_pair)
                # add new pair // store the new pair mapper
                for idx in new_idxs:
                    # (A, [BC], D)
                    # (A, BC)
                    if idx >= 1:
                        new_pair = (new_state[idx-1], new_state[idx])
                        pairs_cnt[new_pair] += freq
                        pair2pre_tokens[new_pair].add(pre_token)
                        change_pair_set.add(new_pair)
                    # (BC, D)
                    if idx <= len(new_state) - 2:
                        new_pair = (new_state[idx], new_state[idx+1])
                        pairs_cnt[new_pair] += freq
                        pair2pre_tokens[new_pair].add(pre_token)
                        change_pair_set.add(new_pair)

        # push changed pair count
        for change_pair in change_pair_set:
            new_freq = pairs_cnt.get(change_pair, 0)
            if new_freq > 0: # check if the freq is valid
                heapq.heappush(pairs_heap, ComparablePair(new_freq, change_pair))
        
        # Remove the pair from pair counter and pair mapper to save memory
        pairs_cnt.pop(pair)
        pair2pre_tokens.pop(pair)
    
    return vocab, merge_list

if __name__ == '__main__':

    # test bpe_training
    vocab, merge_list = run_train_bpe(
        'data/TinyStoriesV2-GPT4-valid.txt',
        500,
        ['<|endoftext|>'],
        **{'n_processes': 3}
    )
    print(vocab[300])
    print(merge_list[:5])