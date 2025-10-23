from typing import List, Dict, Tuple, Iterable, Iterator
import os
from .utils import file_pre_tokenization, pre_tokens2pairs, PAT
import heapq
import regex as re
import json
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


class Tokenizer:
    '''
    The BPE tokenzier to encode and decode
    '''
    def __init__(
            self,
            vocab: Dict[int, bytes],
            merges: List[Tuple[bytes, bytes]],
            special_tokens: List[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.merges_dict = {merge: i for i, merge in enumerate(merges)}
        # inverse vocabulary for decoding
        self.inv_vocab = {v: k for k,v in vocab.items()}

        # check if special_tokens was valid
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            special_PAT = f'({'|'.join(map(re.escape, self.special_tokens))})'
            self.special_PAT = re.compile(special_PAT)
        else:
            self.special_tokens = []

        # pre_token pattern
        self.extract_PAT = re.compile(PAT)

    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: List[str] | None = None):

        # load vocab_file 
        # json: {token: id}
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            raw_vocab = json.load(f)
        # rearrange
        vocab = {idx: t.encode('utf-8') for t, idx in raw_vocab.items()}

        # load merges_file (txt)
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # check null line
                if not line:
                    continue
                
                # check length
                parts = line.split()
                if len(parts) == 2:
                    merges.append((parts[0].encode('utf-8'), parts[1].encode('utf-8')))
        
        return cls(vocab, merges, special_tokens)
    
    def chunk_encode(
        self,
        text: str,    
    ) -> List[int]:
        '''
        encode the text splited by special tokens
        '''
        encode_result = []
        if not text:
            return encode_result

        for matching in self.extract_PAT.finditer(text):
            pre_token = matching.group(0).encode('utf-8')
            pre_token = [bytes([t]) for t in pre_token] # List[bytes]

            while True:
                # if all bytes are merged, end the loop
                if len(pre_token) == 1:
                    break
                # get all pairs
                pairs = set(zip(pre_token[:-1], pre_token[1:]))
                valid_pairs = [(self.merges_dict[p], p) for p in pairs if p in self.merges_dict]
                # merge ends when there no valid pairs
                if len(valid_pairs) == 0:
                    break
                # get the merge_pair
                _, merge_pair = min(valid_pairs)
                # merge the pre_token
                new_token = []
                i = 0
                n = len(pre_token)
                while i < n:
                    if i < n - 1 and (pre_token[i], pre_token[i+1]) == merge_pair:
                        new_token.append(pre_token[i] + pre_token[i+1])
                        i += 2
                    else:
                        new_token.append(pre_token[i])
                        i += 1
                pre_token = new_token

            # encode the merged pre tokens
            for t in pre_token:
                encode_result.append(self.inv_vocab[t])
        
        return encode_result


    def encode(
            self, 
            text: str
    ) -> List[int]:
                
        # pre_tokenization
        encode_result = []
        # check if there were special tokens
        if not self.special_tokens:
            return self.chunk_encode(text)
        else:
            # split by special token
            for i, chunk_text in enumerate(self.special_PAT.split(text)):
                # encode the special token directly
                if i % 2 == 1:
                    encode_result.append(self.inv_vocab[chunk_text.encode('utf-8')])
                else:
                    encode_result.extend(self.chunk_encode(chunk_text))

        return encode_result
    
    def encode_iterable(
            self,
            iterable: Iterable[str]
        ) -> Iterator[int]:

        # each chunk is a row
        # as the \r is captured by \s+, we never cross the boundary
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(
            self, 
            idxs: list[int]
        ) -> str:
        return (b''.join([self.vocab[idx] for idx in idxs])).decode('utf-8', errors='replace')

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