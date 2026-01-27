from collections import defaultdict


class BPE_TRAINER:

    def __init__(self, pre_tokens, special_tokens, vocab_size):
        self.pre_tokens = pre_tokens
        self.special_tokens = special_tokens
        self.vocab_size = vocab_size

    def train(self):
        # vocab initialization
        vocab = {i: bytes([i]) for i in range(256)}
        merges = []

        # add special tokens
        for i, t in enumerate(self.special_tokens):
            vocab[256 + i] = t.encode('utf-8')

        # Convert pre_tokens to a list-based representation for efficient updates
        # Each entry: [list of token bytes, count]
        token_sequences = []
        for token_tuple, count in self.pre_tokens.items():
            token_sequences.append([list(token_tuple), count])

        # Initial pair count and track which sequences contain each pair
        pair_counts = defaultdict(int)
        pair_to_sequences = defaultdict(set)
        
        for seq_idx, (tokens, count) in enumerate(token_sequences):
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += count
                pair_to_sequences[pair].add(seq_idx)

        # Main merge loop
        while len(vocab) < self.vocab_size:
            if not pair_counts:
                break

            # Find the best pair (most frequent, with lexicographic tie-breaking)
            best_pair = None
            best_count = 0
            for pair, count in pair_counts.items():
                if count > best_count or (count == best_count and (best_pair is None or pair > best_pair)):
                    best_count = count
                    best_pair = pair

            if best_pair is None or best_count == 0:
                break

            # Add to vocab and merges
            new_token = best_pair[0] + best_pair[1]
            vocab[len(vocab)] = new_token
            merges.append((best_pair[0], best_pair[1]))

            # Get sequences that contain this pair
            affected_sequences = list(pair_to_sequences.get(best_pair, []))
            
            for seq_idx in affected_sequences:
                tokens, count = token_sequences[seq_idx]
                
                # Remove all current pairs from this sequence
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_counts[pair] -= count
                    pair_to_sequences[pair].discard(seq_idx)
                
                # Merge all occurrences of best_pair
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                
                # Update the sequence
                token_sequences[seq_idx][0] = new_tokens
                
                # Add all new pairs from this sequence
                for i in range(len(new_tokens) - 1):
                    pair = (new_tokens[i], new_tokens[i + 1])
                    pair_counts[pair] += count
                    pair_to_sequences[pair].add(seq_idx)
            
            # Clean up any zero or negative counts
            pairs_to_remove = [p for p, c in pair_counts.items() if c <= 0]
            for p in pairs_to_remove:
                del pair_counts[p]
                if p in pair_to_sequences:
                    del pair_to_sequences[p]

        return (vocab, merges)
