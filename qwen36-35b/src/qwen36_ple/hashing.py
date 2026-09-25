import math

import torch


MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB
LAYER_PRIME = 10007


def splitmix64(value):
    value = (value + SPLITMIX_GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * SPLITMIX_M1) & MASK64
    value = ((value ^ (value >> 27)) * SPLITMIX_M2) & MASK64
    return (value ^ (value >> 31)) & MASK64


def layer_multipliers(vocab_size, ngram_size, ple_layer_index=0, seed=1234):
    multiplier_max = ((1 << 63) - 1) // max(vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + LAYER_PRIME * ple_layer_index
    return tuple(
        2 * (splitmix64((base_seed + SPLITMIX_GAMMA * (index + 1)) & MASK64) % half_bound) + 1
        for index in range(ngram_size)
    )


def is_prime(value):
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def nth_prime_after(start, count):
    prime = start
    for _ in range(count):
        prime += 1
        while not is_prime(prime):
            prime += 1
    return prime


def head_layout(ngram_size=3, heads_per_ngram=8, vocab_size_base=20_000_000, ple_layer_index=0):
    head_count = (ngram_size - 1) * heads_per_ngram
    sizes = tuple(
        nth_prime_after(vocab_size_base - 1, ple_layer_index * head_count + head + 1)
        for head in range(head_count)
    )
    offsets = []
    total = 0
    for size in sizes:
        offsets.append(total)
        total += size
    return sizes, tuple(offsets)


def shift_right_ignore_eos(token_ids, shift, eos_token_id):
    if shift == 0:
        return token_ids
    batch_size, sequence_length = token_ids.shape
    positions = torch.arange(sequence_length, device=token_ids.device, dtype=torch.long)
    eos_positions = torch.where(token_ids == eos_token_id, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat(
        [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1
    )
    position_in_segment = positions.unsqueeze(0) - (previous_eos + 1)
    source_positions = positions - shift
    gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
    shifted = token_ids.gather(dim=1, index=gather_positions)
    valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
    return torch.where(valid, shifted, token_ids.new_full((), eos_token_id))


def ngram_indices(
    input_ids,
    eos_token_id=248044,
    vocab_size=248320,
    ngram_size=3,
    heads_per_ngram=8,
    vocab_size_base=20_000_000,
    ple_layer_index=0,
    seed=1234,
):
    input_ids = input_ids.long()
    multipliers = torch.tensor(
        layer_multipliers(vocab_size, ngram_size, ple_layer_index, seed),
        device=input_ids.device,
        dtype=torch.long,
    )
    sizes, offsets = head_layout(ngram_size, heads_per_ngram, vocab_size_base, ple_layer_index)
    sizes = torch.tensor(sizes, device=input_ids.device, dtype=torch.long)
    offsets = torch.tensor(offsets, device=input_ids.device, dtype=torch.long)
    shifted = [shift_right_ignore_eos(input_ids, shift, eos_token_id) for shift in range(ngram_size)]
    blocks = []
    for ngram in range(2, ngram_size + 1):
        start = (ngram - 2) * heads_per_ngram
        mixed = shifted[0] * multipliers[0]
        for position in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[position] * multipliers[position])
        blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes[start : start + heads_per_ngram]) + offsets[start : start + heads_per_ngram])
    return torch.cat(blocks, dim=-1)
