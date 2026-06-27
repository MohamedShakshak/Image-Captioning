"""Beam search + greedy decode functions. CPU-testable, used by evaluate.py and Captioner.

See PLAN.md § Inference masking / Beam search implementation."""

from __future__ import annotations

import torch

from data.vocab import END_IDX, PAD_IDX, START_IDX, UNK_IDX


def greedy_decode(
    decoder,
    features: torch.Tensor,
    max_len: int = 20,
) -> list[int]:
    """Single-image greedy decoding. Returns token indices ending with <end>."""
    device = features.device
    seq = [START_IDX]
    h = torch.tanh(decoder.init_h(features)).unsqueeze(0)
    c = torch.tanh(decoder.init_c(features)).unsqueeze(0)
    for _ in range(max_len):
        x = torch.tensor([seq[-1]], device=device)
        emb = decoder.embed(x).unsqueeze(0)  # (1, 1, E)
        out, (h, c) = decoder.lstm(emb, (h, c))
        logits = decoder.head(out.squeeze(0))  # (1, V)
        logits[:, UNK_IDX] = float("-inf")
        logits[:, PAD_IDX] = float("-inf")
        next_tok = int(logits.argmax(-1).item())
        seq.append(next_tok)
        if next_tok == END_IDX:
            break
    return seq[1:]  # drop <start>


def beam_search(
    decoder,
    features: torch.Tensor,
    beam_size: int = 3,
    max_len: int = 20,
    length_norm_alpha: float = 0.7,
) -> list[tuple[list[int], float]]:
    """Single-image beam search. Returns list of (tokens, normalized_score) sorted desc."""
    device = features.device

    # Initial hidden state from image features
    h = torch.tanh(decoder.init_h(features)).unsqueeze(0)  # (1, B=1, H)
    c = torch.tanh(decoder.init_c(features)).unsqueeze(0)

    # Beam state: list of (tokens, score, h, c)
    beams = [(torch.tensor([START_IDX], device=device), 0.0, h, c, False)]
    finished: list[tuple[list[int], float]] = []

    for _ in range(max_len):
        if not beams:
            break
        # Expand each beam by one step
        candidates = []
        for tokens, score, hh, cc, _done in beams:
            x = tokens[-1:].unsqueeze(0)  # (1, 1)
            emb = decoder.embed(x)  # (1, 1, E)
            out, (nh, nc) = decoder.lstm(emb, (hh, cc))
            logits = decoder.head(out[:, -1, :])  # (1, V)
            logits[:, UNK_IDX] = float("-inf")
            logits[:, PAD_IDX] = float("-inf")
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)  # (V,)
            top_lp, top_idx = log_probs.topk(beam_size)
            for i in range(beam_size):
                tok = int(top_idx[i].item())
                lp = float(top_lp[i].item())
                new_tokens = torch.cat([tokens, torch.tensor([tok], device=device)])
                new_score = score + lp
                candidates.append((new_tokens, new_score, nh, nc, tok == END_IDX))

        # Prune to beam_size on raw cumulative score
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_size]

        # Move finished beams aside (length-normalised)
        still_alive = []
        for tokens, score, h_n, c_n, done in beams:
            if done:
                # tokens[1:-1] drops <start> and <end>
                length = tokens.numel() - 1
                norm_score = score / (length**length_norm_alpha)
                finished.append((tokens[1:-1].tolist(), norm_score))
            else:
                still_alive.append((tokens, score, h_n, c_n, done))
        beams = still_alive

        if not beams:
            break

    # Fallback: if no beam finished (e.g. all max_len exhausted without <end>),
    # return the top alive beam. If none, return a trivial seq.
    if not finished:
        if beams:
            tokens, score, *_ = beams[0]
        else:
            tokens, score = torch.tensor([START_IDX, END_IDX], device=device), 0.0
        length = max(1, tokens.numel() - 1)
        norm_score = score / (length ** length_norm_alpha)
        finished.append((tokens[1:].tolist(), norm_score))

    finished.sort(key=lambda x: x[1], reverse=True)
    return finished
