import torch

from inference import beam_search
from models.decoder import Decoder


def _make_decoder():
    torch.manual_seed(0)
    return Decoder(vocab_size=50, hidden_dim=16, embedding_dim=16, dropout=0.0)


def test_beam_search_deterministic_across_runs():
    dec = _make_decoder().eval()
    feats = torch.randn(1, 2048)
    torch.manual_seed(1)
    a = beam_search(dec, feats, beam_size=3, max_len=10)
    torch.manual_seed(2)
    b = beam_search(dec, feats, beam_size=3, max_len=10)
    assert a == b


def test_beam_search_returns_at_least_one():
    dec = _make_decoder().eval()
    feats = torch.randn(1, 2048)
    results = beam_search(dec, feats, beam_size=3, max_len=10)
    assert len(results) >= 1
