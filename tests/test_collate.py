import torch

from data.dataset import collate_fn
from data.vocab import PAD_IDX


def test_collate_pads_to_batch_max():
    feats = torch.randn(2, 2048)
    cap_a = [1, 5, 6, 2]
    cap_b = [1, 7, 2]
    batch = [(100, feats[0], cap_a), (101, feats[1], cap_b)]
    features, captions, lengths = collate_fn(batch)
    assert features.shape == (2, 2048)
    assert captions.shape == (2, 4)
    assert int(lengths.max()) == 4
    assert int(captions[1, 3]) == PAD_IDX


def test_collate_lengths_correct():
    feats = torch.randn(3, 2048)
    caps = [[1, 2], [1, 3, 2], [1, 4, 5, 2]]
    batch = [(i, feats[i], caps[i]) for i in range(3)]
    _, _, lengths = collate_fn(batch)
    assert lengths.tolist() == [2, 3, 4]
