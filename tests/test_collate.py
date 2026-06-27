import torch

from data.dataset import collate_fn
from data.vocab import PAD_IDX


def test_collate_pads_to_batch_max():
    feats = torch.randn(2, 2048)
    cap_a = [1, 5, 6, 2]
    cap_b = [1, 7, 2]
    features, captions, lengths = collate_fn([(feats[0], cap_a), (feats[1], cap_b)])
    assert features.shape == (2, 2048)
    assert captions.shape == (2, 4)
    assert int(lengths.max()) == 4
    # Padding positions should equal PAD_IDX
    assert int(captions[1, 3]) == PAD_IDX


def test_collate_lengths_correct():
    feats = torch.randn(3, 2048)
    caps = [[1, 2], [1, 3, 2], [1, 4, 5, 2]]
    _, _, lengths = collate_fn([(feats[i], caps[i]) for i in range(3)])
    assert lengths.tolist() == [2, 3, 4]
