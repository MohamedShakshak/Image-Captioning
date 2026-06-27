import torch

from models.encoder import Encoder


def test_encoder_output_shape():
    enc = Encoder()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = enc(x)
    assert out.shape == (2, 2048)
