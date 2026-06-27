import torch

from models.decoder import Decoder


def test_decoder_output_shape():
    dec = Decoder(vocab_size=100, hidden_dim=32, embedding_dim=32, dropout=0.0)
    features = torch.randn(4, 2048)
    captions = torch.randint(0, 100, (4, 7))  # (B, T)
    logits = dec(features, captions)
    assert logits.shape == (4, 7, 100)
