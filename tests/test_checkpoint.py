import tempfile

import torch

from models.decoder import Decoder
from train import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip_preserves_weights():
    dec = Decoder(vocab_size=20, hidden_dim=8, embedding_dim=8, dropout=0.0)
    opt = torch.optim.Adam(dec.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/ckpt.pt"
        save_checkpoint(path, dec, opt, sch, epoch=3, best_val_loss=0.42)

        dec2 = Decoder(vocab_size=20, hidden_dim=8, embedding_dim=8, dropout=0.0)
        opt2 = torch.optim.Adam(dec2.parameters(), lr=1e-3)
        sch2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2)
        epoch, best = load_checkpoint(path, dec2, opt2, sch2)

        for k in dec.state_dict():
            assert torch.allclose(dec.state_dict()[k], dec2.state_dict()[k])
        assert epoch == 3
        assert abs(best - 0.42) < 1e-6
