"""High-level captioning API for the demo / CLI / eval. Loads weights from HF Hub."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image

from data.transforms import inference_transforms
from data.vocab import Vocab
from inference import beam_search, greedy_decode
from models.decoder import Decoder


@dataclass
class Beam:
    tokens: list[int]
    score: float


@dataclass
class CaptionResult:
    text: str
    beams: list[Beam] = field(default_factory=list)


class Captioner:
    """Wraps the decoder + vocab + transforms into a callable captioning interface."""

    def __init__(
        self,
        decoder: Decoder,
        vocab: Vocab,
        device: torch.device | str = "cpu",
        beam_size: int = 3,
        length_norm_alpha: float = 0.7,
        max_len: int = 20,
    ) -> None:
        self.decoder = decoder
        self.vocab = vocab
        self.device = torch.device(device)
        self.beam_size = beam_size
        self.length_norm_alpha = length_norm_alpha
        self.max_len = max_len
        self.transform = inference_transforms()
        self.decoder.to(self.device)
        self.decoder.eval()

    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str,
        device: torch.device | str = "cpu",
        beam_size: int = 3,
        length_norm_alpha: float = 0.7,
    ) -> Captioner:
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download(repo_id=hf_repo, filename="best.pt")
        vocab_path = hf_hub_download(repo_id=hf_repo, filename="vocab.json")
        vocab = Vocab.load(vocab_path)
        ckpt = torch.load(weights_path, map_location="cpu")
        decoder = Decoder(vocab_size=vocab.size)
        decoder.load_state_dict(ckpt["model"])
        return cls(
            decoder, vocab, device=device, beam_size=beam_size, length_norm_alpha=length_norm_alpha
        )

    @torch.no_grad()
    def caption(self, image: Image.Image | str | Path, beam: bool = True) -> CaptionResult:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        # NOTE: the cached-feature pipeline doesn't run the encoder here.
        # For the demo, we need a live encoder. The full Captioner used in the
        # demo bundles an Encoder instance too; kept as a TODO for `feat/app`.
        raise NotImplementedError(
            "Live-encode path is wired in `feat/app`. For cached-feature captioning, "
            "load features directly and call Captioner.caption_from_features()."
        )

    @torch.no_grad()
    def caption_from_features(self, features: torch.Tensor, beam: bool = True) -> CaptionResult:
        features = features.to(self.device).unsqueeze(0)  # (1, 2048)
        if beam:
            results = beam_search(
                self.decoder,
                features,
                beam_size=self.beam_size,
                max_len=self.max_len,
                length_norm_alpha=self.length_norm_alpha,
            )
            top = results[0]
            tokens = top[0]
            token_str = self.vocab.decode(tokens)
            beam_objs = [Beam(tokens=t, score=s) for t, s in results]
            return CaptionResult(text=" ".join(token_str), beams=beam_objs)
        tokens = greedy_decode(self.decoder, features, max_len=self.max_len)
        token_str = self.vocab.decode(tokens)
        return CaptionResult(text=" ".join(token_str), beams=[])
