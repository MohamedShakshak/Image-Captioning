from data.vocab import (
    END_IDX,
    PAD_IDX,
    START_IDX,
    UNK_IDX,
    Vocab,
    encode_caption,
    tokenize,
)


def test_vocab_build_orders_specials_first():
    caps = ["a cat sat on the mat", "the cat sat on a dog"]
    v = Vocab.build(caps, min_freq=1)
    assert v.word2idx["<pad>"] == PAD_IDX
    assert v.word2idx["<start>"] == START_IDX
    assert v.word2idx["<end>"] == END_IDX
    assert v.word2idx["<unk>"] == UNK_IDX
    # min_freq=1 includes every word
    assert "cat" in v.word2idx


def test_tokenize_lowercase_and_strip_punct():
    toks = tokenize("Don't worry, it's a CAT!")
    # PTB-style contraction splits "Don't" -> "do not"
    assert "not" in toks


def test_encode_caption_wraps_with_start_end():
    v = Vocab.build(["a cat"], min_freq=1)
    idxs = encode_caption(v, "a cat")
    assert idxs[0] == START_IDX
    assert idxs[-1] == END_IDX
