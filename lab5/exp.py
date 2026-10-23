"""Measurements for the lab5 report (Q1-Q6)."""
import pathlib
import numpy as np
from src.attention import causal_mask, scaled_dot_product_attention, MultiHeadAttention
from src.tokenizer import BPETokenizer, compression_ratio

corpus = pathlib.Path("data/the-verdict.txt").read_text(encoding="utf-8")
print(f"corpus: {len(corpus)} chars, {len(corpus.encode('utf-8'))} bytes")

# -- Q4/Q6: BPE trained on the-verdict at two vocab sizes --
for vs in (270, 5000):
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=vs)
    ids = tok.encode(corpus)
    print(f"vocab={vs}: learned {len(tok.merges)} merges, corpus tokens={len(ids)}, "
          f"compression={compression_ratio(corpus, tok):.4f} bytes/token")

# char-level baseline (lab4 tokenizer): 1 token per character
print(f"char-level baseline: {len(corpus)} tokens, "
      f"{len(corpus.encode('utf-8'))/len(corpus):.4f} bytes/char")

# -- Q5: BPE trained on project source, tested on test_tokenizer.py --
files = sorted(pathlib.Path("src").rglob("*.py")) + sorted(pathlib.Path("tests").rglob("*.py")) + [pathlib.Path("visualize.py")]
code = "\n".join(p.read_text(encoding="utf-8") for p in files)
test_file = pathlib.Path("tests/test_tokenizer.py").read_text(encoding="utf-8")
tok_code = BPETokenizer()
tok_code.train(code, vocab_size=5000)
print(f"source corpus: {len(code)} chars / {len(code.encode())} bytes, merges={len(tok_code.merges)}")
print(f"Q5: compression on tests/test_tokenizer.py = {compression_ratio(test_file, tok_code):.4f} bytes/token "
      f"({len(test_file)} chars -> {len(tok_code.encode(test_file))} tokens)")
tok_v = BPETokenizer()
tok_v.train(corpus, vocab_size=5000)
print(f"     baseline: test_tokenizer.py compressed by the-verdict-trained tokenizer = "
      f"{compression_ratio(test_file, tok_v):.4f}")

# -- Q1: attention WITHOUT a causal mask leaks the future --
rng = np.random.RandomState(0)
seq, d = 6, 8
x = rng.randn(seq, d)
q = rng.randn(seq, d) * 0.5
_, w_nomask = scaled_dot_product_attention(q, q, x, mask=None)
print("\nQ1 no-mask row 0 (query at position 0, attends over future positions):")
print(np.round(w_nomask[0], 4))
_, w_masked = scaled_dot_product_attention(q, q, x, mask=causal_mask(seq))
print("Q1/Q2 causal-masked rows:")
print(np.round(w_masked, 4))

# -- Q3: trailing PAD token must not change outputs at real positions --
mha = MultiHeadAttention(d_in=d, d_out=d, num_heads=2, context_length=16, rng=np.random.RandomState(3))
single = rng.randn(1, 1, d)
padded = np.concatenate([single, np.zeros((1, 1, d))], axis=1)  # [hi, PAD]
out_single = mha(single)
out_padded = mha(padded)
print("\nQ3 max |out_single[0,0] - out_padded[0,0]| =", np.abs(out_single[0, 0] - out_padded[0, 0]).max())
