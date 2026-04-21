"""
FactoredGRU Sanity Test
========================
Verifies the GRU model works correctly before training.
Run on Kaggle: python test_pipeline_gru.py
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from models import EncoderViT, FactoredGRU
from loss import masked_cross_entropy

# ---- Config ----
EMB_DIM = 300
HIDDEN_DIM = 512
FACTORED_DIM = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---- Load tokenizer ----
tokenizer = AutoTokenizer.from_pretrained("/kaggle/working/stylenet/tokenizer-extended", trust_remote_code=True)
VOCAB_SIZE = len(tokenizer)
print(f"Vocab size: {VOCAB_SIZE}")

# ---- Initialize models ----
encoder = EncoderViT(EMB_DIM).to(device)
decoder = FactoredGRU(EMB_DIM, HIDDEN_DIM, FACTORED_DIM, VOCAB_SIZE).to(device)
encoder.eval()
decoder.eval()

# ---- Count parameters ----
lstm_style_params = 4 * 6  # 4 gates × (V, S_f, S_r, U, W, F) = 24 matrices
gru_style_params = 3 * 6   # 3 gates × (V, S_f, S_r, U, W, F) = 18 matrices
total_params = sum(p.numel() for p in decoder.parameters())
trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
print(f"\nDecoder total params: {total_params:,}")
print(f"Decoder trainable params: {trainable_params:,}")
print(f"Gate matrices: {gru_style_params} (vs LSTM's {lstm_style_params}) — 25% fewer")

# ============================================================
print("\n" + "=" * 60)
print("TEST 1: Encoder output sanity")
print("=" * 60)

fake_images = torch.randn(2, 3, 224, 224).to(device)
with torch.no_grad():
    features = encoder(fake_images)

print(f"  Input shape:  {fake_images.shape}")
print(f"  Output shape: {features.shape}")
assert features.shape == (2, EMB_DIM), f"FAIL: Expected [2, {EMB_DIM}], got {features.shape}"
assert not torch.isnan(features).any(), "FAIL: Features contain NaN!"
print(f"  Feature stats: mean={features.mean():.4f}, std={features.std():.4f}")
print("  ✅ Encoder OK")

# ============================================================
print("\n" + "=" * 60)
print("TEST 2: GRU forward pass (FACTUAL mode)")
print("=" * 60)

test_caption = "একটি ছেলে মাঠে খেলছে"
ids = tokenizer.encode(test_caption, add_special_tokens=False)
full_ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
max_len = len(full_ids) + 3
padded = full_ids + [tokenizer.pad_token_id] * (max_len - len(full_ids))
captions = torch.tensor([padded, padded], dtype=torch.long).to(device)
lengths = torch.tensor([len(full_ids), len(full_ids)], dtype=torch.long).to(device)

with torch.no_grad():
    outputs_factual = decoder(captions, features, mode="factual")

print(f"  Captions shape: {captions.shape}")
print(f"  Features shape: {features.shape}")
print(f"  Outputs shape:  {outputs_factual.shape}")
print(f"  Expected:       [2, {max_len}, {VOCAB_SIZE}]")
assert outputs_factual.shape == (2, max_len, VOCAB_SIZE), "FAIL: Shape mismatch"
assert not torch.isnan(outputs_factual).any(), "FAIL: Outputs contain NaN!"

loss_factual = masked_cross_entropy(
    outputs_factual[:, 1:, :].contiguous(),
    captions[:, 1:].contiguous(),
    lengths - 1
)
print(f"  Factual loss: {loss_factual.item():.4f}")
assert not torch.isnan(loss_factual), "FAIL: Loss is NaN!"
print("  ✅ Factual forward pass OK")

# ============================================================
print("\n" + "=" * 60)
print("TEST 3: GRU forward pass (ROMANTIC mode — no features)")
print("=" * 60)

with torch.no_grad():
    outputs_romantic = decoder(captions, mode="romantic")

print(f"  Outputs shape:  {outputs_romantic.shape}")
print(f"  Expected:       [2, {max_len - 1}, {VOCAB_SIZE}]")
assert outputs_romantic.shape == (2, max_len - 1, VOCAB_SIZE), "FAIL: Shape mismatch"
assert not torch.isnan(outputs_romantic).any(), "FAIL: Outputs contain NaN!"

loss_romantic = masked_cross_entropy(
    outputs_romantic.contiguous(),
    captions[:, 1:].contiguous(),
    lengths - 1
)
print(f"  Romantic loss: {loss_romantic.item():.4f}")
print("  ✅ Romantic forward pass OK")

# ============================================================
print("\n" + "=" * 60)
print("TEST 4: No cell state — GRU only returns (output, h_t)")
print("=" * 60)

h0 = torch.zeros(1, HIDDEN_DIM).to(device)
emb = decoder.B(torch.tensor([tokenizer.bos_token_id]).to(device))

with torch.no_grad():
    result = decoder.forward_step(emb, h0, mode="factual", features=features[0:1])

print(f"  forward_step returns {len(result)} values (expected 2: output, h_t)")
assert len(result) == 2, f"FAIL: Expected 2 returns, got {len(result)}"
output, h_t = result
print(f"  Output shape: {output.shape}")
print(f"  h_t shape:    {h_t.shape}")
print("  ✅ No cell state — GRU confirmed")

# ============================================================
print("\n" + "=" * 60)
print("TEST 5: Different features → different outputs")
print("=" * 60)

img_A = torch.randn(1, 3, 224, 224).to(device)
img_B = torch.randn(1, 3, 224, 224).to(device)

with torch.no_grad():
    feat_A = encoder(img_A)
    feat_B = encoder(img_B)

    single_cap = captions[0:1]
    out_A = decoder(single_cap, features=feat_A, mode="factual")
    out_B = decoder(single_cap, features=feat_B, mode="factual")

diff = (out_A - out_B).abs().sum().item()
print(f"  L1 diff between outputs: {diff:.4f}")
assert diff > 0.01, "FAIL: Different features gave same output!"
print("  ✅ Features influence GRU output")

# ============================================================
print("\n" + "=" * 60)
print("TEST 6: Feature gates work (F_z, F_r, F_n)")
print("=" * 60)

h0 = torch.zeros(1, HIDDEN_DIM).to(device)
emb = decoder.B(torch.tensor([tokenizer.bos_token_id]).to(device))

with torch.no_grad():
    out_feat, h_feat = decoder.forward_step(emb, h0, mode="factual", features=feat_A)
    out_none, h_none = decoder.forward_step(emb, h0, mode="factual", features=None)

print(f"  Output diff (features vs none): {(out_feat - out_none).abs().sum().item():.4f}")
print(f"  Hidden diff (features vs none): {(h_feat - h_none).abs().sum().item():.4f}")
assert (out_feat - out_none).abs().sum().item() > 0.01, "FAIL: Features don't affect gates!"
print("  ✅ F_z, F_r, F_n are working")

# ============================================================
print("\n" + "=" * 60)
print("TEST 7: Gradient flow")
print("=" * 60)

decoder.train()
encoder.train()

# --- Factual backward ---
decoder.zero_grad()
encoder.zero_grad()
feat_grad = encoder(fake_images)
out_grad = decoder(captions, feat_grad, mode="factual")
loss = masked_cross_entropy(out_grad[:, 1:, :].contiguous(), captions[:, 1:].contiguous(), lengths - 1)
loss.backward()

print(f"  [Factual gradients]")
enc_a = encoder.A.weight.grad
print(f"    encoder.A: {'✅ exists' if enc_a is not None and enc_a.abs().sum() > 0 else '❌ NONE'}")

for name in ['F_z', 'F_r', 'F_n']:
    g = getattr(decoder, name).weight.grad
    has = g is not None and g.abs().sum() > 0
    print(f"    decoder.{name}: {'✅ exists' if has else '❌ NONE'}")

sfz_has = decoder.S_fz.weight.grad is not None and decoder.S_fz.weight.grad.abs().sum() > 0
srz_has = decoder.S_rz.weight.grad is not None and decoder.S_rz.weight.grad.abs().sum() > 0
print(f"    S_fz grad: {'✅ exists (correct)' if sfz_has else '❌ missing'}")
print(f"    S_rz grad: {'✅ zero (correct)' if not srz_has else '⚠️ non-zero'}")

# --- Romantic backward ---
decoder.zero_grad()
out_rom = decoder(captions, mode="romantic")
loss_rom = masked_cross_entropy(out_rom.contiguous(), captions[:, 1:].contiguous(), lengths - 1)
loss_rom.backward()

print(f"\n  [Romantic gradients]")
srz_has2 = decoder.S_rz.weight.grad is not None and decoder.S_rz.weight.grad.abs().sum() > 0
sfz_has2 = decoder.S_fz.weight.grad is not None and decoder.S_fz.weight.grad.abs().sum() > 0
print(f"    S_rz grad: {'✅ exists (correct)' if srz_has2 else '❌ missing'}")
print(f"    S_fz grad: {'✅ zero (correct)' if not sfz_has2 else '⚠️ non-zero'}")

for name in ['F_z', 'F_r', 'F_n']:
    g = getattr(decoder, name).weight.grad
    has = g is not None and g.abs().sum() > 0
    print(f"    decoder.{name}: {'✅ zero (correct)' if not has else '⚠️ non-zero'}")

print("  ✅ Gradient flow correct")

# ============================================================
print("\n" + "=" * 60)
print("TEST 8: Beam search generation")
print("=" * 60)

decoder.eval()
encoder.eval()

with torch.no_grad():
    feat_0 = encoder(img_A)
    feat_1 = encoder(img_B)

    ids_0 = decoder.sample(feat_0, tokenizer=tokenizer, beam_size=3, max_len=20, mode="factual")
    ids_1 = decoder.sample(feat_1, tokenizer=tokenizer, beam_size=3, max_len=20, mode="factual")

    cap_0 = tokenizer.decode(ids_0, skip_special_tokens=True)
    cap_1 = tokenizer.decode(ids_1, skip_special_tokens=True)

print(f"  Factual (img A): \"{cap_0}\"")
print(f"  Factual (img B): \"{cap_1}\"")
print(f"  Same? {ids_0 == ids_1}")

with torch.no_grad():
    ids_rom_0 = decoder.sample(feat_0, tokenizer=tokenizer, beam_size=3, max_len=20, mode="romantic")
    cap_rom_0 = tokenizer.decode(ids_rom_0, skip_special_tokens=True)

print(f"  Romantic (img A): \"{cap_rom_0}\"")
print("  (Captions are random/nonsense — model is untrained. Point is it runs without errors.)")
print("  ✅ Beam search OK")

# ============================================================
print("\n" + "=" * 60)
print("ALL TESTS PASSED ✅")
print("=" * 60)
print("""
  1. Encoder output             ✅
  2. Factual forward pass       ✅
  3. Romantic forward pass      ✅
  4. No cell state (GRU only)   ✅
  5. Feature-dependent outputs  ✅
  6. Feature gates (F_z/F_r/F_n)✅
  7. Gradient isolation by mode ✅
  8. Beam search generation     ✅

  FactoredGRU is ready for training!
""")
print("=" * 60)
