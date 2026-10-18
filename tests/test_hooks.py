"""Hook infrastructure on a tiny randomly-initialised Gemma-2 (CPU, no download).

The design spec's acceptance test for utils/hooks.py: cache-then-patch-self must reproduce
the plain logits exactly, and a zero dose must be a no-op. Real-model behaviour is checked on
the GPU smoke run; these tests pin down the mechanics.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from utils.hooks import patch, residual_cache, steer


@pytest.fixture(scope="module")
def tiny():
    cfg = transformers.Gemma2Config(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=64, attn_implementation="eager")
    torch.manual_seed(0)
    model = transformers.Gemma2ForCausalLM(cfg).eval()
    ids = torch.randint(3, 128, (2, 9))
    return model, ids


def _logits(model, ids):
    with torch.inference_mode():
        return model(input_ids=ids).logits


def test_zero_dose_is_bit_identical(tiny):
    model, ids = tiny
    plain = _logits(model, ids)
    with steer(model, 1, torch.ones(32), 0.0):
        steered = _logits(model, ids)
    assert torch.equal(plain, steered)


def test_steer_adds_exact_offset_at_every_position(tiny):
    model, ids = tiny
    v = torch.randn(32)
    with residual_cache(model, [1]) as plain, torch.inference_mode():
        model(input_ids=ids)
    with steer(model, 1, v, 2.5), residual_cache(model, [1]) as moved, torch.inference_mode():
        model(input_ids=ids)
    diff = moved[1].float() - plain[1].float()
    assert torch.allclose(diff, (2.5 * v).expand_as(diff), atol=1e-5)


def test_steer_changes_logits_and_hook_is_removed(tiny):
    model, ids = tiny
    plain = _logits(model, ids)
    with steer(model, 1, torch.randn(32), 4.0):
        steered = _logits(model, ids)
    assert not torch.allclose(plain, steered)
    assert torch.equal(_logits(model, ids), plain)          # no leftover hook


def test_patch_self_reproduces_logits(tiny):
    model, ids = tiny
    plain = _logits(model, ids)
    with residual_cache(model, [1]) as store, torch.inference_mode():
        model(input_ids=ids)
    rows = [r for r in range(2) for _ in range(9)]
    pos = [p for _ in range(2) for p in range(9)]
    donor = store[1][rows, pos]
    with patch(model, 1, rows, pos, donor):
        patched = _logits(model, ids)
    assert torch.equal(plain, patched)


def test_patch_last_layer_everywhere_gives_donor_logits(tiny):
    model, ids = tiny
    donor_ids = torch.randint(3, 128, (2, 9))
    last = model.config.num_hidden_layers - 1
    with residual_cache(model, [last]) as store, torch.inference_mode():
        donor_logits = model(input_ids=donor_ids).logits
    rows = [r for r in range(2) for _ in range(9)]
    pos = [p for _ in range(2) for p in range(9)]
    with patch(model, last, rows, pos, store[last][rows, pos]):
        patched = _logits(model, ids)
    assert torch.allclose(patched, donor_logits, atol=1e-5)


def test_patch_single_position_only_moves_that_row(tiny):
    model, ids = tiny
    plain = _logits(model, ids)
    with patch(model, 0, [0], [8], torch.randn(1, 32)):
        patched = _logits(model, ids)
    assert torch.equal(plain[1], patched[1])                 # other row untouched
    assert not torch.allclose(plain[0, 8], patched[0, 8])   # the patched position moved
    assert torch.equal(plain[0, :8], patched[0, :8])        # causal: earlier positions untouched


def test_patch_rejects_ragged_inputs(tiny):
    model, _ = tiny
    with pytest.raises(ValueError):
        with patch(model, 0, [0, 1], [3], torch.zeros(2, 32)):
            pass
