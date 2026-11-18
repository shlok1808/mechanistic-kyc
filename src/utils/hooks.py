"""Forward-hook infrastructure shared by the interp battery (S4 cache, S6 patch, S7/S8 steer).

This is the one piece of real infrastructure in the project: thin context managers over
PyTorch forward hooks on `model.model.layers[i]`. Every downstream interp script is a
consumer of this module, which is what keeps the components swappable.

`residual_cache` (S4), `steer` (S7 steering, S8 self-vs-client) and `patch` (S6) all edit or
read the same site: the hidden state a decoder layer returns, i.e. the residual stream after
that block. Hooks registered later see the output of hooks registered earlier, so a
`residual_cache` opened inside a `steer` block records the steered stream.

torch is imported lazily inside the context managers so this file is importable on a box
without PyTorch (the pure offset logic in s4 lives there, not here).
"""

from contextlib import contextmanager


@contextmanager
def residual_cache(model, layers):
    """Capture the residual-stream output of each decoder layer in `layers`.

    Yields a dict `store` that, after the forward pass inside the `with` block, holds
    `{layer_idx: tensor[B, T, D]}` -- the hidden states *after* that decoder block (the
    residual stream). Tensors are detached but stay on the model's device; the caller
    indexes the positions it wants and moves the result to CPU.

    A Gemma-2 / Llama decoder layer returns a tuple whose first element is the hidden
    state; we take `out[0]` (and tolerate a bare-tensor return for safety).
    """
    store = {}
    handles = []

    def make_hook(idx):
        def hook(_module, _inp, out):
            store[idx] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    for i in layers:
        handles.append(model.model.layers[i].register_forward_hook(make_hook(i)))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def _hidden(out):
    """The hidden state inside a decoder layer's return value (tuple or bare tensor)."""
    return out[0] if isinstance(out, tuple) else out


def _replace_hidden(out, h):
    """`out` with its hidden state swapped for `h`, keeping the original structure."""
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


@contextmanager
def steer(model, layer, vector, alpha):
    """Add `alpha * vector` to the residual stream after decoder layer `layer`, at every
    position (prompt and any generated tokens).

    The sum is taken in float32 and cast back, so a small dose is not lost to bf16
    rounding. `alpha == 0` registers nothing, which makes the unsteered baseline
    bit-identical to a plain forward pass.
    """
    import torch
    if alpha == 0:
        yield
        return
    delta = alpha * torch.as_tensor(vector, dtype=torch.float32)

    def hook(_module, _inp, out):
        h = _hidden(out)
        steered = (h.float() + delta.to(h.device)).to(h.dtype)
        return _replace_hidden(out, steered)

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def steer_from(model, layer, vector, alpha, starts):
    """Like `steer`, but only at positions >= starts[b] in row b (PADDED batch indices).
    Used to push a direction on the tokens after the client narrative only, so the client's
    own tokens are read unsteered. Meant for single scoring passes of one batch shape.
    """
    import torch
    if alpha == 0:
        yield
        return
    delta = alpha * torch.as_tensor(vector, dtype=torch.float32)
    starts_t = torch.as_tensor(starts, dtype=torch.long)

    def hook(_module, _inp, out):
        h = _hidden(out)
        pos = torch.arange(h.shape[1], device=h.device)
        mask = (pos[None, :] >= starts_t.to(h.device)[:, None]).unsqueeze(-1)   # [B, T, 1]
        steered = (h.float() + mask * delta.to(h.device)).to(h.dtype)
        return _replace_hidden(out, steered)

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def patch(model, layer, rows, positions, donor):
    """Overwrite the residual after decoder layer `layer` at (rows[k], positions[k]) with
    `donor[k]`, for every k. Positions index the PADDED batch, so callers add each row's
    left-pad count first.

    Applies to the forward pass that matches the batch shape it was built for; it is meant
    for single scoring passes, not generation.
    """
    import torch
    rows_t = torch.as_tensor(rows, dtype=torch.long)
    pos_t = torch.as_tensor(positions, dtype=torch.long)
    donor_t = torch.as_tensor(donor)
    if not (len(rows_t) == len(pos_t) == len(donor_t)):
        raise ValueError("rows, positions and donor must have the same length")

    def hook(_module, _inp, out):
        h = _hidden(out).clone()
        h[rows_t.to(h.device), pos_t.to(h.device)] = donor_t.to(h.device, h.dtype)
        return _replace_hidden(out, h)

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
