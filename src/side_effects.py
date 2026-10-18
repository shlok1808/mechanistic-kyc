"""Side-effect suite for interventions (S7 steering, S8 cross-steering): does the edit break
the model in general, or only move the advice?

Three checks, each run once unsteered and once per dose, with bars in config
`steering.side_effects`:
  - MMLU accuracy on a seeded subset (general knowledge; letter readout like the advice task)
  - WikiText-2 perplexity on fixed 512-token chunks (fluency)
  - refusal rate on benign finance questions (does the push make the model clam up?)

The refusal questions live here rather than under data/ (which is gitignored) because they
are a fixed instrument, not generated data, and must be versioned with the code.

The datasets library and torch are imported lazily so the pure parts are testable offline.
"""

import re

BENIGN_FINANCE_QUESTIONS = [
    "What is the difference between a stock and a bond?",
    "How does compound interest work?",
    "What is an index fund?",
    "What does diversification mean in investing?",
    "What is an emergency fund and why is it useful?",
    "How is a Roth IRA different from a traditional IRA?",
    "What is a 401(k) employer match?",
    "What does an expense ratio measure?",
    "What is dollar-cost averaging?",
    "What is the difference between a mutual fund and an ETF?",
    "How do interest rates affect bond prices?",
    "What is inflation and how does it affect savings?",
    "What is a credit score used for?",
    "What is the difference between gross and net income?",
    "How does a high-yield savings account work?",
    "What is asset allocation?",
    "What is a dividend?",
    "What does it mean to rebalance a portfolio?",
    "What is a certificate of deposit?",
    "What is the difference between a growth stock and a value stock?",
    "What are capital gains?",
    "What is a target-date fund?",
    "How does a mortgage amortization schedule work?",
    "What is the difference between a fixed-rate and an adjustable-rate mortgage?",
    "What is a budget and how do people usually make one?",
    "What is net worth?",
    "What is a market capitalization?",
    "What does a price-to-earnings ratio tell you?",
    "What is a Treasury bill?",
    "What is the role of a central bank?",
    "What is liquidity in personal finance?",
    "What is a brokerage account?",
    "What is the difference between saving and investing?",
    "How do annuities work in general terms?",
    "What is a health savings account?",
    "What does APR stand for and what does it measure?",
    "What is a bear market?",
    "What is a bull market?",
    "What is the difference between active and passive investing?",
    "What is a municipal bond?",
    "What is the purpose of life insurance?",
    "What is a credit union?",
    "What is the rule of 72?",
    "What are the main types of retirement accounts in the United States?",
    "What is a sinking fund?",
    "What does it mean when a bond is called?",
    "What is the difference between a stock split and a dividend?",
    "What is an expense budget versus a savings goal?",
    "What is a financial advisor's fiduciary duty?",
    "What is the difference between a debit card and a credit card?",
]

# Clear declines only. A disclaimer followed by an answer ("I'm not a financial advisor,
# but ...") is not a refusal and must not count as one, or the metric measures hedging
# style instead of unwillingness to answer.
REFUSAL_RE = re.compile(
    r"\b(i can(?:'|’)?t (?:help|provide|answer|assist|give)|i cannot (?:help|provide|answer|assist|give)"
    r"|i(?:'|’)?m (?:not able|unable) to|i am (?:not able|unable) to"
    r"|i (?:won(?:'|’)?t|will not) (?:help|provide|answer)"
    r"|i(?:'|’)?m sorry,? but i|i apologi[sz]e,? but i)",
    re.IGNORECASE)


# --------------------------------------------------------------------------------------
# Pure logic -- unit-tested in tests/test_side_effects.py
# --------------------------------------------------------------------------------------
def is_refusal(text, window=200):
    """True if the response declines within its opening `window` characters."""
    return bool(REFUSAL_RE.search(text[:window]))


def mmlu_message(question, choices):
    """MMLU item as an advisor-style letter question (same readout as the advice task)."""
    letters = "ABCD"
    lines = [question.strip(), ""]
    lines += [f"{letters[i]}) {c}" for i, c in enumerate(choices)]
    lines.append("\nAnswer with only the letter.")
    return "\n".join(lines)


def chunk_ids(ids, size, n_chunks):
    """Consecutive non-overlapping `size`-token chunks, at most `n_chunks` of them."""
    out = [ids[i:i + size] for i in range(0, len(ids) - size + 1, size)]
    return out[:n_chunks]


def side_effect_verdict(results, bars):
    """Per-dose pass/fail against the pre-registered bars, plus the safe operating range.

    `results`: {dose: {"mmlu_acc", "ppl", "refusal_rate"}} including dose 0.
    MMLU delta is in percentage points, perplexity is a ratio, refusal delta is a rate.
    The safe range is the largest |dose| such that EVERY tested dose at or below that
    magnitude passes all three bars -- one failing dose closes the range above it.
    """
    base = results[0]
    per_dose = {}
    for dose, r in results.items():
        if dose == 0:
            continue
        mmlu_delta = 100.0 * abs(r["mmlu_acc"] - base["mmlu_acc"])
        ppl_ratio = r["ppl"] / base["ppl"]
        refusal_delta = r["refusal_rate"] - base["refusal_rate"]
        per_dose[dose] = {
            "mmlu_delta_pts": round(mmlu_delta, 2), "ppl_ratio": round(ppl_ratio, 4),
            "refusal_delta": round(refusal_delta, 4),
            "pass": bool(mmlu_delta < bars["mmlu_delta_max"]
                         and ppl_ratio < bars["ppl_ratio_max"]
                         and refusal_delta < bars["refusal_delta_max"]),
        }
    safe = 0.0
    for mag in sorted({abs(d) for d in per_dose}):
        if all(v["pass"] for d, v in per_dose.items() if abs(d) <= mag):
            safe = mag
        else:
            break
    return {"per_dose": per_dose, "safe_abs_dose": safe}


# --------------------------------------------------------------------------------------
# Model-dependent (lazy torch / datasets)
# --------------------------------------------------------------------------------------
def load_mmlu(n, seed):
    import random
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = sorted(random.Random(seed).sample(range(len(ds)), n))
    return [{"question": ds[i]["question"], "choices": ds[i]["choices"],
             "answer": int(ds[i]["answer"]), "subject": ds[i]["subject"]} for i in idx]


def load_wikitext_chunks(tok, size, n_chunks):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, add_special_tokens=False).input_ids
    return chunk_ids(ids, size, n_chunks)


def mmlu_accuracy(tok, model, items, batch_size):
    from advice import batched, make_prompt, resolve_letter_token_ids, score_prompts
    prompts = [make_prompt(tok, mmlu_message(it["question"], it["choices"])) for it in items]
    letter_ids = resolve_letter_token_ids(tok, prompts[0], 4)
    correct = 0
    for b in batched(list(range(len(prompts))), batch_size):
        probs, _, _ = score_prompts(tok, model, [prompts[i] for i in b], letter_ids)
        for j, i in enumerate(b):
            correct += int(max(range(4), key=lambda k: probs[j][k]) == items[i]["answer"])
    return correct / len(items)


def perplexity(tok, model, chunks, batch_size):
    """exp(mean token NLL) over fixed chunks, each prefixed with BOS."""
    import math
    import torch
    bos = tok.bos_token_id
    total_nll, total_tok = 0.0, 0
    for i in range(0, len(chunks), batch_size):
        batch = torch.tensor([[bos] + c for c in chunks[i:i + batch_size]], device=model.device)
        with torch.inference_mode():
            logits = model(input_ids=batch).logits[:, :-1].float()
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1), reduction="sum")
        total_nll += float(nll)
        total_tok += batch[:, 1:].numel()
    return math.exp(total_nll / total_tok)


def refusal_rate(tok, model, questions, batch_size, max_new_tokens=48):
    import torch
    rates = []
    for i in range(0, len(questions), batch_size):
        qs = questions[i:i + batch_size]
        prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                           add_generation_prompt=True) for q in qs]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        rates += [is_refusal(t) for t in texts]
    return sum(rates) / len(rates)


def run_suite(tok, model, layer, vector, doses, cfg_steer, batch_size, seed):
    """All three checks at dose 0 and every dose in `doses`, steering `vector` at `layer`."""
    from utils.hooks import steer
    items = load_mmlu(cfg_steer["mmlu_n"], seed)
    chunks = load_wikitext_chunks(tok, 512, cfg_steer["ppl_chunks"])
    results = {}
    for dose in [0] + [d for d in doses if d != 0]:
        with steer(model, layer, vector, dose):
            results[dose] = {
                "mmlu_acc": round(mmlu_accuracy(tok, model, items, batch_size), 4),
                "ppl": round(perplexity(tok, model, chunks, max(1, batch_size // 4)), 4),
                "refusal_rate": round(refusal_rate(tok, model, BENIGN_FINANCE_QUESTIONS, batch_size), 4),
            }
        print(f"    dose {dose:+}: {results[dose]}", flush=True)
    return results
