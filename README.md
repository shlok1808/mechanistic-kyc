# Mechanistic KYC

Looking inside an LLM financial advisor to audit *Know Your Customer* (suitability).

**Thesis:** Behavioral testing can show an AI advisor gave bad advice. Only mechanistic
(internal) testing shows *why* — and the *why* determines the fix.

> Status: redesigned September 2026 after the project review (see below). Data
> regenerated and quality-checked; model runs next.

## Project review

The September 2026 [audit and literature review](AUDIT_AND_LITERATURE_REVIEW.md)
documents the current pipeline, methodological risks, related research, and revised
experiment plan.

## Research questions
- **RQ1 (Existence):** Does the model hold a linearly-decodable internal estimate of the
  client's **willingness**, **capacity** and **goals** separately, even when the client
  never uses risk words?
- **RQ2 (Causation):** Does that estimate *causally drive* advice? (patching, steering,
  and the novelty: is the *client's* risk direction distinct from the model's *own*?)
- **RQ3 (Pressure):** Under client pressure, is it *belief capture* (internal estimate
  moves) or *hypocritical compliance* (estimate holds, advice caves)?

## Pipeline
Each stage is one script in `src/`, named for what it does.

| Script | What it does |
|--------|--------------|
| `profiles.py` | Sample clients, balanced over the willingness x capacity grid |
| `vignettes.py` | Render explicit/implicit twins + one-factor counterfactual pairs |
| `qc.py` | Banned-lexicon and coherence checks on the rendered text |
| `advice.py` | Ask the model for a portfolio recommendation (option logits) |
| `gate.py` | Behavioral go/no-go: does advice track the client? |
| `patterns.py` | Behavioral pattern mining (field weights, conflict collapse) |
| `activations.py` | Cache residual-stream activations for the probe layers |
| `probes.py` | Per-factor linear probes (RQ1), dev-selected layer/position |
| `dissociation.py` | Is a collapse in the readout or in the encoding? |
| `elicit.py` | Verbal-elicitation baseline: just ask the model |
| `report.py`, `figures.py` | Advisor packet and publication figures |

## The client model
Suitability law separates what a client **wants** from what they can **afford**, so the
rubric scores three factors independently rather than blending them into one number:

| Factor | Means | Fields |
|--------|-------|--------|
| `willingness` | comfort with volatility and loss | drawdown reaction, experience |
| `capacity` | objective ability to absorb a loss | income stability, emergency fund, dependents, age, horizon |
| `goals` | what the money is for | stated goal |

A client is **conflicted** when willingness and capacity point opposite ways. Those are
the FINRA 2111 core cases, so profiles are sampled to fill the willingness x capacity
grid evenly rather than to balance the blended tier: balancing a weighted sum cannot
balance its parts. Counterfactual pairs differ in **exactly one** rubric field, which is
what lets an intervention be attributed to a factor rather than to "a different client".

## Models & data
- Primary: `google/gemma-2-9b-it` (bf16). Replication: `meta-llama/Llama-3.1-8B-Instruct`.
- SAEs: Gemma Scope (`google/gemma-scope-9b-it-res`), Llama Scope (`fnlp/Llama-Scope`).
- Data: fully synthetic, self-generated (~6k profiles -> ~8k vignettes). No scraping.

## Pre-registered thresholds
All success thresholds live in `config.yaml` and are meant to be **agreed by the team
before any results are seen** (no moving goalposts). Missing a threshold is not failure —
it changes which claim headlines.

## Layout
    src/         one script per pipeline stage, named for what it does
    notebooks/   Colab dev (prototype on 2B/T4, run on A100)
    data/        generated data (gitignored)
    results/     outputs, figures, metrics (gitignored)
    config.yaml  all experiment params + pre-registered thresholds

Model-specific outputs live under `results/<model-tag>/`, so the model is always a
directory and never part of a filename. A 2B dev run cannot overwrite a 9B production
run, and filenames stay readable (`results/gemma-2-9b-it/probes.json`).

Activation caches sit under `results/<model-tag>/activations/<fingerprint>/`, where the
fingerprint covers the model, the prompt, the data files, the layer set and the dtype.
Changing any of them writes a new cache instead of silently resuming on top of
activations that were produced differently.
