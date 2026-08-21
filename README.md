# Dual-Channel Token Recurrence Experiment

Working name: **"Say Dog, Think Dog/Wolf"**

## Hypothesis

Ordinary autoregressive generation collapses the model's pre-token
uncertainty into a single hard token at every step:

```
hidden state -> probability distribution -> hard token y -> display y
                                                       \-> feed E[y] back in
```

This project tests a single causal question in a real pretrained LM
(GPT-2), **inference-only** (no fine-tuning, no adapters, no weight
changes, no LLM judges, no sampling in the primary experiment):

> Can two runs display exactly the same token to the human, but think
> differently afterward, because one run internally consumes the hard
> token embedding `E[y]` while the other consumes a richer soft
> representation of the model's pre-token uncertainty
> (`sum_i p_i E[token_i]`)?

## Install

```
pip install -r requirements.txt
```

## Run

Full suite (sanity check, free-run continuations, dose response,
locked-visible impulse response, negative controls, WikiText-2 scan,
plot, report):

```
python -m src.experiments --model gpt2 --all
```

Single prompt:

```
python -m src.experiments \
    --model gpt2 \
    --prompt "Although it resembled a wolf, genetic testing showed that the domesticated animal was a" \
    --top-k 3 \
    --steps 40
```

Useful flags: `--seed`, `--locked-steps`, `--beta-control`,
`--scan-n 0` (skip the corpus scan), `--outdir`.

Tests:

```
python -m pytest tests/
```

## Experiments

1. **One-shot internal handoff** — at one branch point, feed each branch
   a different internal embedding (`hard` = `E[y]`, `soft_raw` = top-k
   probability mixture, `soft_normmatched` = mixture rescaled to
   `||E[y]||`) via `inputs_embeds` + `past_key_values`, then return to
   ordinary greedy decoding. The visible token is identical in every
   branch.
2. **Dose response** — controlled top-1/top-2 mixtures
   `c(beta) = (1-beta) E[y] + beta E[z]` for beta in
   {0, .05, .1, .2, .3, .4, .5}, raw and norm-matched.
3. **Locked-visible impulse response** — after the injection, the hard
   branch's argmax token is fed identically to all branches, so visible
   text never diverges; the only persistent difference is KV state at
   the branch position. Per-step JS/KL divergence measures how long the
   perturbation echoes.
4. **Negative controls** — norm-matched top1/top2 mixture (B), random
   perturbation orthogonal to `E[y]` at the same L2 distance (C), and a
   mixtures with an unrelated token (D) — both norm-matched and
   distance-matched to control B's displacement — to separate
   semantic-mixture effects from generic perturbation effects.

## Layout

```
prompts.txt              frozen hand-written prompt set (not tuned post hoc)
src/model_utils.py       loading, seeding, environment metadata
src/branch.py            internal-embedding constructions + controls
src/generate.py          manual forward() loops; no model.generate()
src/metrics.py           JS/KL/top-k-overlap with numerical safeguards
src/experiments.py       CLI driver for all experiments
src/report.py            report.md + latent_impulse_response.png
tests/                   embedding equivalence, beta-zero, cache isolation
outputs/                 results.jsonl, locked_results.jsonl, report.md, plot
```

## Interpretation discipline

A positive result establishes **only** that the hard visible token and
the internal representation fed back at that boundary can be separated,
and that alternative internal handoffs causally alter subsequent model
behavior. It does not establish that soft handoffs are "better", that
the mixture is the true latent state, or anything about cognition. The
V1 report is purely descriptive.

V2 (carrying forward a projection of the actual contextual hidden state
instead of a vocabulary-space mixture) is deliberately **not**
implemented here.
