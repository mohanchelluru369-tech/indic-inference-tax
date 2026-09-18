# Paper outline (working)

**Working title:** The Indic Inference Tax: What Indian Languages Pay for the Same Answer on the Hardware India Owns, and How Much of It Can Be Refunded

**One-sentence contribution:** The first paired, end-to-end measurement of the token, latency, energy and quality cost of Hindi and Telugu (native, romanized, code-mixed) relative to English across consumer hardware down to a budget phone, plus a laptop-scale mitigation on an existing open model.

1. **Introduction.** The ₹12,000-phone framing. The tax compounds. Contributions as three bullets: measurement, harness, mitigation.
2. **Background and related work.** From `docs/02_prior_work.md`. Be generous to The Language-Energy Divide and The Tokenizer Tax and precise about what is new.
3. **Method.** Paired design, metrics, controls, hardware matrix. From `docs/03_methodology.md`. This section is the paper's credibility.
4. **Exp 01: tokens.** 22 languages x tokenizers; native vs romanized vs code-mixed for hi/te; fragment rate as mechanism.
5. **Exp 02: what tokens turn into.** TTFT, time, energy per answer by hardware; H1 (decode rate is language-independent) as the control; cut-off rate and script drift; joules per correct answer.
6. **Exp 03-04: the multipliers.** Quantization x language; speculative decoding x language.
7. **The compound tax on a budget phone.**
8. **Exp 05: mitigation.** Indic-calibrated quantization; tokenizer expansion; language-aware drafting; stacked.
9. **Discussion.** The romanization trade-off as a policy-relevant finding. What model builders should report (per-language cost in model cards). What this implies for sovereign-model tokenizer choices.
10. **Limitations.** From the threats-to-validity list. Two languages; small models; open models only.
11. **Reproducibility.** Repo, configs, raw results, hardware.

## Figures to plan for

- F1: tax heat-map, languages x tokenizers.
- F2: for one model on one machine, tax on tokens vs TTFT vs time vs energy, per variant, with intervals.
- F3: same content, four hardware rows, time per answer (log scale).
- F4: accuracy vs bits per language.
- F5: before/after mitigation waterfall.

## Numbers that must exist before writing

Every hypothesis H1-H8 answered yes, no, or "could not measure, because".
