# 04. Roadmap: ten weeks to a preprint

Week 1 starts Monday 21 September 2026. Target: arXiv submission in the week of 23 November 2026. Evenings-and-weekends pace is assumed; if a week slips, cut scope from the bottom of that week's list, not from the controls in the methodology.

| Week | Dates | Goal | Done when |
|---|---|---|---|
| 1 | 21-27 Sep | **First signal.** Run setup and first measurement on the Mac. Review and correct the seed corpus (Telugu first). Fix any wrong repo ids in `configs/tokenizers.yaml`; add current-generation models. | `results/` holds one fertility run and one clean Exp 02 run; the go/redirect decision in `01_problem_statement.md` is made and written down. |
| 2 | 28 Sep-4 Oct | **Exp 01 at full width.** FLORES+/IN22 across all 22 scheduled languages x 10+ tokenizers. Build Belebele corpus; Exp 02 on the Mac with 3-4 models. | Tokenizer-tax table and first joules-per-correct-answer table. |
| 3 | 5-11 Oct | **Hardware matrix.** Same GGUFs on the RTX workstation and the GCP T4. Get a budget Android phone (about ₹10-15k, 4-6 GB RAM) and run under Termux. | One results table with four hardware rows. |
| 4 | 12-18 Oct | **Exp 03: quantization x language.** Q8, Q5, Q4, Q3, Q2 of two models; accuracy, latency and energy per language. | Accuracy-vs-bits curves per language; H5 answered. |
| 5 | 19-25 Oct | **Exp 04: speculative decoding x language.** llama.cpp draft-model and n-gram lookup decoding; acceptance and speedup per variant. | H6 answered. |
| 6-7 | 26 Oct-8 Nov | **Exp 05: the fix.** (a) Indic-calibrated importance-matrix quantization (cheap, do first). (b) Tokenizer expansion for Telugu + Hindi on one 1-4B model with light continued training (MLX or the RTX box). (c) Language-matched draft model. | Each mitigation has a before/after row measured with the same harness. |
| 8 | 9-15 Nov | **Compound result and ablations.** Stack the mitigations; measure on the phone; find what did not work and say so. | The headline number: how much of the end-to-end tax was closed. |
| 9 | 16-22 Nov | **Write.** Paper from `paper/outline.md`; clean the repo; leaderboard page. Re-run the prior-work search. | Full draft reviewed by at least one outside reader. |
| 10 | 23-29 Nov | **Publish.** arXiv; repo public; one plain-language write-up. Send to AI4Bharat, Sarvam, Krutrim, BharatGen, People+ai with a specific ask (run it on your model). | Preprint live. |

## Checkpoints where the plan can change

- **End of week 1:** if current tokenizers already put hi/te within about 1.5x of English, shift weight from the tokenizer fix to quantization, drafting and the romanization trade-off (see `01_problem_statement.md`).
- **End of week 3:** if the phone cannot run a 1B model at usable speed, that *is* the finding for the on-device section; drop to sub-1B models and report it.
- **End of week 7:** if tokenizer expansion does not hold accuracy, publish the measurement paper with the two cheaper mitigations and report the negative result honestly. A solid measurement paper with a partial fix beats a late paper.

## Where Kneepoint comes in

Not yet, and not as a merge. Kneepoint's own docs draw the line: it measures *agents* under concurrency, and says a bare model endpoint is a serving benchmark that belongs to other tools. This repo is that serving benchmark, single-stream, per language. Bolting it onto Kneepoint would blur the product's positioning while it is mid-launch, so the two stay separate and share only the discipline (paired runs, no invented numbers, JSONL outputs, integrity docs).

The one place they meet is **Exp 06, concurrency by language (week 5-6, after the hardware matrix)**: does the knee point arrive earlier for Telugu than for English on the same server, because longer token sequences fill the KV cache sooner? That is exactly Kneepoint's instrument, and the first pass needs no new feature: run `kneepoint run` twice against the same llama.cpp or vLLM server, once with the `en` prompts as the corpus and once with `te`, and compare the two knees and $/resolved task. If the gap is real, the feature that earns its place in Kneepoint is small and generic: a *cohort* tag per prompt file so one run reports the knee and $/resolved task per cohort. The same result becomes a Kneepoint Index entry, "the Indic knee", which is the marketing bridge between the two projects. Energy per resolved task (this repo's sampler) is a possible Kneepoint plugin later; it is not on the path to the paper.

## Where to publish

Preprint on arXiv first (cs.CL, cross-list cs.PF or cs.LG). Then a workshop with proceedings: look at the low-resource and multilingual workshops co-located with ACL/EMNLP/COLING (for example LoResLM) and efficient-NLP venues. Deadlines were not checked when this was written; look them up in week 1 and let the nearest sensible one set the pace. First-time arXiv submitters to cs.CL may need an endorsement; sort that out in week 8, not week 10.

## Standing risks

- **Scooped.** Three adjacent papers appeared between May and August 2026. Mitigation: get the measurement out early (a short tech report after week 3 is an option) and keep the fix as the differentiator.
- **Corpus quality.** A reviewer who reads Telugu will check. Native review is not optional.
- **One person's evenings.** The plan has no slack for a second project in the same weeks.
