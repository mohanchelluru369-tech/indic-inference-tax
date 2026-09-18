# 01. Problem statement

## The claim

A Telugu or Hindi speaker pays more than an English speaker for the same AI answer: more tokens, so more prefill compute, more KV-cache memory, more decode steps, more latency, more energy, and a smaller effective context window. On a datacenter GPU that is a line on an invoice. On a ₹12,000 phone it is the difference between usable and unusable, and that phone is where most of India will meet AI. Sarvam's on-device speech model needs a Snapdragon 8 Gen 3 (phones from about ₹40,000); most phones sold in India cost under ₹15,000.

The individual pieces of this tax have each been measured once, separately, mostly not for Indic languages and never on commodity hardware (details in [02_prior_work.md](02_prior_work.md)):

- token counts: yes (14 Indian languages), but no latency, energy or cost
- energy per language: yes, on L40S / RTX 6000 datacenter GPUs, unquantized, only Gujarati, Kannada, Odia and Malayalam from India
- quantization damage per language: yes, accuracy only, Hindi the only Indic language
- speculative decoding per language: yes, no Indic language at all
- romanized / code-mixed input: accuracy only, no tokens, cost or latency, and no Telugu

## What is unsolved

1. **Nobody knows the compound, end-to-end tax for Indic languages on hardware people own.** The effects multiply (more tokens x quantization loss x weaker speculative decoding x tighter memory), and they have only been studied one at a time.
2. **Nobody has priced the way Indians actually type.** Most people type Telugu and Hindi in Latin script and mix in English. Romanized text was assumed here to be cheap in tokens and worse understood — a straight cost-quality trade-off that hundreds of millions of people make implicitly. Experiment 01 shows the cost half is wrong on the models that matter most for these languages: romanizing is *dearer* than native script on every tokenizer with real Indic coverage (see the week 1 checkpoint below). So the trade-off is worse than assumed, and still unquantified.
3. **The known fix requires pretraining.** Krutrim's MUTANT-Indic tokenizer shows a 44% throughput gain, by pretraining with a better tokenizer. What fraction of that is recoverable on an *existing* open 1-4B model, with a laptop's worth of compute, is open. This is the hard part of the project and the part that would matter most.

## Research questions

- **RQ1** For the same content, what is the multiplier vs English on input tokens, output tokens, time to first token, time per answer and energy per answer, for Hindi and Telugu in native, romanized and code-mixed form, across tokenizer families and across an M-series Mac, a consumer NVIDIA GPU, a cloud T4 and a budget Android phone?
- **RQ2** How much of that multiplier is explained by token count alone? (If decode tokens/s is language-independent, almost all of it should be. Anything left over is a finding.)
- **RQ3** What does the tax do to quality under real constraints: answers cut off at a fixed token cap, answers that drift into English or the wrong script, accuracy per joule?
- **RQ4** Does 4-bit and lower quantization cost Hindi and Telugu more accuracy than English at the same bit width, and does calibrating the quantizer on Indic text recover it?
- **RQ5** Is the speculative-decoding speedup lower for Hindi and Telugu, and do language-matched or n-gram drafts recover it?
- **RQ6 (the fix)** On an existing open 1-4B model, how much of the end-to-end tax can be closed with tokenizer expansion plus light continued training, Indic-calibrated quantization and language-aware drafting, without losing accuracy on Belebele / MILU?

## Hypotheses (written down before measuring, so they can be wrong)

- **H1** Decode tokens/s is the same across languages within about 5%. Time and energy per answer scale with the token tax.
- **H2** The TTFT tax approaches the input-token tax for long contexts and is much smaller for short prompts, where fixed overhead dominates.
- **H3** Under a fixed `max_tokens`, native-script answers are cut off several times more often than English ones.
- **H4** Romanized and code-mixed prompts cost within 1.5x of English in tokens but lose accuracy and script fidelity relative to native script. **Token half refuted for romanized (2026-09-18).** Code-mixed holds at 1.04-1.41x, but romanized costs 1.57-2.18x on every tokenizer measured, and on the tokenizers that cover the script it is *dearer than native script*, not cheaper. The accuracy and script-fidelity half is untested.
- **H5** At Q4 and below, accuracy drops more for te/hi than for en on the same items; an Indic-calibrated importance matrix recovers part of the difference.
- **H6** Speculative decoding acceptance is lower for te/hi than en with a generic draft model.
- **H7** On the budget phone the taxes compound: cost per correct Telugu answer exceeds the product of the separately measured taxes' lower bounds.
- **H8** Tokenizer expansion plus light training recovers at least half of the token tax on a 1-4B model with no significant accuracy loss.

## Week 1 checkpoint: what the first measurement decided

The rule written here before any data was: *if current tokenizers already put hi/te within about 1.5x of English, the tokenizer stops being the main story.* [Experiment 01](../results/20260918-095750-fertility/fertility.md) triggered it. The decision, taken 2026-09-18:

**The tokenizer tax is not one number per language. It is a per-model number, and the spread across models dwarfs the spread across languages.** Telugu costs 7.59x English on Llama 3.2 and 1.19x on Sarvam-1 — same sentences, same measurement, a 6x difference decided by nothing but which model is being served. Reporting "the Indic tokenizer tax" as a per-language average, which is what the literature currently does, averages over the only variable a practitioner can actually act on.

Three consequences for the plan:

1. **The measurement paper keeps its full scope.** It gains a finding it did not expect, which is the one above.
2. **The mitigation work (Exp 05) narrows.** Closing a 1.19x gap on Sarvam-1 is not worth a paper; closing a 7.59x gap on Llama-3-class models — the most widely deployed open weights there are — is. Tokenizer expansion targets the byte-fallback families, where the mechanism is visible and the headroom is real.
3. **Weight shifts to what is still unmeasured:** quantization x language (Exp 03), speculative decoding x language (Exp 04), the on-device compound effect, and the romanization trade-off, which the same run made the most novel thing in the set.

**The romanization trade-off is now a first-class result, not a side observation.** Most people type Hindi and Telugu in Latin script. On Sarvam-1 that costs 1.91x English against 1.19x for native script; on Gemma 3 it is a wash (1.57 vs 1.58); only on the byte-fallback tokenizers is it the large saving it is assumed to be (1.68 vs 7.59 on Llama 3.2). So the habit hundreds of millions of people have — and that every romanized-input product is built around — is a *penalty* on exactly the models built for them. Nobody has priced it. Whether it also costs accuracy is [Indi-RomCoM](02_prior_work.md)'s question, and joining the two is ours.

**The natural experiment this hands us.** Llama 3.2 3B and Gemma 3 4B run on the same machine, through the same engine, from the same quantization, and differ by 4.8x in Telugu token cost. Exp 02 already has a config for each (`configs/exp02_llama32_3b.yaml`, `configs/exp02_gemma3_4b.yaml`). Running the pair back to back isolates tokenizer coverage as the cause of a latency and energy difference about as cleanly as it can be isolated on real hardware, and it directly tests H1.

### What is still open at this checkpoint

Everything downstream of tokens. Exp 01 counts tokens; it says nothing about whether the 7.59x becomes 7.59x of time and joules on a real machine (H1, H2), nothing about truncation or script drift (H3), nothing about quantization or drafting (H5, H6), and nothing about accuracy. The redirect changes emphasis, not the method.

### Health warning on the numbers above

Eighteen prompts, written by an LLM, **not yet reviewed by native speakers**. The direction is not in doubt — a 6x spread and a 98% fragment rate do not come from translation wobble — but no third digit here should be quoted, and none of it belongs in a paper until the corpus is reviewed and the same measurement has been repeated on FLORES+/IN22.

## Non-goals

Training a new foundation model. Building a new benchmark of cultural knowledge. Speech. Anything that needs data collection from human annotators at scale. These are real problems; they are bottlenecked on money and people, not on the skill this project brings.
