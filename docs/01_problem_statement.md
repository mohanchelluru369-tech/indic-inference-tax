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
2. **Nobody has priced the way Indians actually type.** Most people type Telugu and Hindi in Latin script and mix in English. Romanized text is cheap in tokens but models understand it worse. That is a cost-quality trade-off hundreds of millions of people make implicitly, and it has not been quantified.
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
- **H4** Romanized and code-mixed prompts cost within 1.5x of English in tokens but lose accuracy and script fidelity relative to native script.
- **H5** At Q4 and below, accuracy drops more for te/hi than for en on the same items; an Indic-calibrated importance matrix recovers part of the difference.
- **H6** Speculative decoding acceptance is lower for te/hi than en with a generic draft model.
- **H7** On the budget phone the taxes compound: cost per correct Telugu answer exceeds the product of the separately measured taxes' lower bounds.
- **H8** Tokenizer expansion plus light training recovers at least half of the token tax on a 1-4B model with no significant accuracy loss.

## What would redirect the project (decide after the first measurement)

Newer tokenizers have large multilingual vocabularies (Gemma 3: 262k, OpenAI o200k). If Exp 01 shows Hindi and Telugu already within about 1.5x of English on current-generation tokenizers, then the tokenizer is no longer the main story for new models, and the weight of the paper moves to quantization, speculative decoding, the romanization trade-off and the on-device compound effect (RQ3-RQ5), with the tokenizer fix aimed at the large installed base of Llama-3-class models. That is still a paper. It is a different paper, and the first run tells us which one we are writing.

## Non-goals

Training a new foundation model. Building a new benchmark of cultural knowledge. Speech. Anything that needs data collection from human annotators at scale. These are real problems; they are bottlenecked on money and people, not on the skill this project brings.
