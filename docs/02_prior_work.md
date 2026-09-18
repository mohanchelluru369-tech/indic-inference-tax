# 02. Prior work and the gap

Surveyed on 17 September 2026. This area moves fast (five of these papers are from the last four months), so **re-run the search before writing the paper and before each milestone**. The summaries below come from abstracts and HTML versions read through a summarising tool, not from a close reading of each PDF. Read the papers properly before citing a number from them.

| Work | What it measured | What it did not |
|---|---|---|
| [The Tokenizer Tax](https://arxiv.org/abs/2607.24276) (Jul 2026) | Token fertility for 14 Indian languages on FLORES-200, 6 tokenizers. Average 8.0x vs English, Malayalam 13.0x, effective context as low as 12%. Root cause: failed BPE merges leaving single-byte tokens (r = 0.89). | No latency, throughput, energy or serving cost. No romanized or code-mixed text. |
| [The Language-Energy Divide](https://arxiv.org/html/2606.21869v1) (Jun 2026) | Energy per language, 122 languages (Belebele), Qwen3 8-32B, Gemma3-27B, Llama-3.1-8B on L40S and RTX 6000 Pro. Per-token energy varies up to 8.3x, energy per response up to 179x. | Datacenter GPUs only, no quantization, no edge or on-device. From India only Gujarati, Kannada, Odia, Malayalam called out. No mitigation tested. **Closest prior work; position against it explicitly.** |
| [The Multilingual Quantization Tax](https://arxiv.org/html/2608.09941) (Aug 2026) | 4-bit NF4 accuracy loss across 8 languages on Gemma 4 and Qwen 3.5 (2B/4B). Hindi collapses below chance on one model. | Hindi is the only Indic language. Accuracy only: no latency, memory or energy. No calibrated methods (AWQ, GPTQ, imatrix). No generative tasks. |
| [Speculative Decoding Across Languages](https://arxiv.org/abs/2605.30580) (May 2026) | Acceptance rate and speedup for 11 low-resource languages on Qwen 3.5. About 1.02x baseline speedup; n-gram drafts reach about 1.30x. | No Indic language (Nepali is the nearest). Suggests language-specific drafts as future work. |
| [MUTANT-Indic tokenizer](https://arxiv.org/html/2511.03237v2) (Krutrim, 2025) | A better Indic tokenizer: 39.5% better fertility than Llama 4, 44% higher inference throughput in a 1B model. Also tests swapping the tokenizer into an existing model via continual pretraining. | Requires (continual) pretraining. Models up to 1B. No edge hardware, no energy. |
| [Indi-RomCoM](https://arxiv.org/html/2606.30790v1) (Jun 2026) | LLM accuracy on romanized code-mixed Hindi, Bengali, Gujarati, Tamil instructions at 25/50/75% mixing. | No Telugu. No token counts, cost or latency. |
| [Inspect India Evals](https://arxiv.org/abs/2607.25375v1), [IndicSafeEval](https://arxiv.org/abs/2609.03781) (2026) | Safety, bias and cultural knowledge in Indian languages. | No efficiency metrics. (Also why safety was not chosen: it is crowded.) |
| [Urja Labs: on-device LLMs on mid-range Android](https://urjalabs.in/blog/on-device-llm-benchmarks-mid-range-android/) | tok/s, TTFT, memory for small models on a Snapdragon 7 Gen 1 phone. | CPU only, plugged in, no energy, no thermal soak, Indian languages not benchmarked. |
| [Sarvam Edge coverage](https://ucstrategies.com/news/indias-best-offline-ai-only-works-on-phones-80-of-indians-cant-afford/) | Reports that Sarvam's on-device model needs Snapdragon 8 Gen 3 class hardware. | "No latency numbers for a ₹12,000 phone." |

## The gap, in one paragraph

Each component of the tax has a paper. None of them is about Indic languages on the hardware Indians own; none covers romanized or code-mixed input on the cost side; none measures the effects together; and the only demonstrated fix needs pretraining. The contribution here is (a) the first end-to-end, paired, systems-level measurement of the Indic inference tax across consumer hardware down to a budget phone, in the forms people actually type, and (b) a measured, laptop-scale mitigation on an existing open model.

## Still to check before claiming novelty

- AI4Bharat, Sarvam, Krutrim, BharatGen and Tech Mahindra (Indus) technical reports for any per-language serving numbers.
- Any follow-up to The Language-Energy Divide covering edge devices.
- MobiBench (arXiv 2609.13159, Sep 2026): on-device LLM benchmark; could not be fetched during the survey. Check whether input language is one of its dimensions.
- Tokenizer-transplant methods to build on rather than reinvent: Zero-Shot Tokenizer Transfer (2405.07883), Model-Aware Tokenizer Transfer (2510.21954), In-Place Tokenizer Expansion (2607.15232), and the 2026 EACL paper on tokenizer-aware cross-lingual adaptation.
