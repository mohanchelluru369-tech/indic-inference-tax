# Tokenizer tax

corpus: `data/prompts/seed_v0.jsonl` (18 items)

## Tokenizer tax vs English (same content; 1.00 = parity)

| tokenizer | vocab | en | hi | te | hi_rom | te_rom | hi_cm | te_cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unsloth/Llama-3.2-3B-Instruct | 128,256 | 1.00 | 2.44 | 7.59 | 1.87 | 1.68 | 1.21 | 1.09 |
| unsloth/gemma-3-4b-it | 262,145 | 1.00 | 1.18 | 1.58 | 1.59 | 1.57 | 1.04 | 1.06 |
| Qwen/Qwen3-4B | 151,669 | 1.00 | 4.30 | 6.48 | 1.87 | 1.68 | 1.22 | 1.10 |
| microsoft/Phi-4-mini-instruct | 200,029 | 1.00 | 1.45 | 1.75 | 1.71 | 1.57 | 1.16 | 1.05 |
| HuggingFaceTB/SmolLM3-3B | 128,256 | 1.00 | 2.44 | 7.59 | 1.87 | 1.68 | 1.21 | 1.09 |
| openai/gpt-oss-20b | 200,019 | 1.00 | 1.45 | 1.75 | 1.71 | 1.57 | 1.16 | 1.05 |
| sarvamai/sarvam-1 | 68,096 | 1.00 | 1.14 | 1.19 | 2.18 | 1.91 | 1.41 | 1.22 |
| krutrim-ai-labs/Krutrim-2-instruct | 131,072 | 1.00 | 1.75 | 2.05 | 1.82 | 1.62 | 1.19 | 1.09 |

## Fragment rate (tokens that are lone bytes / partial characters)

| tokenizer | en | hi | te | hi_rom | te_rom | hi_cm | te_cm |
|---|---:|---:|---:|---:|---:|---:|---:|
| unsloth/Llama-3.2-3B-Instruct | 0.0% | 0.5% | 98.2% | 0.0% | 0.0% | 0.0% | 0.0% |
| unsloth/gemma-3-4b-it | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Qwen/Qwen3-4B | 0.0% | 52.1% | 81.5% | 0.0% | 0.0% | 0.0% | 0.0% |
| microsoft/Phi-4-mini-instruct | 0.0% | 0.5% | 2.4% | 0.0% | 0.0% | 0.0% | 0.0% |
| HuggingFaceTB/SmolLM3-3B | 0.0% | 0.5% | 98.2% | 0.0% | 0.0% | 0.0% | 0.0% |
| openai/gpt-oss-20b | 0.0% | 0.5% | 2.4% | 0.0% | 0.0% | 0.0% | 0.0% |
| sarvamai/sarvam-1 | 0.0% | 0.0% | 6.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| krutrim-ai-labs/Krutrim-2-instruct | 0.0% | 1.1% | 4.3% | 0.0% | 0.0% | 0.0% | 0.0% |
