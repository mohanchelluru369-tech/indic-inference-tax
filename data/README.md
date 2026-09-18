# Data

## prompts/seed_v0.jsonl

18 parallel prompts: 16 short questions in 7 variants and 2 long-context items in 3 variants, across agriculture, government services, finance, health information, education, travel, daily life and technology.

**Provenance and status.** Written by an LLM (Claude) on 17 Sep 2026. Every item is marked `"review": "pending"`. Nothing here has been checked by a native speaker yet. Use it to shake down the pipeline and get a first signal. Do not publish results that rest on it until it has been reviewed.

**How to review (Telugu first).** For each item, read `en`, then check:

1. `te` / `hi`: is it correct, natural, and the *same content* as the English: nothing added, nothing dropped? Prefer what a person would really write over textbook translation.
2. `te_rom` / `hi_rom`: is this how people actually type it in Latin script? Spelling varies between people; pick the most common form and be consistent within an item.
3. `te_cm` / `hi_cm`: is the English mixing natural (the words people really switch for), not forced?
4. When satisfied, change `"review": "pending"` to `"review": "native-ok"` and note your initials in git.

Keep the content parallel. If you improve one variant in a way that changes meaning, change all of them. `pytest` checks that each variant is in the script it claims and that the variant sets are complete.

## Variants

| key | language | written as | answer expected in |
|---|---|---|---|
| en | English | Latin | Latin |
| hi / te | Hindi / Telugu | native script | Devanagari / Telugu |
| hi_rom / te_rom | Hindi / Telugu | romanized | Latin |
| hi_cm / te_cm | Hindi-English / Telugu-English | code-mixed, Latin | Latin |

Long-context items exist only in `en`, `hi`, `te`: documents are written in native script; it is the user's *question* that gets romanized.

## Adding corpora

One JSON object per line: `id`, `domain`, `kind`, `variants` (must include `en`), optional `gold` (A-D for multiple choice), `review`. `indictax make-belebele` writes this format.

## License

CC BY 4.0 for `seed_v0.jsonl`. Belebele and any other downloaded corpus keep their own licenses and are not committed to the repo.
