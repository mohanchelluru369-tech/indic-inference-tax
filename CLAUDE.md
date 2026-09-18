# Project memory for AI assistants

Research project: measure and then reduce the extra inference cost Indian languages pay vs English on commodity hardware (Mac, consumer GPU, cloud T4, budget Android phone). Goal is an arXiv preprint plus this open-source harness. Start with `README.md`, then `docs/01_problem_statement.md` and `docs/03_methodology.md`. Plan and dates: `docs/04_roadmap.md`.

## Commands

```bash
.venv/bin/pytest                       # must stay green
.venv/bin/indictax doctor              # machine readiness
.venv/bin/indictax fertility           # Exp 01
./scripts/first_measurement.sh         # Exp 01 + Exp 02 with a managed llama-server
.venv/bin/indictax report results/<run>
```

## Conventions

- Every comparison is paired by corpus item and reported as a ratio to `en` with a bootstrap interval over items. Do not add unpaired averages to reports.
- Never fabricate or fill in a measurement. Missing energy is `null`; a tokenizer that fails to load is listed as skipped; a failed request is a row with `ok: false`.
- Anything that can bias a number (caching, ordering, thermal state, power state, thinking tokens) belongs in `docs/03_methodology.md` with the control for it, and ideally a test.
- Results folders are immutable once written. Bad run: delete the folder, do not edit it.
- Seed corpus text was LLM-written and needs native review (`data/README.md`). Do not mark items `native-ok` yourself.
- Repo ids marked `verified: false` and the items listed under "not verified on real hardware" in `docs/05_hardware_matrix.md` were written without access to Hugging Face or Apple Silicon. Confirm, then update the note.
- New experiment = new YAML in `configs/` + a section in the methodology, not a flag buried in code.
- Python 3.10+, stdlib + httpx, pyyaml, tokenizers, regex, numpy. Keep dependencies few: this has to install on a phone under Termux.
