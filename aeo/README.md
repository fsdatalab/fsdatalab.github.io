# Tracking AI answers about the lab

This folder checks whether AI assistants recommend Quail, DocETL, and the lab
when people ask the questions they ask before choosing a tool or a lab. It
follows the measurement loop from Granola's AEO write-up: write down the
questions, ask them, diagnose the misses, fix pages, and ask again next month.

Jekyll does not publish this folder (see `exclude` in `_config.yml`).

## Files

- `prompts.yaml` lists the questions, tagged by audience (practitioner, student,
  collaborator) and project. Prompts that name us are marked `branded` and are
  left out of the headline numbers.
- `terms.yaml` lists the names and sites to look for. For Quail, the
  alternatives are the AI functions built into data warehouses (Snowflake
  Cortex, Databricks AI Functions, BigQuery AI), plus vLLM and SGLang.
- `track.py` asks each prompt and writes the results.
- `runs/` holds one `<date>.jsonl` (every answer, with its citations) and one
  `<date>-summary.md` per run.

## Running it

You need an Anthropic API key in `ANTHROPIC_API_KEY`, or a Claude Code OAuth
token (from `claude setup-token`) in `CLAUDE_CODE_OAUTH_TOKEN`. The API key wins
if both are set.

```bash
uv run aeo/track.py run
```

That asks every prompt 3 times, because answers change from one ask to the
next. To try a couple of prompts first:

```bash
uv run aeo/track.py run --samples 1 --only quail-cost-10m-rows,lab-cmu-db-hci
```

A full run is 60 answers. Each one uses web search, so a full run should cost
roughly $10 to $15. That is an estimate; check the first run's bill.

To rebuild a summary after you edit `terms.yaml`:

```bash
uv run aeo/track.py report aeo/runs/2026-11-22.jsonl
```

## Reading the summary

- **Headline:** how many non-branded answers named us, and how many cited one
  of our sites, by audience.
- **Every prompt:** the same numbers for each prompt, the change since the last
  run, who the answers named, and which sites they cited.
- **Misses to diagnose:** prompts where no answer named us, with the sources
  the answers used instead. The note "search found one of our pages but the
  answer did not use it" means the page exists and is findable but does not
  answer the question well enough to be cited. That points to fixing the page.
  If our pages never show up in search at all, the fix is coverage elsewhere.
  The note "did not search the web" means the answer came from what the model
  learned in training. Page fixes will not change those quickly; papers, talks,
  and mentions in other people's writing feed the next model's training.

## Limits

- It asks Claude through the API with web search on. ChatGPT, Perplexity, and
  Gemini can give different answers, and the API can differ from the consumer
  apps. Spot-check a few prompts by hand in each app.
- Branded prompts send the lab's and Shreya's names to the API. Pass
  `--skip-branded` to leave them out.
- Mention matching uses the patterns in `terms.yaml`. Read a few answers in the
  `.jsonl` file before trusting a big change in the numbers.
