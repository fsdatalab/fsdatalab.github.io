# /// script
# requires-python = ">=3.10"
# dependencies = ["anthropic>=1.1", "pyyaml>=6"]
# ///
"""Ask an AI assistant the questions in prompts.yaml and record who it recommends.

For every prompt, the tracker asks Claude (with web search on) a few times and
records, for each answer:
  - which of our projects it names (terms.yaml: ours)
  - which alternatives it names (terms.yaml: competitors)
  - which URLs it cites, and whether any are ours
  - which URLs search found but the answer did not cite

Usage (from the repo root):
  uv run aeo/track.py run                  # all prompts, 3 samples each
  uv run aeo/track.py run --samples 1 --only quail-cost-10m-rows,lab-cmu-db-hci
  uv run aeo/track.py run --skip-branded
  uv run aeo/track.py report aeo/runs/2026-11-22.jsonl   # rebuild a summary

Each run writes aeo/runs/<date>.jsonl (one line per answer) and
aeo/runs/<date>-summary.md, which compares against the previous run.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import yaml

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
MODEL = "claude-opus-5-5"
MAX_PAUSE_RESUMES = 4
OAUTH = False  # set when authenticating with CLAUDE_CODE_OAUTH_TOKEN


# ---------------------------------------------------------------- config


def load_yaml(name: str) -> dict:
    return yaml.safe_load((HERE / name).read_text())


def compile_entities(section: dict) -> dict[str, list[re.Pattern]]:
    out = {}
    for name, spec in (section or {}).items():
        flags = 0 if spec.get("case_sensitive") else re.IGNORECASE
        out[name] = [re.compile(p, flags) for p in spec["patterns"]]
    return out


def find_entities(text: str, entities: dict[str, list[re.Pattern]]) -> list[str]:
    return [name for name, pats in entities.items() if any(p.search(text) for p in pats)]


def source_key(url: str) -> str:
    """Group a URL by site. GitHub and Reddit are grouped one level deeper."""
    u = urlparse(url)
    host = u.netloc.lower().removeprefix("www.")
    parts = [p for p in u.path.split("/") if p]
    if host == "github.com" and parts:
        return f"github.com/{parts[0].lower()}"
    if host.endswith("reddit.com") and len(parts) >= 2 and parts[0] == "r":
        return f"reddit.com/r/{parts[1].lower()}"
    return host


def is_ours(url: str, our_sources: list[str]) -> bool:
    u = urlparse(url)
    host = u.netloc.lower().removeprefix("www.")
    full = host + u.path.lower()
    for src in our_sources:
        src = src.lower()
        if full.startswith(src):
            return True
        if "/" not in src and host.endswith("." + src):
            return True
    return False


# ---------------------------------------------------------------- asking


def ask(client, prompt: str) -> dict:
    """One answer from Claude with web search, resuming paused turns."""
    messages = [{"role": "user", "content": prompt}]
    content: list[dict] = []
    resp = None
    for _ in range(MAX_PAUSE_RESUMES + 1):
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            output_config={"effort": "medium"},
            # The basic search tool, on purpose. The newer web_search_20260209 runs
            # searches from code, which returns no citations and, in testing, used up
            # its search limit mid-batch and then answered from memory.
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 10}],
            # If a safety classifier declines, retry on a fallback model chosen
            # by the server instead of recording an empty answer.
            betas=["server-side-fallback-2026-07-01"] + (["oauth-2025-04-20"] if OAUTH else []),
            fallbacks="default",
            messages=messages,
        )
        content.extend(b.model_dump() for b in resp.content)
        if resp.stop_reason != "pause_turn":
            break
        messages = messages + [{"role": "assistant", "content": resp.content}]

    text_parts, cited, searched = [], [], []
    for block in content:
        if block.get("type") == "text":
            text_parts.append(block.get("text") or "")
            for c in block.get("citations") or []:
                if c.get("url"):
                    cited.append({"url": c["url"], "title": c.get("title")})
        elif block.get("type") == "web_search_tool_result":
            results = block.get("content")
            if isinstance(results, list):  # an error comes back as a dict
                for r in results:
                    if r.get("url"):
                        searched.append({"url": r["url"], "title": r.get("title")})

    return {
        "model": resp.model if resp else MODEL,
        "stop_reason": resp.stop_reason if resp else None,
        "answer": "".join(text_parts),
        "cited": cited,
        "searched": searched,
    }


def analyze(rec: dict, ours, competitors, our_sources) -> dict:
    text = rec.get("answer") or ""
    cited_urls = list(dict.fromkeys(c["url"] for c in rec.get("cited", [])))
    searched_urls = list(dict.fromkeys(s["url"] for s in rec.get("searched", [])))
    rec["mentions_ours"] = find_entities(text, ours)
    rec["mentions_competitors"] = find_entities(text, competitors)
    rec["cited_ours"] = [u for u in cited_urls if is_ours(u, our_sources)]
    rec["searched_ours_not_cited"] = [
        u for u in searched_urls if is_ours(u, our_sources) and u not in cited_urls
    ]
    rec["cited_sources"] = sorted({source_key(u) for u in cited_urls})
    return rec


def cmd_run(args) -> None:
    import anthropic

    prompts = load_yaml("prompts.yaml")["prompts"]
    terms = load_yaml("terms.yaml")
    ours = compile_entities(terms["ours"])
    competitors = compile_entities(terms.get("competitors"))
    our_sources = terms.get("our_sources", [])

    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - {p["id"] for p in prompts}
        if unknown:
            sys.exit(f"Unknown prompt ids: {', '.join(sorted(unknown))}")
        prompts = [p for p in prompts if p["id"] in wanted]
    if args.skip_branded:
        prompts = [p for p in prompts if not p.get("branded")]
    if args.audience:
        prompts = [p for p in prompts if p["audience"] == args.audience]

    date = dt.date.today().isoformat()
    out_path = RUNS / f"{date}.jsonl"
    existing: list[dict] = []
    if out_path.exists():
        if args.append:
            existing = read_jsonl(out_path)
            done = {r["id"] for r in existing}
            prompts = [p for p in prompts if p["id"] not in done]
            if not prompts:
                sys.exit(f"Every selected prompt is already in {out_path}.")
        elif not args.overwrite:
            sys.exit(f"{out_path} already exists. Pass --append to add new prompts or --overwrite to replace it.")

    jobs = [(p, i) for p in prompts for i in range(args.samples)]
    print(f"Asking {len(prompts)} prompts x {args.samples} samples = {len(jobs)} answers with {MODEL}")

    global OAUTH
    oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if not os.environ.get("ANTHROPIC_API_KEY") and oauth_token:
        # Claude Code OAuth tokens go in the Authorization header and need the oauth beta.
        OAUTH = True
        client = anthropic.Anthropic(auth_token=oauth_token, max_retries=5)
    else:
        client = anthropic.Anthropic(max_retries=5)
    RUNS.mkdir(exist_ok=True)
    records = list(existing)

    def work(p, i):
        base = {k: p.get(k) for k in ("id", "audience", "project", "text")}
        base.update(branded=bool(p.get("branded")), sample=i, date=date)
        try:
            base.update(ask(client, p["text"]))
        except anthropic.APIStatusError as e:
            base.update(error=f"{e.status_code} {e.message}", answer="", cited=[], searched=[])
        except anthropic.APIConnectionError as e:
            base.update(error=f"connection: {e}", answer="", cited=[], searched=[])
        return analyze(base, ours, competitors, our_sources)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, p, i) for p, i in jobs]
        for n, f in enumerate(as_completed(futures), 1):
            rec = f.result()
            records.append(rec)
            flag = "ERROR" if rec.get("error") else ", ".join(rec["mentions_ours"]) or "-"
            print(f"[{n}/{len(jobs)}] {rec['id']} #{rec['sample']}: {flag}")

    records.sort(key=lambda r: (r["id"], r["sample"]))
    with out_path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    summary = write_summary(out_path, records)
    print(f"\nWrote {out_path}\nWrote {summary}")


# ---------------------------------------------------------------- report


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def previous_run(path: Path) -> Path | None:
    runs = sorted(p for p in path.parent.glob("*.jsonl") if p.name < path.name)
    return runs[-1] if runs else None


def per_prompt(records: list[dict]) -> dict[str, dict]:
    out = {}
    for pid, rs in group(records, "id").items():
        ok = [r for r in rs if not r.get("error")]
        n = len(ok)
        comp = collections.Counter(c for r in ok for c in r["mentions_competitors"])
        srcs = collections.Counter(s for r in ok for s in r["cited_sources"])
        ours = collections.Counter(o for r in ok for o in r["mentions_ours"])
        out[pid] = {
            "audience": rs[0]["audience"],
            "project": rs[0]["project"],
            "branded": rs[0].get("branded", False),
            "n": n,
            "errors": len(rs) - n,
            "mentioned": sum(1 for r in ok if r["mentions_ours"]),
            "cited": sum(1 for r in ok if r["cited_ours"]),
            "found_not_cited": sum(1 for r in ok if r["searched_ours_not_cited"] and not r["cited_ours"]),
            "no_search": sum(1 for r in ok if not r.get("searched")),
            "ours": ours,
            "competitors": comp,
            "sources": srcs,
        }
    return out


def group(records, key):
    g = collections.defaultdict(list)
    for r in records:
        g[r[key]].append(r)
    return g


def rate(k: int, n: int) -> str:
    return f"{k}/{n}" if n else "-"


def top(counter: collections.Counter, k: int = 3) -> str:
    return ", ".join(f"{name} ({c})" for name, c in counter.most_common(k)) or "-"


def write_summary(path: Path, records: list[dict]) -> Path:
    stats = per_prompt(records)
    prev_path = previous_run(path)
    prev = per_prompt(read_jsonl(prev_path)) if prev_path else {}

    lines = [f"# AI answer tracking: {path.stem}", ""]
    models = sorted({r.get("model") for r in records if r.get("model")})
    lines.append(f"Model: {', '.join(models) or MODEL}. Answers: {len(records)}.")
    if prev_path:
        lines.append(f"Compared with {prev_path.stem}.")
    lines.append("")

    # Headline numbers, non-branded only, so they are not inflated by prompts that name us.
    nb = [s for s in stats.values() if not s["branded"]]
    lines += ["## Headline (non-branded prompts)", ""]
    lines += ["| Audience | Prompts | Answers naming us | Answers citing our sites |", "|---|---|---|---|"]
    for aud in ("practitioner", "student", "collaborator"):
        ss = [s for s in nb if s["audience"] == aud]
        if not ss:
            continue
        n = sum(s["n"] for s in ss)
        lines.append(
            f"| {aud} | {len(ss)} | {rate(sum(s['mentioned'] for s in ss), n)} | {rate(sum(s['cited'] for s in ss), n)} |"
        )
    lines.append("")

    lines += ["## Every prompt", ""]
    lines += [
        "| Prompt | Named us | Cited us | Change in named | Who it named | Top cited sources |",
        "|---|---|---|---|---|---|",
    ]
    for pid in sorted(stats, key=lambda p: (stats[p]["branded"], stats[p]["audience"], p)):
        s = stats[pid]
        change = ""
        if pid in prev and prev[pid]["n"] and s["n"]:
            delta = s["mentioned"] / s["n"] - prev[pid]["mentioned"] / prev[pid]["n"]
            change = f"{delta:+.0%}" if delta else "same"
        label = pid + (" (branded)" if s["branded"] else "")
        err = f" ({s['errors']} errors)" if s["errors"] else ""
        lines.append(
            f"| {label}{err} | {rate(s['mentioned'], s['n'])} | {rate(s['cited'], s['n'])} | {change} "
            f"| {top(s['ours'] + s['competitors'], 4)} | {top(s['sources'])} |"
        )
    lines.append("")

    misses = [(pid, s) for pid, s in stats.items() if not s["branded"] and s["n"] and s["mentioned"] == 0]
    lines += ["## Misses to diagnose", ""]
    if misses:
        lines.append("Non-branded prompts where no answer named us. The cited sources are where the answers came from.")
        lines.append("")
        for pid, s in sorted(misses):
            note = " Search found one of our pages but the answer did not use it." if s["found_not_cited"] else ""
            if s["no_search"]:
                note += (
                    f" {s['no_search']} of {s['n']} answers did not search the web, so they reflect"
                    " what the model learned in training, not current pages."
                )
            lines.append(f"- **{pid}**: named {top(s['competitors'], 4)}. Cited {top(s['sources'], 5)}.{note}")
    else:
        lines.append("None.")
    lines.append("")

    out = path.with_name(f"{path.stem}-summary.md")
    out.write_text("\n".join(lines))
    return out


def cmd_report(args) -> None:
    path = Path(args.path).resolve()
    print(f"Wrote {write_summary(path, read_jsonl(path))}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="ask every prompt and write a new run")
    r.add_argument("--samples", type=int, default=3, help="answers per prompt (answers vary, so use more than 1)")
    r.add_argument("--only", help="comma-separated prompt ids")
    r.add_argument("--audience", choices=["practitioner", "student", "collaborator"])
    r.add_argument("--skip-branded", action="store_true", help="leave out prompts that name us")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--overwrite", action="store_true", help="replace today's run if it exists")
    r.add_argument("--append", action="store_true", help="add prompts missing from today's run, keep the rest")
    r.set_defaults(func=cmd_run)
    p = sub.add_parser("report", help="rebuild the summary for an existing run")
    p.add_argument("path")
    p.set_defaults(func=cmd_report)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
