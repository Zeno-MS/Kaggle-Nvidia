"""
Generate high-quality cipher CoT traces via GPT-4.1 (Azure OpenAI).

For each of the 1,576 cipher problems:
  - Call GPT-4.1 to build the cipher map and decrypt the query
  - Verify the answer against ground truth
  - Save CoT if correct
  - FAIL LOUDLY if wrong or API error — no silent fallback to template

Output files:
  data/cipher_cot_gpt41.jsonl   — successful CoTs (id → cot, answer)
  data/cipher_cot_failures.jsonl — problems where GPT-4.1 was wrong or errored

After generation, rewrites data/train_formatted.jsonl and data/val_formatted.jsonl
with GPT-4.1 CoTs replacing template CoTs for cipher problems.

Usage:
    python3 scripts/generate_cipher_cot.py             # full run
    python3 scripts/generate_cipher_cot.py --dry-run 5 # test on 5 problems
    python3 scripts/generate_cipher_cot.py --resume     # skip already-completed ids
    python3 scripts/generate_cipher_cot.py --merge-only # skip generation, just merge

Requires:
    pip install openai
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT in .env
    Or set them in the CONFIG block below.
"""

import json
import os
import re
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# ── Load .env if available ─────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent.parent / "Cicero" / ".env")
except ImportError:
    pass  # dotenv optional; set env vars manually if needed

# ── Config ─────────────────────────────────────────────────────────
AZURE_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "https://cicero-debug.openai.azure.com/")
AZURE_API_KEY    = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-41-debug")
AZURE_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

PROJECT_ROOT     = Path(__file__).parent.parent
DATA_DIR         = PROJECT_ROOT / "data"
TRAIN_CSV        = DATA_DIR / "train.csv"
TRAIN_JSONL      = DATA_DIR / "train_formatted.jsonl"
VAL_JSONL        = DATA_DIR / "val_formatted.jsonl"
COT_OUTPUT       = DATA_DIR / "cipher_cot_gpt41.jsonl"
FAILURES_OUTPUT  = DATA_DIR / "cipher_cot_failures.jsonl"

MAX_WORKERS = 3       # concurrent GPT-4.1 requests
MAX_RETRIES = 3       # retries on rate limit / transient error
RETRY_BASE  = 2.0     # exponential backoff base (seconds)

# ── Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────
SYSTEM_PROMPT = """You solve substitution cipher puzzles. Each puzzle gives you example ciphertext→plaintext word pairs, then a query to decrypt.

Your task:
1. Build a character substitution mapping from the example pairs. Show each mapping explicitly (e.g., "t→u, g→r, ...")
2. List ALL discovered mappings in a compact table
3. For characters in the query NOT found in your mapping table: infer the most likely plaintext character from English word patterns, letter frequency, and context. Show your reasoning.
4. Decode the query character by character, showing your work
5. On the VERY LAST LINE of your response, output exactly this format (nothing after it):
   DECODED: <your final decrypted text>

Start your response immediately with "Building the cipher mapping from the examples."
Do not include any preamble, introduction, or closing remarks."""


def load_cipher_problems() -> list[dict]:
    """Load cipher problems from train.csv."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from pipeline.utils import load_problems
    problems = load_problems(str(TRAIN_CSV))
    cipher = [p for p in problems if p.category == "cipher"]
    logger.info(f"Loaded {len(cipher)} cipher problems")
    return cipher


def load_existing_cots() -> dict[str, dict]:
    """Load already-generated CoTs for resume support."""
    if not COT_OUTPUT.exists():
        return {}
    cots = {}
    with open(COT_OUTPUT) as f:
        for line in f:
            entry = json.loads(line)
            cots[entry["id"]] = entry
    logger.info(f"Resuming: {len(cots)} cipher CoTs already generated")
    return cots


def call_gpt41(problem, client) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Call GPT-4.1 to solve a cipher problem.

    Returns:
        (cot_trace, decoded_answer, error_message)
        On success: (cot, answer, None)
        On failure: (None, None, error_message)
    """
    from openai import AzureOpenAI, RateLimitError, APIError

    user_message = problem.prompt

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=AZURE_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.0,   # deterministic for training data
                max_tokens=1024,
            )
            raw = response.choices[0].message.content.strip()

            # Extract answer from last "DECODED: ..." line
            match = re.search(r'^DECODED:\s*(.+)$', raw, re.MULTILINE | re.IGNORECASE)
            if not match:
                return None, None, f"No DECODED: line in response. Raw (last 200 chars): ...{raw[-200:]}"

            answer = match.group(1).strip()

            # Strip the DECODED line from the CoT trace
            cot = raw[:match.start()].rstrip()

            return cot, answer, None

        except RateLimitError as e:
            wait = RETRY_BASE ** attempt
            logger.warning(f"[{problem.id}] Rate limit hit (attempt {attempt}/{MAX_RETRIES}). "
                           f"Waiting {wait:.0f}s before retry...")
            time.sleep(wait)

        except APIError as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_BASE ** attempt
                logger.warning(f"[{problem.id}] API error (attempt {attempt}/{MAX_RETRIES}): {e}. "
                               f"Retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                return None, None, f"API error after {MAX_RETRIES} attempts: {e}"

    return None, None, f"Exhausted {MAX_RETRIES} retries"


def process_problem(problem, client) -> dict:
    """
    Process one cipher problem. Returns a result dict with status.
    """
    from pipeline.utils import verify

    cot, gpt_answer, api_error = call_gpt41(problem, client)

    if api_error:
        return {
            "id": problem.id,
            "status": "api_error",
            "error": api_error,
            "ground_truth": problem.answer,
        }

    correct = verify(problem.answer, gpt_answer)

    if not correct:
        return {
            "id": problem.id,
            "status": "wrong_answer",
            "gpt41_answer": gpt_answer,
            "ground_truth": problem.answer,
            "cot": cot,
        }

    return {
        "id": problem.id,
        "status": "ok",
        "cot": cot,
        "gpt41_answer": gpt_answer,
        "ground_truth": problem.answer,
    }


def generate(problems: list, dry_run: int = 0, resume: bool = False):
    """Run GPT-4.1 on all cipher problems. Returns (successes, failures)."""
    from openai import AzureOpenAI

    if not AZURE_API_KEY:
        logger.error("AZURE_OPENAI_API_KEY is not set. Cannot call GPT-4.1.")
        sys.exit(1)

    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VER,
    )

    existing = load_existing_cots() if resume else {}

    target = problems[:dry_run] if dry_run else problems
    pending = [p for p in target if p.id not in existing]

    logger.info(f"Problems to process: {len(pending)} "
                f"({'dry run, ' if dry_run else ''}{len(existing)} already done)")

    successes = dict(existing)  # carry forward resumed results
    failures  = []

    cot_file      = open(COT_OUTPUT,      "a" if resume else "w")
    failure_file  = open(FAILURES_OUTPUT, "a" if resume else "w")

    completed = 0
    total = len(pending)

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_problem, p, client): p for p in pending}

            for future in as_completed(futures):
                problem = futures[future]
                completed += 1

                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "id": problem.id,
                        "status": "exception",
                        "error": str(e),
                        "ground_truth": problem.answer,
                    }

                pct = 100 * completed / total
                status = result["status"]

                if status == "ok":
                    successes[result["id"]] = result
                    cot_file.write(json.dumps(result) + "\n")
                    cot_file.flush()
                    logger.info(f"[{completed}/{total} {pct:.0f}%] ✓ {problem.id}")
                else:
                    failures.append(result)
                    failure_file.write(json.dumps(result) + "\n")
                    failure_file.flush()

                    if status == "wrong_answer":
                        logger.error(
                            f"[{completed}/{total} {pct:.0f}%] ✗ WRONG ANSWER — {problem.id}\n"
                            f"  GPT-4.1:     {result.get('gpt41_answer', '?')}\n"
                            f"  Ground truth: {result.get('ground_truth', '?')}"
                        )
                    elif status in ("api_error", "exception"):
                        logger.error(
                            f"[{completed}/{total} {pct:.0f}%] ✗ ERROR — {problem.id}: {result.get('error', '?')}"
                        )
    finally:
        cot_file.close()
        failure_file.close()

    return successes, failures


def merge(successes: dict):
    """
    Rewrite train_formatted.jsonl and val_formatted.jsonl, replacing
    cipher problem CoTs with GPT-4.1 versions where available.
    """
    if not successes:
        logger.warning("No successful CoTs to merge.")
        return

    replaced = {"train": 0, "val": 0}
    unchanged = {"train": 0, "val": 0}

    for split, path in [("train", TRAIN_JSONL), ("val", VAL_JSONL)]:
        if not path.exists():
            logger.warning(f"{path} not found — skipping {split} merge")
            continue

        updated = []
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                if ex.get("category") == "cipher" and ex["id"] in successes:
                    s = successes[ex["id"]]
                    # Replace response with GPT-4.1 CoT + ground truth answer
                    ex["response"] = (
                        f"<think>\n{s['cot']}\n</think>\n\n"
                        f"\\boxed{{{ex['ground_truth']}}}"
                    )
                    ex["solved_answer"] = s["gpt41_answer"]
                    replaced[split] += 1
                else:
                    if ex.get("category") == "cipher":
                        unchanged[split] += 1
                updated.append(ex)

        with open(path, "w") as f:
            for ex in updated:
                f.write(json.dumps(ex) + "\n")

        logger.info(
            f"{split}: replaced {replaced[split]} cipher CoTs, "
            f"{unchanged[split]} cipher problems kept template (no GPT-4.1 CoT available)"
        )


def print_report(problems: list, successes: dict, failures: list):
    total_cipher = len(problems)
    n_ok = len(successes)
    n_fail = len(failures)
    wrong = [f for f in failures if f["status"] == "wrong_answer"]
    errors = [f for f in failures if f["status"] in ("api_error", "exception")]

    print("\n" + "="*60)
    print("  GPT-4.1 Cipher CoT Generation Report")
    print("="*60)
    print(f"  Total cipher problems : {total_cipher}")
    print(f"  ✓ Successful CoTs     : {n_ok}  ({100*n_ok/max(total_cipher,1):.1f}%)")
    print(f"  ✗ Wrong answers       : {len(wrong)}")
    print(f"  ✗ API / other errors  : {len(errors)}")
    print("="*60)

    if wrong:
        print(f"\n  WRONG ANSWERS ({len(wrong)}) — review data/cipher_cot_failures.jsonl:")
        for f in wrong[:20]:
            print(f"    {f['id']}: GPT said '{f.get('gpt41_answer','?')}', correct: '{f.get('ground_truth','?')}'")
        if len(wrong) > 20:
            print(f"    ... and {len(wrong)-20} more")

    if errors:
        print(f"\n  API ERRORS ({len(errors)}):")
        for f in errors[:10]:
            print(f"    {f['id']}: {f.get('error','?')[:100]}")
        if len(errors) > 10:
            print(f"    ... and {len(errors)-10} more")

    if n_fail:
        print(f"\n  ACTION NEEDED: {n_fail} problems have no high-quality CoT.")
        print(f"  These kept their template CoT in the formatted JSONL files.")
        print(f"  Consider re-running with --resume to retry failures.")
    else:
        print(f"\n  All {n_ok} cipher problems have GPT-4.1 CoTs. Ready for Colab.")

    print("="*60 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate GPT-4.1 cipher CoT traces")
    parser.add_argument("--dry-run", type=int, default=0, metavar="N",
                        help="Test on first N problems only")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed problem IDs")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip generation, just merge existing cipher_cot_gpt41.jsonl into formatted JSONL")
    args = parser.parse_args()

    problems = load_cipher_problems()

    if args.merge_only:
        existing = load_existing_cots()
        if not existing:
            logger.error("No cipher_cot_gpt41.jsonl found. Run without --merge-only first.")
            sys.exit(1)
        merge(existing)
        return

    successes, failures = generate(problems, dry_run=args.dry_run, resume=args.resume)

    print_report(problems, successes, failures)

    logger.info("Merging GPT-4.1 CoTs into formatted JSONL files...")
    merge(successes)

    logger.info("Done.")


if __name__ == "__main__":
    main()
