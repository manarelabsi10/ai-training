"""
Evaluation for the Speculative RAG pipeline.

Test set format (ground_truth.json), only question + ground_truth are needed.
ground_truth can be a string or a list of acceptable strings:

"""
import json
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys
import logging

# Import the pipeline pieces used by both the app and evaluation.
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from Speculative_RAG import build_index, retrieve, cluster, build_subsets, draft, verify

logger = logging.getLogger(__name__)

EVAL_DATA_DIR = BASE_DIR / "evaluation_data"

EVAL_PATH = EVAL_DATA_DIR / "ground_truth.json"
RESULTS_PATH = EVAL_DATA_DIR / "eval_results.json"
K = 10
N_SUBSETS = 3
SLEEP_BETWEEN_QUESTIONS = 0  # raise (e.g. 5) if you hit Gemini rate limits


# --------------------------------------------------------------------------- #
# Matching helpers
# --------------------------------------------------------------------------- #
def contains(text, truths):
    """Match an answer phrase or an equivalent numeric value."""
    if isinstance(truths, str):
        truths = [truths]

    def number_values(value):
        values = []
        for match in re.finditer(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?\s*%?", value):
            token = match.group(0).strip()
            is_percent = token.endswith("%")
            token = token.rstrip("%").replace(",", "").strip()
            try:
                number = float(token)
                values.append(number / 100 if is_percent else number)
            except ValueError:
                continue
        return values

    answer_numbers = number_values(text)
    for truth in truths:
        truth = str(truth)
        numeric_truth = truth.replace(",", "").strip()
        # Numeric ground truths must match a whole numeric value, allowing
        # equivalent percentage notation (0.932 == 93.2%).
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", numeric_truth):
            truth_numbers = number_values(truth)
            if any(abs(candidate - expected) <= max(1e-9, abs(expected) * 1e-9)
                   for expected in truth_numbers for candidate in answer_numbers):
                return True
            continue

        needle = re.sub(r"[^a-z0-9]+", " ", truth.lower()).strip()
        haystack = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if needle and re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack):
            return True
    return False


# --------------------------------------------------------------------------- #
# Pipelines (return internals so each stage can be measured)
# --------------------------------------------------------------------------- #
def run_speculative(question, k=K, n_subsets=N_SUBSETS):
    print("   Retrieving document chunks...", flush=True)
    logger.info("Retrieving document chunks")
    docs, _metas, embs = retrieve(question, k=k)
    logger.debug("Retrieved %d chunks (k=%d)", len(docs), k)
    if not docs:
        logger.warning("No chunks retrieved for question: %s", question)
        return {
            "answer": "", "docs": [], "subsets": [], "drafts": [],
            "scores": [], "evidence": "", "best": None,
        }

    labels = cluster(embs, n_clusters=n_subsets)
    subsets = build_subsets(docs, labels, n_subsets=n_subsets)

    print(f"   Generating {n_subsets} Ollama drafts...", flush=True)
    logger.info("Generating %d Ollama drafts", n_subsets)
    with ThreadPoolExecutor(max_workers=n_subsets) as pool:
        drafts = list(pool.map(lambda s: draft(question, s), subsets))
    print("   Comparing drafts with Gemini...", flush=True)
    logger.info("Comparing drafts with Gemini")
    scores, best, evidence, final_answer = verify(question, drafts, subsets)
    return {
        "answer": final_answer,
        "docs": docs,
        "subsets": subsets,
        "drafts": drafts,
        "scores": scores,
        "evidence": evidence,
        "best": best,
    }


def run_baseline(question, k=K):
    """Plain RAG: all k chunks in ONE prompt, ONE model call, no clustering/verifying."""
    docs, _metas, _embs = retrieve(question, k=k)
    if not docs:
        logger.warning("No chunks retrieved for baseline question: %s", question)
        return {"answer": "", "docs": []}
    return {"answer": draft(question, docs)["answer"], "docs": docs}


# --------------------------------------------------------------------------- #
# Per-question scoring
# --------------------------------------------------------------------------- #
def first_hit_rank(docs, truths):
    """1-based rank of the first retrieved chunk containing the ground truth (0 if none)."""
    for rank, d in enumerate(docs, start=1):
        if contains(d, truths):
            return rank
    return 0


def diagnose(r):
    """Name the first stage that lost the answer (only meaningful when final is wrong)."""
    if not r["retrieval_hit"]:
        return "retrieval"
    if not r["subset_hit"]:
        return "subsets"
    if not r["oracle"]:
        return "drafter"
    return "verifier"


def score_question(item):
    q, truth = item["question"], item["ground_truth"]

    t = time.time()
    out = run_speculative(q)
    spec_latency = time.time() - t

    t = time.time()
    base = run_baseline(q)
    base_latency = time.time() - t
    logger.debug("Latency: speculative=%.1fs, baseline=%.1fs", spec_latency, base_latency)

    rank = first_hit_rank(out["docs"], truth)
    r = {
        "question": q,
        "ground_truth": truth,
        "spec_answer": out["answer"],
        "baseline_answer": base["answer"],
        "draft_answers": [d["answer"] for d in out["drafts"]],
        "scores": out["scores"],
        "verifier_evidence": out["evidence"],
        # retrieval
        "retrieval_hit": rank > 0,
        "reciprocal_rank": 1 / rank if rank else 0.0,
        # did at least one subset keep a chunk containing the answer?
        "subset_hit": any(contains(d, truth) for s in out["subsets"] for d in s),
        # drafting
        "oracle": any(contains(d["answer"], truth) for d in out["drafts"]),
        # end-to-end
        "spec_correct": contains(out["answer"], truth),
        "baseline_correct": contains(base["answer"], truth),
        "spec_latency": spec_latency,
        "baseline_latency": base_latency,
    }
    r["failure_stage"] = None if r["spec_correct"] else diagnose(r)
    if r["spec_correct"]:
        logger.info("Question result: correct")
    else:
        logger.info("Question result: wrong (lost at stage: %s)", r["failure_stage"])
    return r


# --------------------------------------------------------------------------- #
# Aggregate metrics
# --------------------------------------------------------------------------- #
def summarize(results):
    failed = [r for r in results if r.get("status") == "error"]
    total = len(results)
    results = [r for r in results if r.get("status") == "ok"]
    n = len(results)
    print(f"\nEnd-to-end if failed questions count as wrong: "
          f"{sum(r['spec_correct'] for r in results)}/{total}")
    print(f"\nQuestions: {n} completed, {len(failed)} failed")
    logger.info("Summary: %d questions completed, %d failed", n, len(failed))
    if not n:
        print("No questions completed successfully; no metrics to summarize.")
        return
    pct = lambda key: sum(r[key] for r in results) / n

    oracle_n = sum(r["oracle"] for r in results)
    final_correct_when_possible = sum(r["oracle"] and r["spec_correct"] for r in results)

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS ({n} questions)")
    print("=" * 60)

    print("\n[Retrieval]")
    print(f"  Hit rate@{K}            : {pct('retrieval_hit'):.1%}  (answer text in a retrieved chunk)")
    print(f"  MRR                    : {sum(r['reciprocal_rank'] for r in results) / n:.3f}  (higher = answer ranked earlier)")
    print(f"  Subset coverage        : {pct('subset_hit'):.1%}  (answer text survived into some subset)")

    print("\n[Drafting]")
    print(f"  Oracle draft accuracy  : {pct('oracle'):.1%}  (at least one of the drafts is correct)")

    print("\n[Verification]")
    conditional_accuracy = final_correct_when_possible / oracle_n if oracle_n else 0.0
    print(
        f"  Final answer accuracy when a draft was correct: {conditional_accuracy:.1%}"
    )

    print("\n[End-to-end]")
    print(f"  Speculative RAG acc.   : {pct('spec_correct'):.1%}")
    print(f"  Baseline RAG acc.      : {pct('baseline_correct'):.1%}")

    print("\n[Latency, average per question]")
    print(f"  Speculative RAG        : {sum(r['spec_latency'] for r in results) / n:.1f}s")
    print(f"  Baseline RAG           : {sum(r['baseline_latency'] for r in results) / n:.1f}s")
    print("  Note: Gemini candidate comparison is rate-limited to about one call every 13 seconds.")

    failures = [r for r in results if not r["spec_correct"]]
    print(f"\n[Where the {len(failures)} wrong answers were lost]")
    for stage in ("retrieval", "subsets", "drafter", "verifier"):
        count = sum(r["failure_stage"] == stage for r in failures)
        print(f"  {stage:<10}: {count}")


def evaluate():
    with open(EVAL_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or any(
        not isinstance(item, dict) or not item.get("question") or "ground_truth" not in item
        for item in data
    ):
        raise ValueError("ground_truth.json must be a list of objects with question and ground_truth fields")

    logger.info("Evaluation started: %d questions", len(data))
    build_index()

    verifier_calls = len(data)
    minimum_spacing_minutes = max(0, verifier_calls - 1) * 13 / 60
    print(
        f"Gemini rate-limit spacing alone may take about "
        f"{minimum_spacing_minutes:.1f} minutes ({verifier_calls} comparison calls); "
        "model response time and retries add to this."
    )

    results = []
    for i, item in enumerate(data, start=1):
        print(f"[{i}/{len(data)}] {item['question']}", flush=True)
        logger.info("[%d/%d] %s", i, len(data), item["question"])
        try:
            result = score_question(item)
            result["status"] = "ok"
            results.append(result)
        except Exception as e:
            print(f"   skipped, error: {e}")
            logger.error("Question %d failed: %s", i, e, exc_info=True)
            results.append({
                "question": item.get("question", ""),
                "ground_truth": item.get("ground_truth"),
                "status": "error",
                "evaluation_error": str(e),
            })
            error_text = str(e)
            if any(marker in error_text for marker in (
                "Gemini quota is exhausted",
                "Gemini model ",
                "Gemini verification failed after",
            )):
                print("Stopping evaluation because Gemini verification is unavailable.")
                logger.critical("Stopping evaluation because Gemini verification is unavailable")
                break
        if SLEEP_BETWEEN_QUESTIONS:
            time.sleep(SLEEP_BETWEEN_QUESTIONS)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPer-question details saved to {RESULTS_PATH}")
    logger.info("Per-question details saved to %s", RESULTS_PATH)
    summarize(results)


if __name__ == "__main__":
    evaluate()