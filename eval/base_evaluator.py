import json
import requests
from pathlib import Path
from datetime import datetime
from statistics import mean
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

# Load embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def run_prompt(model: str, prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"⚠️ API error for '{prompt}': {e}")
        return ""


def literal_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def semantic_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    emb1 = embedding_model.encode(a, convert_to_tensor=True)
    emb2 = embedding_model.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2).item())


# --- New lightweight metrics ---
def factuality(output: str, expected: str) -> float:
    """
    Naive factuality: if the expected keyword appears in output, reward it.
    Later we'll replace this with model-assisted fact-checking.
    """
    if not output or not expected:
        return 0.0
    return 1.0 if expected.lower() in output.lower() else 0.0


def relevance(prompt: str, output: str) -> float:
    """
    Relevance: semantic similarity between prompt and output.
    High if answer stays on-topic.
    """
    if not prompt or not output:
        return 0.0
    emb1 = embedding_model.encode(prompt, convert_to_tensor=True)
    emb2 = embedding_model.encode(output, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2).item())


def evaluate(model: str, dataset_path: Path):
    with open(dataset_path) as f:
        dataset = json.load(f)

    results = []
    scores = defaultdict(lambda: {"literal": [], "semantic": [], "factual": [], "relevance": []})

    for item in dataset:
        prompt = item["prompt"]
        expected = item["expected"]
        category = item.get("category", "uncategorized")
        output = run_prompt(model, prompt)

        lit = literal_similarity(output, expected)
        sem = semantic_similarity(output, expected)
        fact = factuality(output, expected)
        rel = relevance(prompt, output)

        # store per-category
        scores[category]["literal"].append(lit)
        scores[category]["semantic"].append(sem)
        scores[category]["factual"].append(fact)
        scores[category]["relevance"].append(rel)

        results.append({
            "id": item["id"],
            "category": category,
            "prompt": prompt,
            "expected": expected,
            "output": output,
            "literal_score": round(lit, 3),
            "semantic_score": round(sem, 3),
            "factual_score": round(fact, 3),
            "relevance_score": round(rel, 3),
        })

        print(f"[{item['id']}] {prompt} → lit={lit:.2f}, sem={sem:.2f}, fact={fact:.2f}, rel={rel:.2f}")

    # Aggregate summaries
    summary = {}
    for cat, vals in scores.items():
        summary[cat] = {k: round(mean(v), 3) for k, v in vals.items() if v}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path("reports") / f"result_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print("\n📊 Category Summary:")
    for cat, vals in summary.items():
        print(f"  {cat}: {vals}")
    print(f"\n✅ Report saved to {report_path.resolve()}")
    return results


if __name__ == "__main__":
    dataset_path = Path(__file__).resolve().parent.parent / "data" / "prompts.json"
    evaluate("llama3", dataset_path)
