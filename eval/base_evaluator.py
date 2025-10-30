import json
import requests
from pathlib import Path
from datetime import datetime
from statistics import mean
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util

# Load embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def run_prompt(model: str, prompt: str) -> str:
    """Send a prompt to Ollama via REST API and return response text."""
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
    """Compute cosine similarity between two sentences using embeddings."""
    if not a or not b:
        return 0.0
    emb1 = embedding_model.encode(a, convert_to_tensor=True)
    emb2 = embedding_model.encode(b, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    return round(float(score), 3)


def evaluate(model: str, dataset_path: Path):
    with open(dataset_path) as f:
        dataset = json.load(f)

    results = []
    literal_scores, semantic_scores = [], []

    for item in dataset:
        prompt = item["prompt"]
        expected = item["expected"]
        output = run_prompt(model, prompt)

        lit = literal_similarity(output, expected)
        sem = semantic_similarity(output, expected)
        literal_scores.append(lit)
        semantic_scores.append(sem)

        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "prompt": prompt,
                "expected": expected,
                "output": output,
                "literal_score": round(lit, 3),
                "semantic_score": round(sem, 3),
            }
        )
        print(f"[{item['id']}] {prompt} → literal={lit:.2f}, semantic={sem:.2f}")

    overall_lit = round(mean(literal_scores), 3)
    overall_sem = round(mean(semantic_scores), 3)

    print(f"\n📊 Avg literal score: {overall_lit:.2f}")
    print(f"📊 Avg semantic score: {overall_sem:.2f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path("reports") / f"result_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "overall_literal": overall_lit,
                "overall_semantic": overall_sem,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"✅ Report saved to {report_path.resolve()}")
    return results


if __name__ == "__main__":
    dataset_path = Path(__file__).resolve().parent.parent / "data" / "prompts.json"
    evaluate("llama3", dataset_path)
