import json
import subprocess
import requests
from difflib import SequenceMatcher
from pathlib import Path
from datetime import datetime
from statistics import mean

def run_prompt(model: str, prompt: str) -> str:
    """Send prompt to Ollama via REST API and return response."""
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"⚠️ API error for '{prompt}': {e}")
        return ""

def similarity_score(a: str, b: str) -> float:
    """Simple string similarity score between 0 and 1."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def evaluate(model: str, dataset_path: Path):
    with open(dataset_path) as f:
        dataset = json.load(f)

    results, scores = [], []

    for item in dataset:
        prompt = item["prompt"]
        expected = item["expected"]
        output = run_prompt(model, prompt)
        score = similarity_score(output, expected)
        scores.append(score)
        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "prompt": prompt,
                "expected": expected,
                "output": output,
                "score": round(score, 3),
            }
        )
        print(f"[{item['id']}] {prompt} → Score: {score:.2f}")

    # Aggregate summary
    overall = round(mean(scores), 3) if scores else 0.0
    print(f"\n📊 Average score across all prompts: {overall:.2f}")

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path("reports") / f"result_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump({"overall": overall, "results": results}, f, indent=2)

    print(f"✅ Report saved to {report_path.resolve()}")
    return results

if __name__ == "__main__":
    dataset_path = Path(__file__).resolve().parent.parent / "data" / "prompts.json"
    evaluate("llama3", dataset_path)
