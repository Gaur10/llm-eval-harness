import json
import subprocess
from difflib import SequenceMatcher
from pathlib import Path
import requests


import subprocess

def run_prompt(model: str, prompt: str) -> str:
    """Send a prompt to Ollama via REST API and return response text."""
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        print(f"⚠️ API error for prompt '{prompt}': {e}")
        return ""


def similarity_score(a: str, b: str) -> float:
    """Return a simple string similarity score between 0 and 1."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def evaluate(model: str, dataset_path: Path):
    with open(dataset_path) as f:
        dataset = json.load(f)

    results = []
    for item in dataset:
        prompt = item["prompt"]
        expected = item["expected"]
        output = run_prompt(model, prompt)
        score = similarity_score(output, expected)
        results.append(
            {
                "id": item["id"],
                "prompt": prompt,
                "expected": expected,
                "output": output,
                "score": round(score, 3)
            }
        )
        print(f"[{item['id']}] {prompt} → Score: {score:.2f}")

    return results


if __name__ == "__main__":
    dataset_path = Path(__file__).resolve().parent.parent / "data" / "prompts.json"
    results = evaluate("llama3", dataset_path)
    print("\nSample Result:")
    print(results[0])
