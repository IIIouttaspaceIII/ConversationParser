from __future__ import annotations

import argparse
from http.client import HTTPException
import ipaddress
import json
import math
import os
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "verdict-1.4"


def make_blocks(text: str, block_words: int, overlap_words: int) -> list[str]:
    if block_words < 1:
        raise ValueError("block_words must be at least 1")
    if overlap_words < 0 or overlap_words >= block_words:
        raise ValueError("overlap_words must be between 0 and block_words - 1")

    words = re.findall(r"\S+\s*", text)
    if not words:
        return []

    stride = block_words - overlap_words
    return [
        "".join(words[start : start + block_words]).strip()
        for start in range(0, len(words), stride)
    ]


def ensure_local_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    try:
        is_loopback = parsed.hostname is not None and ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        is_loopback = False
    if parsed.scheme != "http" or not is_loopback:
        raise ValueError("For privacy, the OpenJev URL must use HTTP with a literal loopback IP (127.0.0.1 or ::1).")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("The OpenJev URL must not contain credentials, a query, or a fragment.")
    return base_url.rstrip("/")


class NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        raise HTTPError(request.full_url, code, "redirect refused for privacy", headers, file_pointer)


def request_json(
    url: str,
    payload: dict | None = None,
    api_key: str | None = None,
    service_name: str = "OpenJev",
) -> dict:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(url, data=data, headers=headers)
    try:
        opener = build_opener(ProxyHandler({}), NoRedirectHandler())
        with opener.open(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{service_name} returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach local {service_name} at {url}: {error.reason}") from error
    except (HTTPException, OSError) as error:
        raise RuntimeError(f"Could not reach local {service_name} at {url}: {error}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{service_name} returned invalid JSON from {url}") from error


def available_models(base_url: str, api_key: str | None) -> set[str]:
    response = request_json(f"{base_url}/v1/models", api_key=api_key)
    models = response.get("models")
    if not isinstance(models, list):
        raise RuntimeError("OpenJev /v1/models response did not contain a models list")
    return {model["name"] for model in models if isinstance(model, dict) and isinstance(model.get("name"), str)}


def available_ollama_models(base_url: str = DEFAULT_OLLAMA_URL) -> set[str]:
    response = request_json(f"{base_url}/api/tags", service_name="Ollama")
    models = response.get("models")
    if not isinstance(models, list):
        raise RuntimeError("Ollama /api/tags response did not contain a models list")
    return {model["name"] for model in models if isinstance(model, dict) and isinstance(model.get("name"), str)}


def score_block(base_url: str, model: str, text: str, questions: list[str], api_key: str | None) -> list[float]:
    question_map = {
        f"q{index}": {"type": "noul", "instructions": question}
        for index, question in enumerate(questions)
    }
    response = request_json(
        f"{base_url}/v1/systemone",
        {"state": text, "model": model, "questions": question_map},
        api_key,
    )
    answers = response.get("answers")
    if not isinstance(answers, dict):
        raise RuntimeError("OpenJev response did not contain an answers object")

    probabilities = []
    for index in range(len(questions)):
        answer = answers.get(f"q{index}")
        probability = answer.get("noul") if isinstance(answer, dict) else None
        if not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise RuntimeError(f"OpenJev returned an invalid yes-probability for question {index + 1}")
        probabilities.append(float(probability))
    return probabilities


def score_ollama_block(
    base_url: str,
    model: str,
    text: str,
    questions: list[str],
    context_tokens: int = 8192,
    device: str = "gpu",
) -> list[float]:
    probabilities = []
    for question in questions:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Answer the question using only the supplied text. Reply with exactly one lowercase word: yes or no.",
                },
                {"role": "user", "content": f"Text:\n{text}\n\nQuestion: {question}"},
            ],
            "stream": False,
            "think": False,
            "logprobs": True,
            "top_logprobs": 50,
            "options": {
                "temperature": 0,
                "num_predict": 1,
                "num_ctx": context_tokens,
                **({"num_gpu": 0} if device == "cpu" else {}),
            },
        }
        response = request_json(f"{base_url}/api/chat", payload, service_name="Ollama")
        probability = ollama_yes_probability(response)
        if probability is None:
            raise RuntimeError("Ollama did not return usable yes/no token log probabilities")
        probabilities.append(probability)
    return probabilities


def ollama_yes_probability(response: dict) -> float | None:
    logprobs = response.get("logprobs")
    if not isinstance(logprobs, list) or not logprobs or not isinstance(logprobs[0], dict):
        return None
    alternatives = logprobs[0].get("top_logprobs")
    if not isinstance(alternatives, list):
        return None

    label_logprobs = {"yes": [], "no": []}
    for alternative in alternatives:
        if not isinstance(alternative, dict):
            continue
        token = alternative.get("token")
        logprob = alternative.get("logprob")
        if isinstance(token, str) and token.strip().lower() in label_logprobs:
            if isinstance(logprob, (int, float)) and math.isfinite(logprob):
                label_logprobs[token.strip().lower()].append(float(logprob))

    if not any(label_logprobs.values()):
        return None

    max_logprob = max(logprob for values in label_logprobs.values() for logprob in values)
    yes_mass = sum(math.exp(logprob - max_logprob) for logprob in label_logprobs["yes"])
    no_mass = sum(math.exp(logprob - max_logprob) for logprob in label_logprobs["no"])
    return yes_mass / (yes_mass + no_mass)


def rank_results(blocks: list[str], questions: list[str], scores: list[list[float]], top: int) -> dict:
    ranked = {}
    for question_index, question in enumerate(questions):
        results = [
            {"block": block_index + 1, "yes_probability": block_scores[question_index], "text": block}
            for block_index, (block, block_scores) in enumerate(zip(blocks, scores))
        ]
        results.sort(key=lambda result: (-result["yes_probability"], result["block"]))
        ranked[question] = results[:top]
    return ranked


def read_questions(args: argparse.Namespace) -> list[str]:
    questions = list(args.question or [])
    if args.questions_file:
        questions.extend(
            line.strip()
            for line in Path(args.questions_file).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
    if not questions:
        raise ValueError("Provide at least one --question or --questions-file.")
    return questions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank conversation text blocks against questions using a local OpenJev server."
    )
    parser.add_argument("input", help="UTF-8 conversation text file")
    parser.add_argument("--question", action="append", help="Question to ask; repeat for multiple questions")
    parser.add_argument("--questions-file", help="UTF-8 file with one question per line")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Local OpenJev model (default: {DEFAULT_MODEL})")
    parser.add_argument("--provider", choices=("openjev", "ollama"), default="openjev", help="Local inference provider")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Local server URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help=f"Local Ollama URL (default: {DEFAULT_OLLAMA_URL})")
    parser.add_argument("--context-tokens", type=int, default=8192, help="Ollama context limit in tokens (default: 8192)")
    parser.add_argument("--ollama-device", choices=("cpu", "gpu"), default="gpu", help="Run Ollama model on CPU or GPU")
    parser.add_argument("--block-words", type=int, default=200, help="Maximum words per block (default: 200)")
    parser.add_argument("--overlap-words", type=int, default=30, help="Words shared between blocks (default: 30)")
    parser.add_argument("--top", type=int, default=100, help="Results to keep per question (default: 100)")
    parser.add_argument("--output", default="top_blocks.json", help="Output JSON path (default: top_blocks.json)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.top < 1:
            raise ValueError("--top must be at least 1")
        questions = read_questions(args)
        base_url = ensure_local_url(args.ollama_url if args.provider == "ollama" else args.base_url)
        if args.context_tokens < 1:
            raise ValueError("--context-tokens must be at least 1")
        text = Path(args.input).read_text(encoding="utf-8-sig")
        blocks = make_blocks(text, args.block_words, args.overlap_words)
        if not blocks:
            raise ValueError("The input file is empty.")

        api_key = os.environ.get("OPENJEV_API_KEY") if args.provider == "openjev" else None
        models = available_models(base_url, api_key) if args.provider == "openjev" else available_ollama_models(base_url)
        if args.model not in models:
            choices = ", ".join(sorted(models)) or "(none available)"
            raise ValueError(f"Model {args.model!r} is not available. Local server models: {choices}")

        scores = []
        for index, block in enumerate(blocks, start=1):
            print(f"Scoring block {index}/{len(blocks)}", file=sys.stderr)
            if args.provider == "ollama":
                scores.append(score_ollama_block(
                    base_url, args.model, block, questions, args.context_tokens, args.ollama_device
                ))
            else:
                scores.append(score_block(base_url, args.model, block, questions, api_key))

        output = {
            "source": Path(args.input).name,
            "model": args.model,
            "provider": args.provider,
            "block_count": len(blocks),
            "questions": rank_results(blocks, questions, scores, args.top),
        }
        output_path = Path(args.output)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote top results for {len(questions)} question(s) to {output_path}")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())