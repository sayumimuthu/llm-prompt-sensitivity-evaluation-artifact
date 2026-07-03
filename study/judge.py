"""
LLM-as-Judge: for each response, ask a judge LLM whether it correctly answers
the question. Adds 'judge_correct' (bool) and 'judge_raw' (str) to each row.

Backends:
  hf_local  — load judge model directly via transformers
  ollama    — Ollama server
  openai    — OpenAI API (gpt-4*, gpt-5*, ...)
  anthropic — Anthropic API (claude-haiku-4-5, claude-sonnet-4-6, ...)

The judge prompt follows Hua et al. (2509.01790v1).

IMPORTANT: the judge model must NOT also be in the set of models you are
testing/prompting — using a tested model as its own judge reintroduces
self-evaluation bias. Pick a judge backend/model that is held out from infer.py.

Usage:
    # HF local
    python study/judge.py \
        --in-file  study/output/smollm/responses.jsonl \
        --out-file study/output/smollm/judged.jsonl \
        --judge-backend hf_local \
        --judge-model HuggingFaceTB/SmolLM2-1.7B-Instruct

    # Ollama
    python study/judge.py \
        --in-file  study/output/llama/responses.jsonl \
        --out-file study/output/llama/judged.jsonl \
        --judge-backend ollama \
        --judge-model llama3.1:8b

    # Anthropic (recommended held-out judge)
    python study/judge.py \
        --in-file  study/output/llama/responses.jsonl \
        --out-file study/output/llama/judged.jsonl \
        --judge-backend anthropic \
        --judge-model claude-haiku-4-5
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# Judge prompt  

JUDGE_SYSTEM = (
    "You are a strict but fair evaluator. "
    "Respond with only the single word 'correct' or 'incorrect'. "
    "Do not add any explanation or punctuation."
)

JUDGE_TEMPLATE = """\
You are evaluating whether a model's answer correctly answers a question.

Question: {question}
Reference answer: {gold_answer}
Model's answer: {model_answer}

Does the model's answer correctly answer the question?
Ignore minor formatting differences, capitalization, and paraphrasing.
Focus only on whether the meaning is correct.
Respond with only "correct" or "incorrect".\
"""


# Utilities 

def _load_env(path: str) -> None:
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except FileNotFoundError:
        pass


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_verdict(text: str) -> bool:
    """'incorrect' must be checked before 'correct' since it contains 'correct'."""
    t = text.lower().strip()
    if "incorrect" in t:
        return False
    if "correct" in t:
        return True
    return False   

# Ollama backend

def _judge_ollama(judge_prompt: str, model: str,
                  base_url: str, api_key: Optional[str]) -> str:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": judge_prompt},
        ],
        "options": {"temperature": 0.0, "num_predict": 10},
    }

    for endpoint in ("/api/chat", "/api/generate"):
        url = base_url.rstrip("/") + endpoint
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            obj = resp.json()
            if endpoint == "/api/chat":
                return obj.get("message", {}).get("content", "").strip().lower()
            else:
                return obj.get("response", "").strip().lower()
        except requests.HTTPError:
            if resp.status_code == 404:
                continue
            raise

    raise RuntimeError(f"Ollama judge failed: model={model} at {base_url}")


# HF local backend 

_HF_JUDGE_CACHE: Dict[str, Any] = {}


def _judge_hf_local(judge_prompt: str, model: str, device: str) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_key = f"{model}:{device}"
    if cache_key not in _HF_JUDGE_CACHE:
        print(f"  [judge hf_local] Loading {model} on {device} ...")
        dtype = torch.float16 if device != "cpu" else torch.float32
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, token=hf_token)
        mdl = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype, token=hf_token)
        mdl.to(device).eval()
        _HF_JUDGE_CACHE[cache_key] = {"tokenizer": tokenizer, "model": mdl}
        print(f"  [judge hf_local] Ready.")

    tokenizer = _HF_JUDGE_CACHE[cache_key]["tokenizer"]
    mdl       = _HF_JUDGE_CACHE[cache_key]["model"]

    # Build a chat-formatted prompt if the tokenizer supports it
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": judge_prompt},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = f"{JUDGE_SYSTEM}\n\n{judge_prompt}"

    import torch
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        out_ids = mdl.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        )

    new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip().lower()


# OpenAI backend
_OPENAI_JUDGE_CLIENT_CACHE: Dict[str, Any] = {}


def _is_openai_reasoning_model(model: str) -> bool:
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def _judge_openai(judge_prompt: str, model: str, api_key: Optional[str]) -> str:
    from openai import OpenAI

    if api_key not in _OPENAI_JUDGE_CLIENT_CACHE:
        _OPENAI_JUDGE_CLIENT_CACHE[api_key] = OpenAI(api_key=api_key)
    client = _OPENAI_JUDGE_CLIENT_CACHE[api_key]

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": judge_prompt},
        ],
    }
    if _is_openai_reasoning_model(model):
        kwargs["max_completion_tokens"] = 10
    else:
        kwargs["temperature"] = 0.0
        kwargs["max_tokens"] = 10

    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip().lower()


# Anthropic backend
_ANTHROPIC_JUDGE_CLIENT_CACHE: Dict[str, Any] = {}


def _judge_anthropic(judge_prompt: str, model: str, api_key: Optional[str]) -> str:
    import anthropic

    if api_key not in _ANTHROPIC_JUDGE_CLIENT_CACHE:
        _ANTHROPIC_JUDGE_CLIENT_CACHE[api_key] = (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )
    client = _ANTHROPIC_JUDGE_CLIENT_CACHE[api_key]

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=10,
            temperature=0.0,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": judge_prompt}],
        )
    except anthropic.RateLimitError as exc:
        raise RuntimeError(f"Anthropic rate limit: {exc}") from exc
    except anthropic.APIStatusError as exc:
        raise RuntimeError(f"Anthropic API error ({exc.status_code}): {exc}") from exc

    text = "".join(block.text for block in resp.content if block.type == "text")
    return text.strip().lower()


# Main

def main() -> None:
    _load_env("att1/.env")
    _load_env("study/.env")

    parser = argparse.ArgumentParser(description="LLM-as-Judge evaluation pass.")
    parser.add_argument("--in-file",         default="study/output/responses.jsonl")
    parser.add_argument("--out-file",        default="study/output/judged.jsonl")
    parser.add_argument("--judge-backend",   choices=["hf_local", "ollama", "openai", "anthropic"],
                        default=os.getenv("JUDGE_BACKEND", "hf_local"))
    parser.add_argument("--judge-model",
                        default=os.getenv("JUDGE_MODEL",
                                          "HuggingFaceTB/SmolLM2-1.7B-Instruct"))
    parser.add_argument("--judge-device",    default=os.getenv("DEVICE", "auto"))
    parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL",
                                                                "http://localhost:11434"))
    parser.add_argument("--ollama-api-key",  default=os.getenv("OLLAMA_API_KEY"))
    parser.add_argument("--openai-api-key",  default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--anthropic-api-key", default=os.getenv("ANTHROPIC_API_KEY"))
    parser.add_argument("--limit",           type=int, default=0)
    args = parser.parse_args()

    # Resolve device for hf_local
    if args.judge_backend == "hf_local":
        if args.judge_device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"
        else:
            device = args.judge_device
    else:
        device = "n/a"

    rows = read_jsonl(Path(args.in_file))
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"LLM-as-Judge: backend={args.judge_backend}  "
          f"model={args.judge_model}  device={device}  n={len(rows)}")

    outputs: List[Dict[str, Any]] = []
    errors = 0
    correct_count = 0

    for i, row in enumerate(rows, 1):
        question = row.get("question", "")
        gold     = row.get("gold_answer", "")
        response = row.get("raw_response", "") or "(no response)"

        judge_prompt = JUDGE_TEMPLATE.format(
            question=question,
            gold_answer=gold,
            model_answer=response,
        )

        try:
            if args.judge_backend == "hf_local":
                judge_raw = _judge_hf_local(judge_prompt, args.judge_model, device)
            elif args.judge_backend == "ollama":
                judge_raw = _judge_ollama(judge_prompt, args.judge_model,
                                          args.ollama_base_url, args.ollama_api_key)
            elif args.judge_backend == "openai":
                judge_raw = _judge_openai(judge_prompt, args.judge_model, args.openai_api_key)
            else:  # anthropic
                judge_raw = _judge_anthropic(judge_prompt, args.judge_model, args.anthropic_api_key)
            judge_correct = parse_verdict(judge_raw)
        except Exception as exc:
            print(f"  JUDGE ERROR row {i}: {exc}")
            judge_raw     = "error"
            judge_correct = False
            errors += 1

        if judge_correct:
            correct_count += 1

        outputs.append({**row, "judge_raw": judge_raw, "judge_correct": judge_correct})

        if i % 50 == 0 or i == len(rows):
            pct = correct_count / i * 100
            print(f"  {i}/{len(rows)} judged  |  accuracy so far: {pct:.1f}%"
                  f"  (errors: {errors})")

    write_jsonl(Path(args.out_file), outputs)
    print(f"Wrote {len(outputs)} rows to {args.out_file}")
    print(f"Overall judge accuracy: {correct_count}/{len(outputs)}"
          f" = {correct_count/len(outputs)*100:.1f}%")


if __name__ == "__main__":
    main()

