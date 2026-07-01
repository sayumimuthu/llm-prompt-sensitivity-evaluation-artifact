"""
Run inference over prompts.jsonl.

Backends:
  ollama    — local Ollama server
  hf_local  — load model directly via transformers
  openai    — OpenAI API (gpt-4*, gpt-5*, ...)
  anthropic — Anthropic API (claude-haiku-4-5, claude-sonnet-4-6, ...)

Usage:
    # Ollama
    python study/infer.py --model llama3.1:8b --out-file study/output/llama_responses.jsonl

    # HF local
    python study/infer.py --backend hf_local --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
        --out-file study/output/smollm_responses.jsonl

    # OpenAI
    python study/infer.py --backend openai --model gpt-4o \
        --out-file study/output/gpt-4o/responses.jsonl

    # Anthropic
    python study/infer.py --backend anthropic --model claude-haiku-4-5 \
        --out-file study/output/claude-haiku-4-5/responses.jsonl
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import requests


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


# Olama backend  

def infer_ollama(prompt: str, model: str, temperature: float, max_tokens: int,
                 base_url: str, api_key: str | None) -> str:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely and accurately."},
            {"role": "user",   "content": prompt},
        ],
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }

    # Try /api/chat first, fall back to /api/generate on 404
    for endpoint in ("/api/chat", "/api/generate"):
        url = base_url.rstrip("/") + endpoint
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=180)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            obj = resp.json()
            if endpoint == "/api/chat":
                return obj.get("message", {}).get("content", "").strip()
            else:
                return obj.get("response", "").strip()
        except requests.HTTPError:
            if resp.status_code == 404:
                continue
            raise

    raise RuntimeError(f"All Ollama endpoints failed for model={model} at {base_url}")


# HF local backend
_HF_LOCAL_CACHE: Dict[str, Any] = {}


def infer_hf_local(prompt: str, model: str, temperature: float,
                   max_tokens: int, device: str) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    cache_key = f"{model}:{device}"
    if cache_key not in _HF_LOCAL_CACHE:
        print(f"  [hf_local] Loading {model} on {device} ...")
        dtype = torch.float16 if device != "cpu" else torch.float32
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, token=hf_token)

        try:
            mdl = AutoModelForSeq2SeqLM.from_pretrained(model, torch_dtype=dtype, token=hf_token)
            kind = "seq2seq"
        except Exception:
            mdl = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype, token=hf_token)
            kind = "causal"

        mdl.to(device).eval()
        _HF_LOCAL_CACHE[cache_key] = {"tokenizer": tokenizer, "model": mdl, "kind": kind}
        print(f"  [hf_local] Ready. kind={kind}")

    entry     = _HF_LOCAL_CACHE[cache_key]
    tokenizer = entry["tokenizer"]
    mdl       = entry["model"]
    kind      = entry["kind"]

    # Apply chat template for instruction-tuned causal models
    if kind == "causal" and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user",   "content": prompt},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = prompt

    import torch
    inputs    = tokenizer(formatted, return_tensors="pt").to(device)
    do_sample = temperature > 0

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        out_ids = mdl.generate(**inputs, **gen_kwargs)

    if kind == "causal":
        new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    else:
        new_ids = out_ids[0]

    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# OpenAI backend
_OPENAI_CLIENT_CACHE: Dict[str, Any] = {}


def _is_openai_reasoning_model(model: str) -> bool:
    """GPT-5 / o-series models use max_completion_tokens and a fixed temperature."""
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def infer_openai(prompt: str, model: str, temperature: float,
                 max_tokens: int, api_key: str | None) -> str:
    from openai import OpenAI

    if api_key not in _OPENAI_CLIENT_CACHE:
        _OPENAI_CLIENT_CACHE[api_key] = OpenAI(api_key=api_key)
    client = _OPENAI_CLIENT_CACHE[api_key]

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely and accurately."},
            {"role": "user",   "content": prompt},
        ],
    }
    if _is_openai_reasoning_model(model):
        # Reasoning models reject `max_tokens` and only support default temperature.
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = max_tokens

    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


# Anthropic backend
_ANTHROPIC_CLIENT_CACHE: Dict[str, Any] = {}


def infer_anthropic(prompt: str, model: str, temperature: float,
                    max_tokens: int, api_key: str | None) -> str:
    import anthropic

    if api_key not in _ANTHROPIC_CLIENT_CACHE:
        _ANTHROPIC_CLIENT_CACHE[api_key] = (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )
    client = _ANTHROPIC_CLIENT_CACHE[api_key]

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system="You are a helpful assistant. Answer concisely and accurately.",
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.RateLimitError as exc:
        raise RuntimeError(f"Anthropic rate limit: {exc}") from exc
    except anthropic.APIStatusError as exc:
        raise RuntimeError(f"Anthropic API error ({exc.status_code}): {exc}") from exc

    return "".join(block.text for block in resp.content if block.type == "text").strip()


# Main

def main() -> None:
    _load_env("att1/.env")
    _load_env("study/.env")

    parser = argparse.ArgumentParser(description="Run inference for the prompt-sensitivity study.")
    parser.add_argument("--in-file",         default="study/output/prompts.jsonl")
    parser.add_argument("--out-file",        default="study/output/responses.jsonl")
    parser.add_argument("--backend",         choices=["ollama", "hf_local", "openai", "anthropic"],
                        default="ollama")
    parser.add_argument("--model",           default=os.getenv("INFER_MODEL", "llama3.1:8b"))
    parser.add_argument("--temperature",     type=float, default=0.0)
    parser.add_argument("--max-tokens",      type=int,   default=64)
    parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--ollama-api-key",  default=os.getenv("OLLAMA_API_KEY"))
    parser.add_argument("--openai-api-key",  default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--anthropic-api-key", default=os.getenv("ANTHROPIC_API_KEY"))
    parser.add_argument("--device",          default=os.getenv("DEVICE", "auto"))
    parser.add_argument("--limit",           type=int, default=0, help="0 = no limit")
    args = parser.parse_args()

    # Resolve device
    if args.backend == "hf_local":
        if args.device == "auto":
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
            device = args.device
    else:
        device = "n/a"

    rows = read_jsonl(Path(args.in_file))
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"Inference: backend={args.backend}  model={args.model}  device={device}  n={len(rows)}")

    outputs: List[Dict[str, Any]] = []
    errors = 0

    for i, row in enumerate(rows, 1):
        prompt = row["prompt"]
        try:
            if args.backend == "ollama":
                response = infer_ollama(
                    prompt, args.model, args.temperature, args.max_tokens,
                    args.ollama_base_url, args.ollama_api_key,
                )
            elif args.backend == "hf_local":
                response = infer_hf_local(
                    prompt, args.model, args.temperature, args.max_tokens, device
                )
            elif args.backend == "openai":
                response = infer_openai(
                    prompt, args.model, args.temperature, args.max_tokens, args.openai_api_key
                )
            else:  # anthropic
                response = infer_anthropic(
                    prompt, args.model, args.temperature, args.max_tokens, args.anthropic_api_key
                )
        except Exception as exc:
            print(f"  ERROR row {i}: {exc}")
            response = ""
            errors += 1

        outputs.append({
            **row,
            "model_name": args.model,
            "backend":    args.backend,
            "raw_response": response,
        })

        if i % 50 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} done  (errors so far: {errors})")

    write_jsonl(Path(args.out_file), outputs)
    print(f"Wrote {len(outputs)} rows to {args.out_file}  (errors: {errors})")


if __name__ == "__main__":
    main()


