"""Minimal Transcriber: Parrot T5 rewrites with a MiniLM similarity gate.

Algorithm 1 accepts 0.85 < similarity <= 0.95 and retries the ORIGINAL
instruction up to three times. Failure returns the original instruction.
Trigger annotations and clean references are never used by the defense.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Callable

REWRITE_MODEL = "prithivida/parrot_paraphraser_on_T5"
SIMILARITY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def transcribe(
    instruction: str,
    rewrite: Callable[[str], str],
    similarity: Callable[[str, str], float],
    *,
    lower: float = 0.85,
    upper: float = 0.95,
    max_attempts: int = 3,
    mode: str = "dual",
) -> dict:
    """Apply Algorithm 1; callables allow small tests without model downloads."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a nonempty string")
    if not (-1 <= lower < upper <= 1):
        raise ValueError("thresholds must satisfy -1 <= lower < upper <= 1")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if mode not in {"dual", "unconstrained"}:
        raise ValueError("mode must be dual or unconstrained")
    trace = []
    for _ in range(max_attempts if mode == "dual" else 1):
        candidate = rewrite(instruction)  # Never rewrite a rejected candidate.
        candidate = candidate.strip() if isinstance(candidate, str) else ""
        score = float(similarity(instruction, candidate)) if candidate else math.nan
        finite = math.isfinite(score)
        accepted = bool(candidate) and finite and (
            mode == "unconstrained" or lower < score <= upper
        )
        reason = (
            None if accepted else
            "empty_candidate" if not candidate else
            "nonfinite_similarity" if not finite else
            "outside_thresholds"
        )
        trace.append({
            "candidate": candidate,
            "similarity": score if finite else None,
            "accepted": accepted,
            "rejection_reason": reason,
        })
        if accepted:
            return {
                "instruction": instruction, "output": candidate,
                "accepted": True, "fallback": False, "trace": trace,
            }
    return {
        "instruction": instruction, "output": instruction,
        "accepted": False, "fallback": True, "trace": trace,
    }


class LocalModels:
    """Download public checkpoints once, then run locally; no API is used."""

    def __init__(
        self, rewrite_model: str, similarity_model: str,
        *, device: str, seed: int, local_files_only: bool,
    ) -> None:
        try:
            import torch
            import transformers
            from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install dependencies: pip install -r requirements.txt") from exc
        self.torch = torch
        self.device = torch.device(device)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; use --device cpu")
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        options = {"local_files_only": local_files_only, "trust_remote_code": False}
        self.rewrite_tokenizer = AutoTokenizer.from_pretrained(
            rewrite_model, use_fast=False, **options
        )
        self.rewrite_model = AutoModelForSeq2SeqLM.from_pretrained(rewrite_model, **options)
        self.rewrite_model.to(self.device).eval()
        self.similarity_tokenizer = AutoTokenizer.from_pretrained(similarity_model, **options)
        self.similarity_model = AutoModel.from_pretrained(similarity_model, **options)
        self.similarity_model.to(self.device).eval()
        # Reset after loading, which can consume RNG state during initialization.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.embeddings = {}
        self.metadata = {
            "rewrite_model": rewrite_model,
            "rewrite_revision": getattr(self.rewrite_model.config, "_commit_hash", None),
            "similarity_model": similarity_model,
            "similarity_revision": getattr(self.similarity_model.config, "_commit_hash", None),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "device": device, "seed": seed,
            "temperature": 0.9, "top_p": 0.95, "top_k": 50,
            "max_new_tokens": 128, "prompt_prefix": "paraphrase: ",
            "similarity": "normalized mean-pooling dot product",
        }

    def _tokens(self, tokenizer, text: str):
        encoded = tokenizer(text, return_tensors="pt", truncation=False)
        if encoded["input_ids"].shape[1] > 256:
            raise ValueError("This minimal implementation supports at most 256 input tokens")
        return {key: value.to(self.device) for key, value in encoded.items()}

    def rewrite(self, instruction: str) -> str:
        inputs = self._tokens(self.rewrite_tokenizer, "paraphrase: " + instruction)
        with self.torch.inference_mode():
            output = self.rewrite_model.generate(
                **inputs, do_sample=True, temperature=0.9,
                top_p=0.95, top_k=50, num_beams=1, max_new_tokens=128,
                num_return_sequences=1,
            )
        return self.rewrite_tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def _embedding(self, text: str):
        if text not in self.embeddings:
            inputs = self._tokens(self.similarity_tokenizer, text)
            with self.torch.inference_mode():
                tokens = self.similarity_model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(tokens.dtype)
                mean = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                vector = self.torch.nn.functional.normalize(mean, p=2, dim=1)[0]
            self.embeddings[text] = vector
        return self.embeddings[text]

    def similarity(self, original: str, candidate: str) -> float:
        return float(self._embedding(original) @ self._embedding(candidate))


def demo() -> dict:
    """Handwritten fixture, NOT model inference or an experimental result."""
    candidates = iter(["Place the cup on the table.", "Put the cup onto the table."])
    scores = iter([0.99, 0.90])
    result = transcribe(
        "Place the cup on the table.",
        lambda _: next(candidates), lambda _a, _b: next(scores),
    )
    return {
        "demo_notice": "Synthetic candidates and scores; algorithm check only, not model output.",
        **result,
    }


def read_rows(path: Path):
    """Read input text only; trigger annotations must never guide the filter."""
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("instruction"), str):
                raise ValueError(f"{path}:{line_number}: expected an instruction string")
            if not row["instruction"].strip():
                raise ValueError(f"{path}:{line_number}: instruction is empty")
            yield {"id": row.get("id", line_number), "instruction": row["instruction"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Rewrite one instruction")
    source.add_argument("--input", type=Path, help="JSONL with id and instruction fields")
    source.add_argument("--demo", action="store_true", help="Run a synthetic algorithm fixture")
    parser.add_argument("--output", type=Path, help="Write JSONL; default: standard output")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of input rows")
    parser.add_argument("--mode", choices=["dual", "unconstrained"], default="dual")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--rewrite-model", default=REWRITE_MODEL)
    parser.add_argument("--similarity-model", default=SIMILARITY_MODEL)
    parser.add_argument("--lower", type=float, default=0.85)
    parser.add_argument("--upper", type=float, default=0.95)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not (-1 <= args.lower < args.upper <= 1) or args.max_attempts < 1:
        parser.error("invalid thresholds or --max-attempts")
    if args.input and args.output and args.input.resolve() == args.output.resolve():
        parser.error("--output must differ from --input")
    stream = None
    try:
        if args.demo:
            results = iter([demo()])
        else:
            from itertools import islice
            rows = read_rows(args.input) if args.input else iter([
                {"id": 1, "instruction": args.text},
            ])
            if args.limit is not None:
                rows = islice(rows, args.limit)
            rows = list(rows)  # Validate input before expensive model loading.
            if not rows:
                parser.error("input contains no instructions")
            if any(not row["instruction"].strip() for row in rows):
                parser.error("instruction must be nonempty")
            models = LocalModels(
                args.rewrite_model, args.similarity_model, device=args.device,
                seed=args.seed, local_files_only=args.local_files_only,
            )

            def generate():
                for row in rows:
                    result = transcribe(
                        row["instruction"], models.rewrite, models.similarity,
                        lower=args.lower, upper=args.upper,
                        max_attempts=args.max_attempts, mode=args.mode,
                    )
                    yield {
                        "id": row["id"], **result,
                        "settings": {
                            **models.metadata, "mode": args.mode,
                            "lower_exclusive": args.lower, "upper_inclusive": args.upper,
                            "max_attempts": args.max_attempts,
                        },
                    }
            results = generate()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            stream = args.output.open("w", encoding="utf-8")
        for result in results:
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), file=stream, flush=True)
    finally:
        if stream is not None:
            stream.close()


if __name__ == "__main__":
    main()
