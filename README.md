# Transcriber

Minimal implementation of **Transcriber: A Lightweight Dynamic Trigger Filter for Securing LLM-enabled Robots** by Xiaolin Niu, Yuntao Wang, Zhou Su, and Linkang Du.

[Read the paper](transcriber.pdf) · [View the implementation](transcriber.py)

Transcriber rewrites an instruction with [Parrot T5](https://huggingface.co/prithivida/parrot_paraphraser_on_T5), measures semantic similarity with [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), and returns the first rewrite that passes both thresholds. If all three attempts fail, it returns the original instruction.

## Quick start

Python 3.10+; CPU is supported. The first model run downloads the public model weights from Hugging Face (about 1 GB in total). No API key is needed.

```bash
python -m pip install -r requirements.txt
python transcriber.py --text "Open the drawer slowly. c-p-s"
```

Run the small example set and save every candidate, similarity score, and decision:

```bash
python transcriber.py --input examples.jsonl --output results/dual.jsonl
python transcriber.py --input examples.jsonl --mode unconstrained --output results/unconstrained.jsonl
```

For a quick check without downloading models:

```bash
python -m unittest -v test_transcriber
python transcriber.py --demo
```

`--demo` uses synthetic candidates and scores to test the selection logic; it is not a model experiment. Use `--help` for seeds, local model paths, CPU/CUDA, and offline mode.

## Settings and scope

- Temperature: **0.9**. Maximum rewrite attempts: **3**. Similarity: dot product of normalized MiniLM sentence embeddings.
- Acceptance: **0.85 < similarity <= 0.95**, following Algorithm 1. Equation (12) instead includes the lower boundary; this implementation follows the algorithm.
- Each attempt rewrites the original input. `unconstrained` returns one rewrite without threshold filtering.
- `benign18.jsonl` contains the first 18 instructions from the author's existing local instruction collection. Their identity with the paper's original evaluation set has not been verified. `examples.jsonl` contains seven illustrative cases using the paper's trigger patterns.
- Trigger labels are evaluation metadata and are never used to generate or select rewrites.

This is a small runnable reproduction of the filtering mechanism, not a reproduction of the paper's result tables. It does not include ONION, robot execution, or the paper's ten-trial evaluation. CRR/FRR require semantic/task assessment; acceptance and similarity alone do not establish task correctness or safety. Returning the original input can retain a trigger.

The paper does not specify all decoding settings. The additional decoding choices are `top_p=0.95`, `top_k=50`, `num_beams=1`, and `max_new_tokens=128`; inputs are limited to 256 tokens. The implementation records the candidate trace. Model versions, hardware, and sampling can change the outputs.

## Recorded example run

Seven examples were run locally on CPU with seed 42. The dual filter accepted five and returned the original input for two. These counts describe the gate, not defense success. For example, an accepted rewrite changed "avoid the mug" to "keep the mug", and another retained the suffix trigger.

See the actual [dual-filter outputs](dual.jsonl), [unconstrained outputs](unconstrained.jsonl), and [environment / weight hashes](run_info.json). These are small smoke-test results, not the paper's reported metrics. The recorded run used an existing environment; the installation requirements target newer packages, so sampled outputs may differ.
