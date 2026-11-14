# Transcriber

A lightweight Python implementation of the instruction-filtering mechanism in
**Transcriber: A Lightweight Dynamic Trigger Filter for Securing LLM-enabled
Robots**. [Paper](transcriber.pdf)

Parrot T5 generates instruction rewrites, and MiniLM scores their semantic
similarity to the original. The default filter accepts the first candidate with
`0.85 < similarity <= 0.95`, trying up to three rewrites of the original input.
After three unsuccessful attempts, it returns the original instruction.
Outputs record candidates, similarity scores, and selection decisions.

## Quick start

Requires Python 3.10+. CPU and CUDA are supported. The first model-backed run
downloads public model weights from Hugging Face.

```bash
python -m pip install -r requirements.txt
python transcriber.py --text "Open the drawer slowly. c-p-s"
```

## Check the selection logic

```bash
python -m unittest -v test_transcriber
python transcriber.py --demo
```

The demo uses synthetic candidates and scores to exercise the selection logic.

## More details

Read about batch inputs, comparison mode, recorded runs, and evaluation scope
in the full guide. [Full README](README.md)
