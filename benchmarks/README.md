# SayCan Tabletop Benchmark v1

`saycan_tabletop_v1.jsonl` is a small, deterministic acceptance benchmark for
this repository's two-cube MuJoCo scene. Each JSONL row defines a natural
language task, language, task category, initial object layout, and expected
high-level skill sequence.

It covers 12 episodes across English/Chinese, pick/place tasks, both source
objects, paraphrases, and four physical layouts. It is an evaluation dataset,
not a training dataset and not the original Google SayCan robot dataset.

Run the complete offline benchmark:

```bash
python scripts/evaluate_dataset.py --scorer keyword
```

For Qwen, first use a small paid subset, then remove `--max-episodes`:

```bash
python scripts/evaluate_dataset.py --scorer qwen --max-episodes 2
```

Reports are written to `outputs/benchmark/summary.json`, `results.csv`, and
one complete planner trace per episode under `outputs/benchmark/episodes/`.

