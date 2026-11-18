"""Run the JSONL tabletop benchmark and report task-level success rates."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config
from context import Context
from mujoco_env import Runtime, build_model
from saycan import SayCan, Scorer


def load_dataset(path):
    episodes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        episode = json.loads(line)
        required = {"id", "task", "language", "category", "layout", "expected_skills"}
        missing = required - set(episode)
        if missing:
            raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
        episodes.append(episode)
    ids = [episode["id"] for episode in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset episode IDs must be unique")
    return episodes


def grouped_metrics(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(bool(row["success"]))
    return {
        name: {
            "episodes": len(values),
            "passed": sum(values),
            "success_rate": sum(values) / len(values),
        }
        for name, values in sorted(groups.items())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=str(ROOT / "benchmarks" / "saycan_tabletop_v1.jsonl"),
    )
    parser.add_argument("--scorer", choices=["keyword", "qwen"], default="keyword")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "benchmark"))
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--max-rounds", type=int, default=12)
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    episode_output = output / "episodes"
    episode_output.mkdir(parents=True, exist_ok=True)
    episodes = load_dataset(dataset_path)
    if args.episode_id:
        requested = set(args.episode_id)
        episodes = [episode for episode in episodes if episode["id"] in requested]
        missing = requested - {episode["id"] for episode in episodes}
        if missing:
            raise SystemExit(f"Unknown episode IDs: {sorted(missing)}")
    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]
    if not episodes:
        raise SystemExit("No benchmark episodes selected")

    cfg = Config()
    scorer = Scorer(args.scorer)
    rows = []
    started = time.perf_counter()
    for index, episode in enumerate(episodes, 1):
        runtime = None
        episode_started = time.perf_counter()
        try:
            runtime = Runtime(
                build_model(cfg.resolve(), episode["layout"]),
                cfg.width,
                cfg.height,
            )
            context = Context(runtime, episode["task"])
            context.robot.home()
            context.observe()
            result = SayCan(context, scorer).run(
                episode["task"], max_rounds=args.max_rounds
            )
            actual = [step["call"]["name"] for step in result.get("history", []) if "call" in step]
            sequence_match = actual == episode["expected_skills"]
            success = bool(result.get("success")) and sequence_match
            status = result.get("status", "unknown")
            error = "" if sequence_match else "unexpected_skill_sequence"
            trace = {
                "episode": episode,
                "success": success,
                "sequence_match": sequence_match,
                "actual_skills": actual,
                "result": result,
            }
        except Exception as exc:
            success = False
            status = "exception"
            error = f"{type(exc).__name__}: {exc}"
            actual = []
            trace = {"episode": episode, "success": False, "error": error}
        finally:
            if runtime is not None:
                runtime.close()

        trace_path = episode_output / f"{episode['id']}.json"
        trace_path.write_text(
            json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        row = {
            "id": episode["id"],
            "language": episode["language"],
            "category": episode["category"],
            "task": episode["task"],
            "success": success,
            "status": status,
            "expected_skills": " > ".join(episode["expected_skills"]),
            "actual_skills": " > ".join(actual),
            "error": error,
            "elapsed_s": round(time.perf_counter() - episode_started, 3),
            "trace": str(trace_path),
        }
        rows.append(row)
        mark = "PASS" if success else "FAIL"
        print(f"[{index:02d}/{len(episodes):02d}] {mark} {episode['id']}: {status}")

    passed = sum(bool(row["success"]) for row in rows)
    summary = {
        "dataset": str(dataset_path),
        "scorer": args.scorer,
        "episodes": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "success_rate": passed / len(rows),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "by_language": grouped_metrics(rows, "language"),
        "by_category": grouped_metrics(rows, "category"),
        "qwen_request_count": scorer.request_count,
        "results": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output / "results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2, ensure_ascii=False))
    print(f"\nBenchmark reports: {output}")
    if passed != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

