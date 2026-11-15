import os
import json


class Belief:
    def __init__(self):
        self.objects = {}
        self.held = None
        self.completed = []

    def summary(self):
        return {
            "objects": self.objects,
            "held_object": self.held,
            "completed": self.completed,
        }


def build_candidates(task, belief):
    """Deterministic task-state controller that exposes the next skill to Say."""
    text = task.lower()
    is_place_task = ("place" in text) or ("near" in text)

    if is_place_task:
        # A completed placement must transition to Done, never back to PickUp.
        if "place" in belief.completed:
            return [("Done", {})]
        if "red_cube" not in belief.objects:
            return [("Find", {"target": "red_cube"})]
        if "blue_cube" not in belief.objects:
            return [("Find", {"target": "blue_cube"})]
        if belief.held == "red_cube":
            return [(
                "PlaceNear",
                {"object": "red_cube", "target": "blue_cube", "relation": "near"},
            )]
        if belief.held is None:
            return [("PickUp", {"target": "red_cube"})]
        return []

    if any(word in text for word in ("pick", "grab", "take")):
        target = "red_cube" if "red" in text else "blue_cube"
        if target not in belief.objects:
            return [("Find", {"target": target})]
        if belief.held == target:
            return [("Done", {})]
        if belief.held is None:
            return [("PickUp", {"target": target})]
        return []

    return [("Explore", {})]


class Scorer:
    def __init__(self, kind="qwen"):
        self.kind = kind
        if kind == "qwen":
            from openai import OpenAI
            key = os.getenv("DASHSCOPE_API_KEY")
            if not key:
                raise RuntimeError("Set DASHSCOPE_API_KEY")
            self.client = OpenAI(
                api_key=key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            self.model = os.getenv("QWEN_MODEL", "qwen-plus")

    def score(self, task, calls, belief):
        if self.kind == "keyword":
            keywords = {
                "Find": ["find", "locate"],
                "PickUp": ["pick", "grab", "take"],
                "PlaceNear": ["place", "put", "near"],
                "Done": ["done", "finish", "complete"],
                "Explore": ["search", "explore"],
            }
            return {
                c["id"]: {
                    "relevance": 0.9 if any(
                        word in task.lower() for word in keywords.get(c["name"], [])
                    ) else 0.3
                }
                for c in calls
            }

        prompt = {
            "task": task,
            "belief": belief.summary(),
            "candidates": calls,
            "instruction": (
                "Return JSON keyed by candidate id with relevance from 0 to 1 "
                "and rationale. Score semantic next-step relevance only, not "
                "physical feasibility."
            ),
        }
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": json.dumps(prompt)}],
        )
        data = json.loads(response.choices[0].message.content)
        expected = {c["id"] for c in calls}
        if set(data) != expected:
            raise ValueError("Qwen candidate IDs mismatch")
        return {
            key: {"relevance": float(value["relevance"])}
            for key, value in data.items()
        }


class SayCan:
    def __init__(self, ctx, scorer):
        self.c = ctx
        self.scorer = scorer
        self.history = []

    def _can(self, call, belief):
        name = call["name"]
        args = call["args"]
        if name == "PickUp":
            return float(
                belief.held is None and args["target"] in belief.objects
            )
        if name == "PlaceNear":
            return float(
                belief.held == args["object"]
                and args["target"] in belief.objects
            )
        if name == "Done":
            if self.c.task_place:
                return float(
                    belief.held is None and "place" in belief.completed
                )
            return float(
                belief.held == getattr(self.c, "task_target", None)
            )
        return 1.0

    def run(self, task, max_rounds=12):
        for round_index in range(max_rounds):
            self.c.observe()
            belief = self.c.belief
            raw_candidates = build_candidates(task, belief)
            if not raw_candidates:
                return {
                    "success": False,
                    "status": "no_valid_next_skill",
                    "history": self.history,
                }

            candidates = [
                {"id": f"{name}:{index}", "name": name, "args": args}
                for index, (name, args) in enumerate(raw_candidates)
            ]
            say_scores = self.scorer.score(task, candidates, belief)
            ranked = []
            for candidate in candidates:
                say = float(say_scores[candidate["id"]]["relevance"])
                can = self._can(candidate, belief)
                ranked.append((say * can, say, can, candidate))
            ranked.sort(key=lambda item: item[0], reverse=True)

            if not ranked or ranked[0][2] <= 0:
                return {
                    "success": False,
                    "status": "no_feasible_candidate",
                    "history": self.history,
                }

            score, say, can, selected = ranked[0]
            result = self.c.execute(selected)
            self.history.append({
                "round": round_index + 1,
                "call": selected,
                "say": say,
                "can": can,
                "score": score,
                "belief_after": self.c.belief.summary(),
                "result": result,
            })

            if selected["name"] == "Done":
                if result["success"]:
                    return {
                        "success": True,
                        "status": "task_complete",
                        "history": self.history,
                    }
                return {
                    "success": False,
                    "status": result["status"],
                    "history": self.history,
                }

            if not result["success"]:
                return {
                    "success": False,
                    "status": result["status"],
                    "history": self.history,
                }

        return {
            "success": False,
            "status": "max_rounds",
            "history": self.history,
        }
