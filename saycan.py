import copy
import json
import math
import os

import numpy as np


class Belief:
    def __init__(self):
        self.objects = {}
        self.held = None
        self.completed = []

    def summary(self):
        """Return an immutable-in-practice snapshot for experiment history."""
        return copy.deepcopy({
            "objects": self.objects,
            "held_object": self.held,
            "completed": self.completed,
        })


def _task_entities(task):
    """Extract the manipulated object and placement reference for this scene."""
    text = task.lower()
    has_red = "red" in text
    has_blue = "blue" in text

    if has_red and has_blue:
        return "red_cube", "blue_cube"
    if has_red:
        return "red_cube", "blue_cube" if ("place" in text or "near" in text) else None
    if has_blue:
        return "blue_cube", "red_cube" if ("place" in text or "near" in text) else None
    if "place" in text or "near" in text:
        return "red_cube", "blue_cube"
    return None, None


def build_candidates(task, belief):
    """Generate a small, state-grounded candidate set for SayCan.

    The candidate set deliberately includes task-relevant alternatives and a
    Done candidate. Say ranks semantic relevance; Can independently filters
    candidates using gripper state, observations and IK reachability.
    """
    text = task.lower()
    is_place_task = ("place" in text) or ("near" in text)
    source, reference = _task_entities(task)

    if source is None:
        return [("Explore", {})]

    if is_place_task:
        # Missing-object search is represented explicitly. The current Find
        # primitive only refreshes the camera observation; it is not yet an
        # active search policy, so Can will not claim it is physically feasible.
        if source not in belief.objects:
            return [("Find", {"target": source}), ("Done", {})]
        if reference not in belief.objects:
            return [("Find", {"target": reference}), ("Done", {})]

        if "place" in belief.completed:
            return [
                ("Done", {}),
                ("PickUp", {"target": source}),
                ("PickUp", {"target": reference}),
            ]

        if belief.held is None:
            return [
                ("PickUp", {"target": source}),
                ("PickUp", {"target": reference}),
                ("Done", {}),
            ]

        if belief.held == source:
            return [
                (
                    "PlaceNear",
                    {
                        "object": source,
                        "target": reference,
                        "relation": "near",
                    },
                ),
                ("Done", {}),
                ("PickUp", {"target": reference}),
            ]

        # Holding the wrong object is a recovery condition, not permission to
        # silently substitute it for the requested source object.
        return [
            ("Done", {}),
            (
                "PlaceNear",
                {
                    "object": source,
                    "target": reference,
                    "relation": "near",
                },
            ),
            ("PickUp", {"target": source}),
        ]

    # Pick-only task: keep an alternative visible object in the candidate set
    # so Say must distinguish task relevance from physical affordance.
    other = "blue_cube" if source == "red_cube" else "red_cube"
    if source not in belief.objects:
        return [("Find", {"target": source}), ("Done", {})]
    if belief.held == source:
        return [("Done", {}), ("PickUp", {"target": other})]
    if belief.held is None:
        candidates = [("PickUp", {"target": source})]
        if other in belief.objects:
            candidates.append(("PickUp", {"target": other}))
        candidates.append(("Done", {}))
        return candidates
    return [("Done", {}), ("PickUp", {"target": other})]


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
        source, reference = _task_entities(task)

        if self.kind == "keyword":
            scores = {}
            for call in calls:
                name = call["name"]
                args = call["args"]
                relevance = 0.05
                rationale = "Not directly aligned with the requested next step."

                if name == "PickUp":
                    if args.get("target") == source:
                        relevance = 0.95
                        rationale = "This is the object explicitly requested by the task."
                    else:
                        relevance = 0.08
                        rationale = "This object is visible, but it is not the requested source object."
                elif name == "PlaceNear":
                    if (
                        args.get("object") == source
                        and args.get("target") == reference
                    ):
                        relevance = 0.98
                        rationale = "This completes the requested object-to-reference relation."
                    else:
                        relevance = 0.05
                        rationale = "The placement arguments do not match the task."
                elif name == "Done":
                    complete = (
                        belief.held is None and "place" in belief.completed
                        if reference is not None
                        else belief.held == source
                    )
                    relevance = 0.95 if complete else 0.05
                    rationale = (
                        "The task postcondition is satisfied."
                        if complete else "The task postcondition is not yet satisfied."
                    )
                elif name == "Find":
                    relevance = 0.75
                    rationale = "A required object is currently absent from the belief state."
                elif name == "Explore":
                    relevance = 0.2
                    rationale = "Exploration may provide additional observations."

                scores[call["id"]] = {
                    "relevance": relevance,
                    "rationale": rationale,
                }
            return scores

        prompt = {
            "task": task,
            "belief": belief.summary(),
            "candidates": calls,
            "instruction": (
                "Score every candidate independently for semantic relevance as "
                "the NEXT action toward the user's task. Distinguish the requested "
                "source object from distractor objects. Consider the current "
                "gripper state and completed subgoals. Return a JSON object whose "
                "keys are exactly the candidate IDs. Each value must contain "
                "'relevance' (a number from 0 to 1) and a short 'rationale'. "
                "Do not estimate physical feasibility; a separate Can module "
                "does that. Do not omit any candidate."
            ),
        }
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": json.dumps(prompt)}],
        )
        data = json.loads(response.choices[0].message.content)
        expected = {candidate["id"] for candidate in calls}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError(
                f"Qwen candidate IDs mismatch: expected={sorted(expected)}, "
                f"received={sorted(data) if isinstance(data, dict) else type(data).__name__}"
            )

        parsed = {}
        for candidate_id, value in data.items():
            if not isinstance(value, dict) or "relevance" not in value:
                raise ValueError(f"Invalid Qwen score payload for {candidate_id}")
            relevance = float(value["relevance"])
            if not math.isfinite(relevance) or not 0.0 <= relevance <= 1.0:
                raise ValueError(
                    f"Qwen relevance for {candidate_id} must be finite and in [0, 1]"
                )
            parsed[candidate_id] = {
                "relevance": relevance,
                "rationale": str(value.get("rationale", "")),
            }
        return parsed


class SayCan:
    def __init__(self, ctx, scorer):
        self.c = ctx
        self.scorer = scorer
        self.history = []

    def _ik_reachable(self, position, rotation):
        try:
            self.c.robot.ik(
                np.asarray(position, dtype=float),
                target_rotation=np.asarray(rotation, dtype=float),
            )
            return True
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return False

    def _can(self, call, belief):
        """Estimate physical affordance without executing or advancing physics."""
        name = call["name"]
        args = call["args"]

        if name == "PickUp":
            target = args.get("target")
            if (
                belief.held is not None
                or "place" in belief.completed
                or target not in belief.objects
            ):
                return 0.0
            observation = belief.objects[target]
            if float(observation.get("confidence", 0.0)) < 0.1:
                return 0.0
            position = np.asarray(observation.get("position"), dtype=float)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                return 0.0
            rotation = self.c.robot.rotation()
            pregrasp = position + np.array([0.0, 0.0, 0.10])
            # Check both approach and grasp poses. IK is state-preserving.
            return float(
                self._ik_reachable(pregrasp, rotation)
                and self._ik_reachable(position, rotation)
            )

        if name == "PlaceNear":
            object_name = args.get("object")
            target_name = args.get("target")
            if (
                belief.held != object_name
                or target_name not in belief.objects
                or getattr(self.c, "held_tcp_to_object_offset_world", None) is None
            ):
                return 0.0

            model = self.c.rt.m
            geom_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, object_name + "_geom"
            )
            table_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, "table"
            )
            if geom_id < 0 or table_id < 0:
                return 0.0

            reference = np.asarray(
                belief.objects[target_name]["position"], dtype=float
            )
            if reference.shape != (3,) or not np.all(np.isfinite(reference)):
                return 0.0
            table_top = float(
                model.geom_pos[table_id][2] + model.geom_size[table_id][2]
            )
            object_half_height = float(model.geom_size[geom_id][2])
            goal_object = reference.copy()
            goal_object[0] += 0.075
            goal_object[2] = table_top + object_half_height + 0.003
            goal_tcp = goal_object + np.asarray(
                self.c.held_tcp_to_object_offset_world, dtype=float
            )
            preplace_tcp = goal_tcp + np.array([0.0, 0.0, 0.12])
            rotation = self.c.robot.rotation()
            return float(
                self._ik_reachable(preplace_tcp, rotation)
                and self._ik_reachable(goal_tcp, rotation)
            )

        if name == "Done":
            if getattr(self.c, "task_place", False):
                return float(
                    belief.held is None and "place" in belief.completed
                )
            return float(
                belief.held == getattr(self.c, "task_target", None)
            )

        # Find currently only refreshes the camera once; it does not yet
        # implement active viewpoint search. Do not report it as feasible
        # when the target is not already observable.
        if name in ("Find", "Explore"):
            return 0.0

        return 0.0

    def run(self, task, max_rounds=12):
        for round_index in range(max_rounds):
            self.c.observe()
            belief = self.c.belief
            belief_before = belief.summary()
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
            candidate_log = []
            for candidate in candidates:
                score_data = say_scores[candidate["id"]]
                say = float(score_data["relevance"])
                can = self._can(candidate, belief)
                combined = say * can
                rationale = str(score_data.get("rationale", ""))
                candidate_log.append({
                    "id": candidate["id"],
                    "name": candidate["name"],
                    "args": copy.deepcopy(candidate["args"]),
                    "say": say,
                    "can": can,
                    "score": combined,
                    "rationale": rationale,
                })
                ranked.append((combined, say, can, candidate, rationale))

            ranked.sort(key=lambda item: item[0], reverse=True)
            if not ranked or ranked[0][2] <= 0.0:
                self.history.append({
                    "round": round_index + 1,
                    "belief_before": belief_before,
                    "candidate_scores": candidate_log,
                    "selected": None,
                    "status": "no_feasible_candidate",
                    "belief_after": belief.summary(),
                })
                return {
                    "success": False,
                    "status": "no_feasible_candidate",
                    "history": self.history,
                }

            score, say, can, selected, rationale = ranked[0]
            result = self.c.execute(selected)
            self.history.append({
                "round": round_index + 1,
                "belief_before": belief_before,
                "candidate_scores": candidate_log,
                "selected_candidate_id": selected["id"],
                "call": copy.deepcopy(selected),
                "say": say,
                "can": can,
                "score": score,
                "selection_rationale": rationale,
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


# Import MuJoCo lazily so planner/scorer unit tests can run without loading
# the simulator or the Franka Menagerie model.
import mujoco
