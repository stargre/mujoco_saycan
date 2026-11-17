import unittest

from saycan import Belief, Scorer, build_candidates


TASK = "Pick up the red cube and place it near the blue cube"


class PlannerStateTests(unittest.TestCase):
    def setUp(self):
        self.belief = Belief()
        self.belief.objects = {
            "red_cube": {
                "position": [0.4, -0.1, 0.4],
                "confidence": 1.0,
            },
            "blue_cube": {
                "position": [0.55, 0.1, 0.4],
                "confidence": 1.0,
            },
        }

    def test_empty_gripper_generates_multiple_pickup_candidates(self):
        candidates = build_candidates(TASK, self.belief)
        self.assertEqual(candidates[0], ("PickUp", {"target": "red_cube"}))
        self.assertIn(("PickUp", {"target": "blue_cube"}), candidates)
        self.assertIn(("Done", {}), candidates)
        self.assertGreaterEqual(len(candidates), 3)

    def test_holding_target_offers_place_and_infeasible_alternatives(self):
        self.belief.held = "red_cube"
        candidates = build_candidates(TASK, self.belief)
        self.assertEqual(candidates[0][0], "PlaceNear")
        self.assertIn(("Done", {}), candidates)
        self.assertIn(("PickUp", {"target": "blue_cube"}), candidates)

    def test_completed_place_selects_done_first(self):
        self.belief.completed.append("place")
        candidates = build_candidates(TASK, self.belief)
        self.assertEqual(candidates[0][0], "Done")
        self.assertIn(("PickUp", {"target": "red_cube"}), candidates)

    def test_belief_summary_is_a_deep_snapshot(self):
        snapshot = self.belief.summary()
        self.belief.completed.append("pick")
        self.belief.objects["red_cube"]["position"][0] = 9.9
        self.assertEqual(snapshot["completed"], [])
        self.assertEqual(snapshot["objects"]["red_cube"]["position"][0], 0.4)

    def test_keyword_scorer_distinguishes_requested_object_from_distractor(self):
        calls = [
            {"id": "PickUp:0", "name": "PickUp", "args": {"target": "red_cube"}},
            {"id": "PickUp:1", "name": "PickUp", "args": {"target": "blue_cube"}},
            {"id": "Done:2", "name": "Done", "args": {}},
        ]
        scores = Scorer("keyword").score(TASK, calls, self.belief)
        self.assertGreater(
            scores["PickUp:0"]["relevance"],
            scores["PickUp:1"]["relevance"],
        )

    def test_chinese_place_task_generates_same_grounded_plan(self):
        candidates = build_candidates(
            "把红色方块拿起来，放到蓝色方块旁边", self.belief
        )
        self.assertEqual(candidates[0], ("PickUp", {"target": "red_cube"}))
        self.assertIn(("PickUp", {"target": "blue_cube"}), candidates)



if __name__ == "__main__":
    unittest.main()
