import unittest
from types import SimpleNamespace

from saycan import Belief, build_candidates


class PlannerStateTests(unittest.TestCase):
    def setUp(self):
        self.belief = Belief()
        self.belief.objects = {
            "red_cube": {"position": [0.4, -0.1, 0.4]},
            "blue_cube": {"position": [0.55, 0.1, 0.4]},
        }

    def test_empty_gripper_selects_pickup(self):
        self.assertEqual(
            build_candidates("pick up red and place near blue", self.belief)[0][0],
            "PickUp",
        )

    def test_holding_target_selects_place(self):
        self.belief.held = "red_cube"
        self.assertEqual(
            build_candidates("pick up red and place near blue", self.belief)[0][0],
            "PlaceNear",
        )

    def test_completed_place_selects_done_not_pickup(self):
        self.belief.completed.append("place")
        self.assertEqual(
            build_candidates("pick up red and place near blue", self.belief)[0][0],
            "Done",
        )


if __name__ == "__main__":
    unittest.main()
