import json
import unittest
from types import SimpleNamespace

from saycan import Belief, Scorer


CALLS = [
    {"id": "PickUp:0", "name": "PickUp", "args": {"target": "red_cube"}},
    {"id": "Done:1", "name": "Done", "args": {}},
]


class FakeCompletions:
    def __init__(self, payload):
        self.payload = payload
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        message = SimpleNamespace(content=self.payload)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, payload):
        self.chat = SimpleNamespace(completions=FakeCompletions(payload))


class QwenScorerTests(unittest.TestCase):
    def test_parses_complete_candidate_mapping(self):
        payload = json.dumps({
            "PickUp:0": {"relevance": 0.97, "rationale": "requested object"},
            "Done:1": {"relevance": 0.03, "rationale": "not complete"},
        })
        client = FakeClient(payload)
        scorer = Scorer("qwen", client=client, model="qwen-test")
        scores = scorer.score("pick up the red cube", CALLS, Belief())
        self.assertEqual(set(scores), {"PickUp:0", "Done:1"})
        self.assertAlmostEqual(scores["PickUp:0"]["relevance"], 0.97)
        self.assertEqual(scorer.request_count, 1)
        self.assertEqual(
            client.chat.completions.last_request["response_format"],
            {"type": "json_object"},
        )

    def test_rejects_missing_candidate(self):
        client = FakeClient(json.dumps({
            "PickUp:0": {"relevance": 0.9, "rationale": "requested"},
        }))
        scorer = Scorer("qwen", client=client)
        with self.assertRaisesRegex(ValueError, "candidate IDs mismatch"):
            scorer.score("pick up the red cube", CALLS, Belief())

    def test_rejects_out_of_range_score(self):
        client = FakeClient(json.dumps({
            "PickUp:0": {"relevance": 1.2},
            "Done:1": {"relevance": 0.0},
        }))
        scorer = Scorer("qwen", client=client)
        with self.assertRaisesRegex(ValueError, "must be finite"):
            scorer.score("pick up the red cube", CALLS, Belief())


if __name__ == "__main__":
    unittest.main()

