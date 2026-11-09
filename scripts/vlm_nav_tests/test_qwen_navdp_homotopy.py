import unittest

import numpy as np

from qwen_navdp_homotopy import VisualQwenHomotopySelector

RGB = np.zeros((16, 16, 3), dtype=np.uint8)


class VisualHomotopyTest(unittest.TestCase):
    @staticmethod
    def selector(results) -> VisualQwenHomotopySelector:
        selector = VisualQwenHomotopySelector.__new__(VisualQwenHomotopySelector)
        selector.minimum_obstacle_pixels = 2
        selector.release_clear_frames = 2
        selector.consistency_repeats = len(results)
        selector._latched_side = None
        selector._latched_confidence = 0.0
        selector._clear_frames = 0
        iterator = iter(results)
        selector._query = lambda _image: next(iterator)
        return selector

    def test_sign_convention(self) -> None:
        self.assertEqual(VisualQwenHomotopySelector.side_to_sign("LEFT"), -1.0)
        self.assertEqual(VisualQwenHomotopySelector.side_to_sign("RIGHT"), 1.0)

    def test_identical_queries_are_measured_and_latched(self) -> None:
        selector = self.selector([("LEFT", 0.9, "left") for _ in range(5)])
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[:, 7:9] = 1
        first = selector.step(RGB, mask)
        second = selector.step(RGB, mask)
        self.assertEqual(first.side, "LEFT")
        self.assertEqual(first.repeated_sides, ("LEFT",) * 5)
        self.assertEqual(first.consistency_rate, 1.0)
        self.assertFalse(first.used_fallback)
        self.assertTrue(first.queried_qwen)
        self.assertFalse(second.queried_qwen)
        self.assertEqual(second.side, "LEFT")

    def test_majority_vote_is_reported(self) -> None:
        selector = self.selector(
            [
                ("RIGHT", 0.8, "r1"),
                ("LEFT", 0.7, "l1"),
                ("RIGHT", 0.9, "r2"),
            ]
        )
        mask = np.ones((16, 16), dtype=np.uint8)
        decision = selector.step(RGB, mask)
        self.assertEqual(decision.side, "RIGHT")
        self.assertAlmostEqual(decision.consistency_rate, 2.0 / 3.0)

    def test_latch_releases_after_clear_frames(self) -> None:
        selector = self.selector([("LEFT", 0.9, "left")])
        blocked = np.ones((16, 16), dtype=np.uint8)
        clear = np.zeros_like(blocked)
        selector.step(RGB, blocked)
        held = selector.step(RGB, clear)
        released = selector.step(RGB, clear)
        self.assertEqual(held.side, "LEFT")
        self.assertEqual(released.side, "AUTO")
        self.assertEqual(released.circulation_sign, 0.0)

    def test_return_command_prompt_has_a_strict_two_command_contract(self) -> None:
        prompt = VisualQwenHomotopySelector.command_prompt("please come back")
        self.assertIn('"command":"RETURN|STOP"', prompt)
        self.assertIn("please come back", prompt)
        self.assertIn("do not create", prompt.lower())

    def test_empty_return_command_is_rejected_before_inference(self) -> None:
        selector = VisualQwenHomotopySelector.__new__(VisualQwenHomotopySelector)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            selector.classify_command(RGB, "   ")

    def test_freeform_mission_prompt_defines_round_trip(self) -> None:
        prompt = VisualQwenHomotopySelector.mission_prompt(
            "check the marker, then report back"
        )
        self.assertIn('["GO_TO_GOAL", "RETURN_HOME"]', prompt)
        self.assertIn("controller", prompt.lower())
        self.assertIn("report back", prompt)

    def test_round_trip_mission_payload_is_accepted(self) -> None:
        decision = VisualQwenHomotopySelector.parse_mission_payload(
            {
                "plan": ["GO_TO_GOAL", "RETURN_HOME"],
                "confidence": 0.93,
            },
            "raw",
        )
        self.assertEqual(decision.plan, ("GO_TO_GOAL", "RETURN_HOME"))
        self.assertAlmostEqual(decision.confidence, 0.93)

    def test_invalid_mission_steps_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be GO_TO_GOAL"):
            VisualQwenHomotopySelector.parse_mission_payload(
                {"plan": ["TURN_LEFT", "RETURN_HOME"]},
                "raw",
            )


if __name__ == "__main__":
    unittest.main()
