"""
Small stateful session helper the orchestrator drives explicitly (next.md
sec 4) so directive_engine.query() itself stays a pure function per call.
Matches next.md's pseudocode API 1:1: request_human_heading -> submit_heading
-> (if goal not found) retry, incrementing an attempt counter that's logged
for HCI ("# of retries needed").
"""

from typing import Optional

import numpy as np

from vl_direction import config
from vl_direction.client import InternVLClient
from vl_direction.directive_engine import query
from vl_direction.schemas import HeadingResponse, UncertaintyContext, VLDirectiveResult


class UncertaintySession:
    def __init__(
        self,
        episode_id: str,
        covariance_threshold: float,
        covariance_value: float,
        rover_front_reference_deg: float = config.UNCERTAINTY_ROVER_FRONT_REFERENCE_DEG,
        client: Optional[InternVLClient] = None,
    ):
        self.episode_id = episode_id
        self.covariance_threshold = covariance_threshold
        self.covariance_value = covariance_value
        self.rover_front_reference_deg = rover_front_reference_deg
        self.client = client
        self.attempt = 0

    def request_human_heading(
        self, frame: Optional[np.ndarray] = None
    ) -> VLDirectiveResult:
        frames = [frame] if frame is not None else []
        context = UncertaintyContext(
            covariance_value=self.covariance_value,
            threshold_used=self.covariance_threshold,
            rover_front_reference_deg=self.rover_front_reference_deg,
            attempt=self.attempt,
        )
        return query(
            "uncertainty", frames, context, self.episode_id, client=self.client
        )

    def submit_heading(
        self,
        angle_deg: Optional[float] = None,
        angle_range_deg: Optional[tuple] = None,
        max_units: Optional[float] = None,
    ) -> VLDirectiveResult:
        heading_response = HeadingResponse(
            angle_deg=angle_deg, angle_range_deg=angle_range_deg
        )
        context = UncertaintyContext(
            covariance_value=self.covariance_value,
            threshold_used=self.covariance_threshold,
            rover_front_reference_deg=self.rover_front_reference_deg,
            human_heading_response=heading_response,
            max_units=max_units,
            attempt=self.attempt,
        )
        return query("uncertainty", [], context, self.episode_id, client=self.client)

    def retry(self, new_frame: Optional[np.ndarray] = None) -> VLDirectiveResult:
        self.attempt += 1
        return self.request_human_heading(new_frame)


if __name__ == "__main__":
    from vl_direction.client import MockInternVLClient

    session = UncertaintySession(
        episode_id="demo-episode",
        covariance_threshold=1.0,
        covariance_value=2.0,
        client=MockInternVLClient(canned_response="dunes to the left, ridge ahead"),
    )
    r1 = session.request_human_heading(np.zeros((4, 4, 3), dtype=np.uint8))
    print("request_human_heading ->", r1.uncertainty_payload)

    r2 = session.submit_heading(angle_deg=35.0)
    print("submit_heading ->", r2.uncertainty_payload)

    r3 = session.retry(np.zeros((4, 4, 3), dtype=np.uint8))
    print("retry ->", r3.uncertainty_payload, "attempt:", session.attempt)
