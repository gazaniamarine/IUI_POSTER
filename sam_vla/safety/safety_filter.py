from sam_vla.core.types import Action, Observation


def filter(action: Action, obs: Observation) -> Action:
    """
    Passthrough by default. Wraps any policy's output: deliberately independent of which policy produced the action, so it can be built out later (velocity clamping, depth-based emergency stop) without any caller needing to change.
    """
    return action


def clamp_velocity(
    action: Action,
    max_v_fwd: float = 1.0,
    max_v_lat: float = 1.0,
    max_yaw_rate: float = 1.0,
) -> Action:
    return Action(
        v_fwd=min(max(action.v_fwd, 0.0), max_v_fwd),
        v_lat=min(max(action.v_lat, -max_v_lat), max_v_lat),
        yaw_rate=min(max(action.yaw_rate, -max_yaw_rate), max_yaw_rate),
    )


if __name__ == "__main__":
    from sam_vla.core.types import Pose

    obs = Observation(
        rgb=None, depth=None, pose=Pose(x=0.0, y=0.0, z=0.0, yaw=0.0), frame_idx=0
    )
    raw_action = Action(v_fwd=1.5, v_lat=-2.0, yaw_rate=3.0)

    filtered = filter(raw_action, obs)
    clamped = clamp_velocity(raw_action)

    print(f"input:    {raw_action}")
    print(f"filter -> {filtered}")
    print(f"clamp  -> {clamped}")
