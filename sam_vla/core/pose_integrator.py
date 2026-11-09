"""Pure kinematic integration of Action into Pose.

Convention: heading at yaw is (-sin(yaw), -cos(yaw)) in the x-z plane --
matching habitat-sim's actual agent-forward direction (local -Z, rotated by
quaternion.from_rotation_vector([0, yaw, 0]) exactly as set_agent_pose does;
verified numerically and cross-checked against scripts/habitat_tests/
kb_teleop.py's independently-derived, known-correct forward step, which uses
this same (-sin, -cos) form). It is *not* (cos(yaw), sin(yaw)) -- that would
put "forward" 90 degrees off from where the agent's camera actually looks,
which is the root cause of the front/back-looking navigation bug this fixed
(see nav/rover_controller.py's git history). v_fwd moves along the heading
direction, v_lat moves along the rightward perpendicular. y (height) is
untouched here; it comes from terrain sampling elsewhere.
"""

import math

from sam_vla.core.types import Action, Pose


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def integrate_mars(pose: Pose, action: Action, dt: float) -> Pose:
    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)

    dx = (-action.v_fwd * sin_yaw + action.v_lat * cos_yaw) * dt
    dz = (-action.v_fwd * cos_yaw - action.v_lat * sin_yaw) * dt

    new_yaw = _wrap_to_pi(pose.yaw + action.yaw_rate * dt)

    return Pose(x=pose.x + dx, y=pose.y, z=pose.z + dz, yaw=new_yaw)


if __name__ == "__main__":

    def _show(label: str, pose: Pose, action: Action, dt: float) -> None:
        result = integrate_mars(pose, action, dt)
        print(f"{label}:")
        print(f"  in  = {pose}, action={action}, dt={dt}")
        print(f"  out = {result}")
        print()

    # 1. Pure forward motion, facing -z (yaw=0): expect dx=0, dz=-v_fwd*dt.
    _show(
        "forward, yaw=0",
        Pose(x=0.0, y=1.0, z=0.0, yaw=0.0),
        Action(v_fwd=2.0, v_lat=0.0, yaw_rate=0.0),
        dt=1.0,
    )

    # 2. Pure yaw rotation, no translation: expect x,z unchanged, yaw += pi/2.
    _show(
        "yaw rotation only",
        Pose(x=5.0, y=1.0, z=5.0, yaw=0.0),
        Action(v_fwd=0.0, v_lat=0.0, yaw_rate=math.pi / 2),
        dt=1.0,
    )

    # 3. Combined forward + lateral + yaw, facing -x (yaw=pi/2):
    #    forward should move along -x, lateral (right) should move along -z.
    _show(
        "combined, yaw=pi/2",
        Pose(x=0.0, y=1.0, z=0.0, yaw=math.pi / 2),
        Action(v_fwd=1.0, v_lat=1.0, yaw_rate=0.1),
        dt=0.5,
    )
