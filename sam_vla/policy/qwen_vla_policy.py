from sam_vla.core.types import Action, GoalSpec, Observation
from sam_vla.policy.base_policy import NavigationPolicy
from sam_vla.vlm import qwen_client


class QwenVlaPolicy:  # implements NavigationPolicy
    def __init__(self):
        pass

    def act(self, obs: Observation, goal_spec: GoalSpec) -> Action:
        return qwen_client.drive_action(obs.rgb, goal_spec, obs.frame_idx)

    def act_verbose(self, obs: Observation, goal_spec: GoalSpec) -> tuple[Action, dict]:
        """Same as act, but also returns the raw VLA result dict for logging."""
        return qwen_client.drive_action_verbose(obs.rgb, goal_spec, obs.frame_idx)


if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    from sam_vla.core.types import Pose

    rgb = np.array(Image.open("marsyard2022_terrain_texture.png").convert("RGB"))
    obs = Observation(
        rgb=rgb, depth=None, pose=Pose(x=0.0, y=0.0, z=0.0, yaw=0.0), frame_idx=0
    )
    goal_spec = GoalSpec(
        goal_bbox_norm=(0.4, 0.4, 0.6, 0.6),
        obstacle_bboxes_norm=[(0.1, 0.1, 0.3, 0.3)],
        instruction_text="Navigate to the rock target while avoiding obstacles.",
    )

    policy = QwenVlaPolicy()
    action = policy.act(obs, goal_spec)
    print("act ->", action)
