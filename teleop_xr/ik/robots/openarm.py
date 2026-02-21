# pyright: reportCallIssue=false
import math
import os
import sys
from typing import Any

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


import jax
import jax.numpy as jnp
import jaxlie
import pyroki as pk
import yourdfpy

from teleop_xr.ik.robot import BaseRobot, Cost
from teleop_xr import ram


def _make_link0_transform(rpy_x: float, y_offset: float) -> jaxlie.SE3:
    """Build T_linkX_in_world from the URDF bimanual mount parameters."""
    return jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_rpy_radians(rpy_x, 0.0, 0.0),
        jnp.array([0.0, y_offset, 0.698]),
    )


class OpenArmRobot(BaseRobot):
    """
    OpenArm bimanual robot implementation for IK.
    Uses the openarm_description package with bimanual=true.

    Absolute pose mode
    ------------------
    Both arms are mounted on the body with a ±π/2 roll:
      - left  link0: xyz=[0, +0.031, 0.698]  rpy=[-π/2, 0, 0]
      - right link0: xyz=[0, -0.031, 0.698]  rpy=[+π/2, 0, 0]

    The Quest shoulder joints and XR controller poses reach IKController already
    converted to FLU (Forward-Left-Up / ROS) frame by the server. Shoulder-relative
    controller motion is mapped to robot-world targets via ``link0_transforms``.

    ``R_ALIGN`` maps the controller grip frame to the robot EE (link7) frame. It
    defaults to identity, which is a reasonable starting point. Tune it empirically
    once with hardware: hold the controller at a reference pose (arm forward, palm
    facing inward) and adjust until link7 tracks correctly.
    """

    # Fixed transforms from URDF world frame to each arm's base link (link0).
    # Source: v10.urdf.xacro bimanual args for left/right_arm_base_xyz/rpy.
    _T_L_LINK0_IN_WORLD: jaxlie.SE3 = _make_link0_transform(-math.pi / 2, 0.031)
    _T_R_LINK0_IN_WORLD: jaxlie.SE3 = _make_link0_transform(math.pi / 2, -0.031)

    # Grip-frame → EE (link7) frame alignment for absolute pose mode.
    #
    # The OpenArm arms extend along the link0 Z-axis (which points sideways in
    # world frame due to the ±π/2 roll mount). The XR controller grip frame in
    # FLU has X=forward. Ry(+π/2) maps controller-X → link0-Z so that holding
    # the controller straight forward targets the arm pointing outward, matching
    # the robot's natural resting pose.
    _R_ALIGN: jaxlie.SO3 = jaxlie.SO3.from_rpy_radians(0.0, math.pi / 2, 0.0)

    def __init__(self, urdf_string: str | None = None, **kwargs: Any) -> None:
        super().__init__()
        urdf = self._load_urdf(urdf_string)

        self.robot: pk.Robot = pk.Robot.from_urdf(urdf)
        self.robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

        # End effector links for bimanual setup
        self.L_ee: str = "openarm_left_link7"
        self.R_ee: str = "openarm_right_link7"

        if self.L_ee in self.robot.links.names:
            self.L_ee_link_idx: int = self.robot.links.names.index(self.L_ee)
        else:
            raise ValueError(f"Link {self.L_ee} not found in URDF")

        if self.R_ee in self.robot.links.names:
            self.R_ee_link_idx: int = self.robot.links.names.index(self.R_ee)
        else:
            raise ValueError(f"Link {self.R_ee} not found in URDF")

    def _load_default_urdf(self) -> yourdfpy.URDF:
        repo_url = "https://github.com/enactic/openarm_description.git"
        xacro_path = "urdf/robot/v10.urdf.xacro"
        xacro_args = {
            "bimanual": "true",
            "hand": "true",
            "ros2_control": "false",
        }

        self.urdf_path = str(
            ram.get_resource(
                repo_url=repo_url,
                path_inside_repo=xacro_path,
                xacro_args=xacro_args,
                resolve_packages=True,
                convert_dae_to_glb=True,
            )
        )

        repo_path = ram.get_repo(repo_url)
        self.mesh_path = str(repo_path)

        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"OpenArm URDF not found at {self.urdf_path}")

        return yourdfpy.URDF.load(self.urdf_path)

    @property
    @override
    def model_scale(self) -> float:
        return 1.0

    @property
    @override
    def supported_frames(self) -> set[str]:
        return {"left", "right"}

    @property
    @override
    def link0_transforms(self) -> dict[str, jaxlie.SE3]:
        return {
            "left": self._T_L_LINK0_IN_WORLD,
            "right": self._T_R_LINK0_IN_WORLD,
        }

    @property
    @override
    def R_align(self) -> jaxlie.SO3:
        return self._R_ALIGN

    @property
    @override
    def joint_var_cls(self) -> Any:
        return self.robot.joint_var_cls

    @property
    @override
    def actuated_joint_names(self) -> list[str]:
        return list(self.robot.joints.actuated_names)

    @override
    def forward_kinematics(self, config: jax.Array) -> dict[str, jaxlie.SE3]:
        fk = self.robot.forward_kinematics(config)
        return {
            "left": jaxlie.SE3(fk[self.L_ee_link_idx]),
            "right": jaxlie.SE3(fk[self.R_ee_link_idx]),
        }

    @override
    def get_default_config(self) -> jax.Array:
        joint_names = self.actuated_joint_names
        jnp.zeros(len(joint_names))

        default_pose = {
            "openarm_left_joint1": 0.0,
            "openarm_left_joint2": 0.0,
            "openarm_left_joint3": 0.0,
            "openarm_left_joint4": 0.0,
            "openarm_left_joint5": 0.0,
            "openarm_left_joint6": 0.0,
            "openarm_left_joint7": 0.0,
            "openarm_right_joint1": 0.0,
            "openarm_right_joint2": 0.0,
            "openarm_right_joint3": 0.0,
            "openarm_right_joint4": 0.0,
            "openarm_right_joint5": 0.0,
            "openarm_right_joint6": 0.0,
            "openarm_right_joint7": 0.0,
            "openarm_left_finger_joint1": 0.0,
            "openarm_left_finger_joint2": 0.0,
            "openarm_right_finger_joint1": 0.0,
            "openarm_right_finger_joint2": 0.0,
        }

        config_list = []
        for name in joint_names:
            config_list.append(default_pose.get(name, 0.0))

        return jnp.array(config_list)

    @override
    def build_costs(
        self,
        target_L: jaxlie.SE3 | None,
        target_R: jaxlie.SE3 | None,
        target_Head: jaxlie.SE3 | None,
        q_current: jnp.ndarray | None = None,
    ) -> list[Cost]:
        costs = []
        JointVar = self.robot.joint_var_cls

        if q_current is not None:
            costs.append(
                pk.costs.rest_cost(
                    JointVar(0),
                    rest_pose=q_current,
                    weight=5.0,
                )
            )

        costs.append(
            pk.costs.manipulability_cost(
                self.robot,
                JointVar(0),
                jnp.array([self.L_ee_link_idx, self.R_ee_link_idx], dtype=jnp.int32),
                weight=0.01,
            )
        )

        if target_L is not None:
            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    self.robot,
                    JointVar(0),
                    target_L,
                    jnp.array(self.L_ee_link_idx, dtype=jnp.int32),
                    pos_weight=50.0,
                    ori_weight=10.0,
                )
            )

        if target_R is not None:
            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    self.robot,
                    JointVar(0),
                    target_R,
                    jnp.array(self.R_ee_link_idx, dtype=jnp.int32),
                    pos_weight=50.0,
                    ori_weight=10.0,
                )
            )

        costs.append(pk.costs.limit_cost(self.robot, JointVar(0), weight=100.0))

        costs.append(
            pk.costs.self_collision_cost(
                self.robot,
                self.robot_coll,
                JointVar(0),
                margin=0.05,
                weight=10.0,
            )
        )

        return costs
