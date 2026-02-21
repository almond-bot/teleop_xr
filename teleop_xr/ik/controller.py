import jax.numpy as jnp
import jaxlie
import numpy as np
from loguru import logger
from typing import Literal
from teleop_xr.utils.filter import WeightedMovingFilter
from teleop_xr.messages import XRState, XRDeviceRole, XRHandedness, XRPose
from teleop_xr.ik.robot import BaseRobot
from teleop_xr.ik.solver import PyrokiSolver


class IKController:
    """
    High-level controller for teleoperation using IK.

    Supports two pose modes selectable at construction time:

    ``"relative"`` (default)
        On deadman engagement a snapshot is taken of the XR controller poses and
        robot FK poses.  Each frame the EE target is offset from the snapshot FK
        pose by the controller's delta motion:
            target = robot_FK_init + transform(xr_now - xr_init)

    ``"absolute"``
        EE targets are derived directly from the live controller pose relative to
        the operator's shoulder (supplied via ``XRState.body_joints``).  Each arm
        is mapped through the robot's corresponding ``link0`` frame so the
        shoulder-relative workspace maps to the robot's arm workspace:
            T_ctrl_in_shoulder = T_shoulder.inverse() @ T_ctrl
            target = T_link0_in_world @ R_align @ T_ctrl_in_shoulder

        Requires the robot to implement ``link0_transforms``; falls back to
        relative if shoulder poses are missing from the state.

    In both modes the deadman switch (both grips squeezed) must be held.
    """

    robot: BaseRobot
    solver: PyrokiSolver | None
    mode: Literal["relative", "absolute"]
    active: bool
    snapshot_xr: dict[str, jaxlie.SE3]
    snapshot_robot: dict[str, jaxlie.SE3]
    filter: WeightedMovingFilter | None
    _warned_unsupported: set[str]

    def __init__(
        self,
        robot: BaseRobot,
        solver: PyrokiSolver | None = None,
        filter_weights: np.ndarray | None = None,
        mode: Literal["relative", "absolute"] = "relative",
    ):
        """
        Initialize the IK controller.

        Args:
            robot: The robot model.
            solver: The IK solver. If None, step() will return current_config.
            filter_weights: Optional weights for a WeightedMovingFilter on joint outputs.
            mode: ``"relative"`` for delta-based tracking (default) or ``"absolute"``
                for shoulder-relative absolute pose tracking.
        """
        self.robot = robot
        self.solver = solver
        self.mode = mode
        self.active = False
        self._warned_unsupported = set()

        self.snapshot_xr = {}
        self.snapshot_robot = {}

        self.filter = None
        if filter_weights is not None:
            default_config = self.robot.get_default_config()
            self.filter = WeightedMovingFilter(
                filter_weights, data_size=len(default_config)
            )

    def xr_pose_to_se3(self, pose: XRPose) -> jaxlie.SE3:
        """
        Convert an XRPose to a jaxlie SE3 object.

        Args:
            pose: The XR pose to convert.

        Returns:
            jaxlie.SE3: The converted pose.
        """
        translation = jnp.array(
            [pose.position["x"], pose.position["y"], pose.position["z"]]
        )
        rotation = jaxlie.SO3(
            wxyz=jnp.array(
                [
                    pose.orientation["w"],
                    pose.orientation["x"],
                    pose.orientation["y"],
                    pose.orientation["z"],
                ]
            )
        )
        return jaxlie.SE3.from_rotation_and_translation(rotation, translation)

    def compute_teleop_transform(
        self, t_ctrl_curr: jaxlie.SE3, t_ctrl_init: jaxlie.SE3, t_ee_init: jaxlie.SE3
    ) -> jaxlie.SE3:
        """
        Compute the target robot pose based on XR controller motion (relative mode).

        Args:
            t_ctrl_curr: Current XR controller pose.
            t_ctrl_init: XR controller pose at the start of teleoperation.
            t_ee_init: Robot end-effector pose at the start of teleoperation.

        Returns:
            jaxlie.SE3: The calculated target pose for the robot end-effector.
        """
        t_delta_ros = t_ctrl_curr.translation() - t_ctrl_init.translation()
        t_delta_robot = self.robot.ros_to_base @ t_delta_ros

        q_delta_ros = t_ctrl_curr.rotation() @ t_ctrl_init.rotation().inverse()
        q_delta_robot = self.robot.ros_to_base @ q_delta_ros @ self.robot.base_to_ros

        t_new = t_ee_init.translation() + t_delta_robot
        q_new = q_delta_robot @ t_ee_init.rotation()

        return jaxlie.SE3.from_rotation_and_translation(q_new, t_new)

    def compute_absolute_target(
        self,
        T_ctrl: jaxlie.SE3,
        T_shoulder: jaxlie.SE3,
        T_link0_in_world: jaxlie.SE3,
    ) -> jaxlie.SE3:
        """
        Compute an absolute IK target in robot world frame (absolute mode).

        Maps the controller's pose relative to the operator's shoulder into the
        robot's arm workspace by treating the arm's link0 as the matching shoulder
        reference. An optional ``R_align`` rotation from the robot corrects any
        constant offset between the XR grip frame and the robot EE (link7) frame.

        Args:
            T_ctrl: Controller grip pose in FLU world frame.
            T_shoulder: Shoulder joint pose in FLU world frame (from Quest body tracking).
            T_link0_in_world: Fixed SE3 of the arm's link0 in the URDF world frame.

        Returns:
            jaxlie.SE3: IK target in robot URDF world frame.
        """
        # World-frame offset from shoulder to controller (strip shoulder orientation —
        # the Quest body-tracking joint frame does not align with world FLU and would
        # corrupt the translation if we used T_shoulder.inverse() @ T_ctrl directly).
        offset_world = T_ctrl.translation() - T_shoulder.translation()

        # Controller orientation relative to shoulder in world FLU frame.
        ctrl_rot_in_shoulder = T_shoulder.rotation().inverse() @ T_ctrl.rotation()

        R_align = self.robot.R_align

        # Translation: add world-frame offset directly to link0 position.
        # Do NOT compose via SE3 multiplication — that would apply link0's Rx(±π/2)
        # to the FLU offset and swap Y/Z axes.
        target_translation = T_link0_in_world.translation() + offset_world

        # Rotation: link0 frame @ alignment correction @ controller orientation.
        target_rotation = T_link0_in_world.rotation() @ R_align @ ctrl_rot_in_shoulder

        return jaxlie.SE3.from_rotation_and_translation(target_rotation, target_translation)

    def _get_device_poses(self, state: XRState) -> dict[str, jaxlie.SE3]:
        """
        Extract controller and head poses from the current XR state.
        """
        poses = {}
        supported = self.robot.supported_frames
        for device in state.devices:
            frame_name = None
            pose_data = None
            if device.role == XRDeviceRole.CONTROLLER:
                if device.handedness == XRHandedness.LEFT and device.gripPose:
                    frame_name = "left"
                    pose_data = device.gripPose
                elif device.handedness == XRHandedness.RIGHT and device.gripPose:
                    frame_name = "right"
                    pose_data = device.gripPose
            elif device.role == XRDeviceRole.HEAD and device.pose:
                frame_name = "head"
                pose_data = device.pose

            if frame_name and pose_data is not None:
                if frame_name in supported:
                    poses[frame_name] = self.xr_pose_to_se3(pose_data)
                elif frame_name not in self._warned_unsupported:
                    logger.warning(
                        f"[IKController] Warning: Frame '{frame_name}' is available in XRState but not supported by robot. Skipping."
                    )
                    self._warned_unsupported.add(frame_name)
        return poses

    def _get_shoulder_poses(self, state: XRState) -> dict[str, jaxlie.SE3]:
        """
        Extract left/right shoulder poses from Quest body tracking joints.

        Returns an empty dict if body_joints is absent from the state, allowing
        the caller to fall back to relative mode gracefully.
        """
        if state.body_joints is None:
            return {}
        poses = {}
        for side in ("left", "right"):
            pose_data = state.body_joints.get(f"{side}-shoulder")
            if pose_data is not None:
                poses[side] = self.xr_pose_to_se3(pose_data)
        return poses

    def _check_deadman(self, state: XRState) -> bool:
        """
        Check if the deadman switch (grip button) is engaged on both controllers.
        """
        left_squeezed = False
        right_squeezed = False
        for device in state.devices:
            if device.role == XRDeviceRole.CONTROLLER:
                is_squeezed = (
                    device.gamepad is not None
                    and len(device.gamepad.buttons) > 1
                    and device.gamepad.buttons[1].pressed
                )
                if device.handedness == XRHandedness.LEFT:
                    left_squeezed = is_squeezed
                elif device.handedness == XRHandedness.RIGHT:
                    right_squeezed = is_squeezed
        return left_squeezed and right_squeezed

    def reset(self) -> None:
        """
        Resets the controller state, forcing it to re-take snapshots on the next step.
        """
        self.active = False
        self.snapshot_xr = {}
        self.snapshot_robot = {}
        if self.filter is not None:
            self.filter.reset()
        logger.info("[IKController] Reset triggered")

    def set_mode(self, mode: Literal["relative", "absolute"]) -> None:
        """
        Switch between relative and absolute tracking modes.

        Always resets engagement state so the next deadman press re-initialises
        cleanly regardless of what the previous mode was doing.  Safe to call
        at any time, including mid-session.

        Args:
            mode: ``"relative"`` for delta-based tracking or ``"absolute"`` for
                shoulder-relative absolute pose tracking.
        """
        if mode == self.mode:
            return
        self.reset()
        self.mode = mode
        logger.info(f"[IKController] Mode switched to '{mode}'")

    def _solve(
        self,
        target_L: jaxlie.SE3 | None,
        target_R: jaxlie.SE3 | None,
        target_Head: jaxlie.SE3 | None,
        q_current: np.ndarray,
    ) -> np.ndarray:
        """Run the IK solver and apply the output filter if configured."""
        if self.solver is None:
            return q_current

        new_config_jax = self.solver.solve(
            target_L,
            target_R,
            target_Head,
            jnp.asarray(q_current),
        )
        new_config = np.array(new_config_jax)

        if self.filter is not None:
            self.filter.add_data(new_config)
            if self.filter.data_ready():
                return self.filter.filtered_data

        logger.debug(f"[IKController] New Config: {new_config}")
        return new_config

    def step(self, state: XRState, q_current: np.ndarray) -> np.ndarray:
        """
        Execute one control step: update targets and solve for new joint configuration.

        Routes to relative or absolute mode based on ``self.mode``.  Absolute mode
        requires shoulder poses in ``state.body_joints``; if they are missing it
        logs a warning and falls back to relative mode for that step.

        Args:
            state: The current XR state from the headset.
            q_current: The current joint configuration of the robot.

        Returns:
            np.ndarray: The new (possibly filtered) joint configuration.
        """
        if self.mode == "absolute":
            return self._step_absolute(state, q_current)
        return self._step_relative(state, q_current)

    def _step_relative(self, state: XRState, q_current: np.ndarray) -> np.ndarray:
        """Execute one relative-mode control step."""
        is_deadman_active = self._check_deadman(state)
        curr_xr_poses = self._get_device_poses(state)

        required_keys = self.robot.supported_frames
        has_all_poses = all(k in curr_xr_poses for k in required_keys)

        if is_deadman_active and has_all_poses:
            if not self.active:
                self.active = True
                self.snapshot_xr = curr_xr_poses

                fk_poses = self.robot.forward_kinematics(jnp.asarray(q_current))
                self.snapshot_robot = {k: fk_poses[k] for k in required_keys}

                logger.info(f"[IKController] Relative engaged. FK: {self.snapshot_robot}")
                return q_current

            target_L: jaxlie.SE3 | None = None
            target_R: jaxlie.SE3 | None = None
            target_Head: jaxlie.SE3 | None = None

            if "left" in required_keys:
                target_L = self.compute_teleop_transform(
                    curr_xr_poses["left"],
                    self.snapshot_xr["left"],
                    self.snapshot_robot["left"],
                )
            if "right" in required_keys:
                target_R = self.compute_teleop_transform(
                    curr_xr_poses["right"],
                    self.snapshot_xr["right"],
                    self.snapshot_robot["right"],
                )
            if "head" in required_keys:
                target_Head = self.compute_teleop_transform(
                    curr_xr_poses["head"],
                    self.snapshot_xr["head"],
                    self.snapshot_robot["head"],
                )

            return self._solve(target_L, target_R, target_Head, q_current)

        else:
            if self.active:
                self.active = False
                if self.filter is not None:
                    self.filter.reset()
            return q_current

    def _step_absolute(self, state: XRState, q_current: np.ndarray) -> np.ndarray:
        """Execute one absolute-mode control step."""
        is_deadman_active = self._check_deadman(state)
        curr_xr_poses = self._get_device_poses(state)
        shoulder_poses = self._get_shoulder_poses(state)

        required_keys = self.robot.supported_frames
        has_all_poses = all(k in curr_xr_poses for k in required_keys)

        # Fall back to relative if shoulder data is missing
        has_all_shoulders = all(k in shoulder_poses for k in required_keys if k in ("left", "right"))
        if not has_all_shoulders:
            if is_deadman_active and has_all_poses:
                logger.warning(
                    "[IKController] Absolute mode: shoulder poses missing from state.body_joints. "
                    "Falling back to relative mode for this step."
                )
            return self._step_relative(state, q_current)

        link0_tfs = self.robot.link0_transforms

        if is_deadman_active and has_all_poses:
            if not self.active:
                self.active = True
                logger.info("[IKController] Absolute mode engaged.")
                return q_current

            target_L: jaxlie.SE3 | None = None
            target_R: jaxlie.SE3 | None = None
            target_Head: jaxlie.SE3 | None = None

            if "left" in required_keys and "left" in link0_tfs:
                target_L = self.compute_absolute_target(
                    curr_xr_poses["left"],
                    shoulder_poses["left"],
                    link0_tfs["left"],
                )
            if "right" in required_keys and "right" in link0_tfs:
                target_R = self.compute_absolute_target(
                    curr_xr_poses["right"],
                    shoulder_poses["right"],
                    link0_tfs["right"],
                )
            if "head" in required_keys:
                # Head tracking has no shoulder analogue; use relative mode for head
                if self.active and "head" in self.snapshot_xr:
                    target_Head = self.compute_teleop_transform(
                        curr_xr_poses["head"],
                        self.snapshot_xr["head"],
                        self.snapshot_robot.get(
                            "head",
                            self.robot.forward_kinematics(jnp.asarray(q_current))["head"],
                        ),
                    )

            return self._solve(target_L, target_R, target_Head, q_current)

        else:
            if self.active:
                self.active = False
                self.snapshot_xr = {}
                self.snapshot_robot = {}
                if self.filter is not None:
                    self.filter.reset()
            return q_current
