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
        EE targets are derived from the live controller pose relative to the
        operator's shoulder position, with body yaw (from the HMD viewer pose)
        used as the horizontal reference frame:

            R_body_yaw    = yaw-only SO3 extracted from head orientation
            offset_body   = R_body_yaw⁻¹ @ (p_ctrl - p_shoulder)
            ctrl_rot_body = R_body_yaw⁻¹ @ R_ctrl
            target_t      = p_link0 + offset_body
            target_R      = ctrl_rot_body @ R_link0 @ R_align

        Using only the yaw component of the head pose means the body reference
        rotates when the operator turns their torso (correct), but nodding or
        tilting the head does not affect the arm mapping.  Walking is handled
        by the shoulder position subtraction.  Requires both shoulder poses
        (``XRState.body_joints``) and a head device pose; falls back to relative
        if either is missing.

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
        T_head: jaxlie.SE3,
        T_link0_in_world: jaxlie.SE3,
    ) -> jaxlie.SE3:
        """
        Compute an absolute IK target in robot world frame (absolute mode).

        Extracts body yaw from the HMD viewer pose and uses it as the horizontal
        reference frame, with the shoulder position as the translation origin:

            R_body_yaw    = yaw-only SO3 from head forward vector projected onto XY
            offset_body   = R_body_yaw⁻¹ @ (p_ctrl - p_shoulder)
            ctrl_rot_body = R_body_yaw⁻¹ @ R_ctrl
            target_t      = p_link0 + offset_body
            target_R      = ctrl_rot_body @ R_link0 @ R_align

        Using only yaw means nodding or tilting the head has no effect on the arm
        mapping. Turning the torso (which rotates the HMD yaw) correctly rotates
        the reference frame so arm motion stays body-relative.

        Note: offset_body is added directly to p_link0 without applying R_link0 to
        the translation — left-multiplying by R_link0 would apply its Rx(±π/2) to
        the FLU offset and permute axes (e.g. up→right). R_link0 only appears on
        the right side of the rotation formula to set the EE rest orientation.

        Args:
            T_ctrl: Controller grip pose in FLU world frame.
            T_shoulder: Shoulder joint pose in FLU world frame (position used only).
            T_head: HMD viewer pose in FLU world frame (orientation used for yaw).
            T_link0_in_world: Fixed SE3 of the arm's link0 in the URDF world frame.

        Returns:
            jaxlie.SE3: IK target in robot URDF world frame.
        """
        # Extract body yaw: project the head's forward vector (+X in FLU) onto the
        # horizontal plane and build a pure-yaw rotation around world Z (up in FLU).
        head_fwd = T_head.rotation() @ jnp.array([1.0, 0.0, 0.0])
        yaw = jnp.arctan2(head_fwd[1], head_fwd[0])
        R_body_yaw = jaxlie.SO3.from_rpy_radians(0.0, 0.0, yaw)

        # Cancel body yaw from both translation offset and controller orientation.
        offset_body = R_body_yaw.inverse() @ (T_ctrl.translation() - T_shoulder.translation())
        ctrl_rot_body = R_body_yaw.inverse() @ T_ctrl.rotation()

        R_align = self.robot.R_align

        # Translation: add the body-frame offset directly to link0's world position.
        # Do NOT left-multiply by R_link0 — that would apply link0's Rx(±π/2) to the
        # FLU offset, swapping Y/Z axes (e.g. up→right, right→down).
        target_translation = T_link0_in_world.translation() + offset_body

        # Rotation: R_link0 on the RIGHT sets the EE rest orientation in world frame;
        # left-multiplying by ctrl_rot_body applies the controller's body-relative
        # rotation on top, keeping all axes in world frame.
        target_rotation = ctrl_rot_body @ T_link0_in_world.rotation() @ R_align

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

    def _get_head_pose(self, state: XRState) -> jaxlie.SE3 | None:
        """
        Extract the HMD viewer pose from the current XR state.

        Returns None if no head device is present, allowing the caller to fall
        back to relative mode gracefully.
        """
        for device in state.devices:
            if device.role == XRDeviceRole.HEAD and device.pose is not None:
                return self.xr_pose_to_se3(device.pose)
        return None

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
        head_pose = self._get_head_pose(state)

        required_keys = self.robot.supported_frames
        has_all_poses = all(k in curr_xr_poses for k in required_keys)

        # Fall back to relative if shoulder or head data is missing
        has_all_shoulders = all(k in shoulder_poses for k in required_keys if k in ("left", "right"))
        if not has_all_shoulders or head_pose is None:
            if is_deadman_active and has_all_poses:
                missing = []
                if not has_all_shoulders:
                    missing.append("shoulder poses (body_joints)")
                if head_pose is None:
                    missing.append("head pose")
                logger.warning(
                    f"[IKController] Absolute mode: {' and '.join(missing)} missing. "
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
                    head_pose,
                    link0_tfs["left"],
                )
            if "right" in required_keys and "right" in link0_tfs:
                target_R = self.compute_absolute_target(
                    curr_xr_poses["right"],
                    shoulder_poses["right"],
                    head_pose,
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
