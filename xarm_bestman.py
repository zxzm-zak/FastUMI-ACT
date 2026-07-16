"""Compatibility helpers for the external BestMan XArm6 driver."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Type


OFFICIAL_BESTMAN_URL = "https://github.com/UMI-Robotics/BestMan_Xarm"
XARM_GRIPPER_MAX_POSITION = 850.0


def _prepend_import_path(path: Path) -> None:
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)


def _load_from_bestman_root(root: Path) -> Type[Any]:
    module_path = root / "RoboticsToolBox" / "Bestman_real_xarm6.py"
    if not module_path.is_file():
        raise RuntimeError(
            f"{root} is not a BestMan_Xarm checkout: expected {module_path}."
        )

    _prepend_import_path(root)
    try:
        from RoboticsToolBox.Bestman_real_xarm6 import Bestman_Real_Xarm6
    except ImportError as error:
        raise RuntimeError(
            "BestMan_Xarm was found but its Python dependencies are unavailable. "
            "Install the environment from BestMan_Xarm/Install/basic_environment.yaml."
        ) from error
    return Bestman_Real_Xarm6


def _load_from_legacy_sdk(sdk_dir: Path) -> Type[Any]:
    module_path = sdk_dir / "Bestman_real_xarm6.py"
    if not module_path.is_file():
        raise RuntimeError(
            f"{sdk_dir} does not contain Bestman_real_xarm6.py. "
            "Pass the root of BestMan_Xarm with --bestman-root instead."
        )

    _prepend_import_path(sdk_dir)
    try:
        from Bestman_real_xarm6 import Bestman_Real_Xarm6
    except ImportError as error:
        raise RuntimeError(
            "The legacy XArm6 driver was found but could not be imported."
        ) from error
    return Bestman_Real_Xarm6


def load_xarm_driver(
    bestman_root: Optional[str], sdk_dir: Optional[str]
) -> Type[Any]:
    """Load BestMan from the official checkout or a legacy driver directory."""
    if bestman_root:
        return _load_from_bestman_root(Path(bestman_root).expanduser().resolve())

    if sdk_dir:
        sdk_path = Path(sdk_dir).expanduser().resolve()
        if (sdk_path / "RoboticsToolBox" / "Bestman_real_xarm6.py").is_file():
            return _load_from_bestman_root(sdk_path)
        return _load_from_legacy_sdk(sdk_path)

    try:
        from RoboticsToolBox.Bestman_real_xarm6 import Bestman_Real_Xarm6
    except ImportError:
        try:
            from Bestman_real_xarm6 import Bestman_Real_Xarm6
        except ImportError as error:
            raise RuntimeError(
                "Unable to import the XArm6 driver. Clone "
                f"{OFFICIAL_BESTMAN_URL} and pass --bestman-root, or set "
                "ACT_BESTMAN_XARM_ROOT."
            ) from error
    return Bestman_Real_Xarm6


def initialize_robot(robot: Any) -> None:
    """Initialize the robot when the selected driver exposes an initializer."""
    initializer = getattr(robot, "initialize_robot", None)
    if initializer is not None and initializer() is False:
        raise RuntimeError("BestMan failed to initialize the XArm6 robot.")


def get_gripper_position(robot: Any) -> float:
    """Read the XArm gripper position across supported BestMan API versions."""
    reader = getattr(robot, "get_gripper_pose_xarm", None)
    if reader is None:
        reader = getattr(robot, "get_gripper_position", None)
    if reader is None:
        raise RuntimeError("The selected XArm driver has no gripper position reader.")
    return float(reader())


def move_arm_to_joint_values(robot: Any, joint_values: Any) -> Any:
    """Send a non-blocking joint command through the available BestMan API."""
    command = getattr(robot, "move_arm_to_joint_values", None)
    if command is None:
        command = getattr(robot, "move_arm_to_joint_angles", None)
    if command is None:
        raise RuntimeError("The selected XArm driver has no joint motion command.")
    return command(joint_values, wait_for_finish=False)


def command_gripper(robot: Any, position: float) -> Any:
    """Clamp and send an XArm gripper command through the available API."""
    target = float(min(max(position, 0.0), XARM_GRIPPER_MAX_POSITION))
    command = getattr(robot, "gripper_goto_xarm", None)
    if command is None:
        command = getattr(robot, "gripper_goto", None)
    if command is None:
        raise RuntimeError("The selected XArm driver has no gripper motion command.")
    return command(target)
