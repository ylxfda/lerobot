# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
机器人配置模块 (Robot Configuration Module)

功能说明 (Functionality):
    定义机器人系统的完整配置,包括电机、相机和安全参数。
    提供预定义的机器人配置(如 Aloha, Koch)。

    Defines complete robot system configuration including motors, cameras, and safety parameters.
    Provides pre-defined robot configurations (e.g., Aloha, Koch).

配置层次 (Configuration Hierarchy):
    RobotConfig (基类 / Base)
    └── ManipulatorRobotConfig (机械臂基类 / Manipulator base)
        ├── AlohaRobotConfig (Aloha双臂机器人 / Aloha bi-manual robot)
        └── KochRobotConfig (Koch低成本机械臂 / Koch low-cost manipulator)

主要配置参数 (Main Configuration Parameters):
    - leader_arms: 主臂电机配置(用于遥操作) / Leader arm motors (for teleoperation)
    - follower_arms: 从臂电机配置(执行动作) / Follower arm motors (execute actions)
    - cameras: 相机配置 / Camera configuration
    - max_relative_target: 安全限制参数 / Safety limit parameter
    - calibration_dir: 校准文件目录 / Calibration file directory

使用示例 (Usage Example):
    ```python
    from lerobot.common.robot_devices.robots.configs import AlohaRobotConfig
    from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

    # 使用预定义配置 / Use pre-defined config
    config = AlohaRobotConfig()
    robot = ManipulatorRobot(config)

    # 自定义配置 / Custom config
    config = AlohaRobotConfig(
        max_relative_target=10,  # 增加安全限制 / Increase safety limit
        calibration_dir=".cache/calibration/my_aloha"
    )
    ```
"""

import abc
from dataclasses import dataclass, field
from typing import Sequence

import draccus

from lerobot.common.robot_devices.cameras.configs import (
    CameraConfig,
    IntelRealSenseCameraConfig,
    OpenCVCameraConfig,
)
from lerobot.common.robot_devices.motors.configs import (
    DynamixelMotorsBusConfig,
    FeetechMotorsBusConfig,
    MotorsBusConfig,
)


@dataclass
class RobotConfig(draccus.ChoiceRegistry, abc.ABC):
    """
    机器人配置基类 (Robot Configuration Base Class)

    功能说明 (Functionality):
        所有机器人配置的抽象基类,提供类型注册功能。

        Abstract base class for all robot configurations, providing type registration.
    """

    @property
    def type(self) -> str:
        """返回配置类型名称 / Return configuration type name"""
        return self.get_choice_name(self.__class__)


# TODO(rcadene, aliberts): remove ManipulatorRobotConfig abstraction
@dataclass
class ManipulatorRobotConfig(RobotConfig):
    """
    机械臂机器人配置类 (Manipulator Robot Configuration Class)

    功能说明 (Functionality):
        定义机械臂机器人的完整配置,包括主臂、从臂、相机和安全参数。
        适用于所有主从式遥操作机械臂。

        Defines complete configuration for manipulator robots, including leader arms,
        follower arms, cameras, and safety parameters. Suitable for all leader-follower
        teleoperation manipulators.

    属性说明 (Attributes):
        leader_arms (dict[str, MotorsBusConfig]):
            主臂电机总线配置字典 / Leader arm motor bus configuration dictionary
            结构 (Structure): {臂名称: 电机总线配置 / {arm_name: motor_bus_config}}
            用途 (Purpose): 遥操作时人工控制的臂 / Human-controlled arm during teleoperation
            示例 (Example): {"main": DynamixelMotorsBusConfig(...)}

        follower_arms (dict[str, MotorsBusConfig]):
            从臂电机总线配置字典 / Follower arm motor bus configuration dictionary
            结构 (Structure): {臂名称: 电机总线配置 / {arm_name: motor_bus_config}}
            用途 (Purpose): 跟随主臂或执行策略动作的臂 / Arm that follows leader or executes policy actions
            示例 (Example): {"main": DynamixelMotorsBusConfig(...)}

        cameras (dict[str, CameraConfig]):
            相机配置字典 / Camera configuration dictionary
            结构 (Structure): {相机名称: 相机配置 / {camera_name: camera_config}}
            示例 (Example): {"top": OpenCVCameraConfig(...), "wrist": OpenCVCameraConfig(...)}

        max_relative_target (list[float] | float | None):
            安全限制:单步最大相对移动量(度数) / Safety limit: max relative movement per step (degrees)
            用途 (Purpose): 防止电机突然大幅移动,保护硬件和操作人员
            类型 (Types):
            - float: 所有电机使用相同限制 / Same limit for all motors
            - list[float]: 每个电机独立限制 / Individual limit per motor
            - None: 无限制(不推荐) / No limit (not recommended)
            示例 (Example): 5.0 表示每步最多移动5度 / 5.0 means max 5 degrees per step
            默认值 (Default): None

        gripper_open_degree (float | None):
            夹爪开启角度(用于扭矩模式) / Gripper open angle (for torque mode)
            用途 (Purpose): 设置主臂夹爪在扭矩模式下的弹簧回位角度
            None 表示不使用扭矩模式 / None means don't use torque mode
            示例 (Example): 0.0 表示完全打开 / 0.0 means fully open
            默认值 (Default): None

        mock (bool):
            是否使用模拟模式(用于测试) / Whether to use mock mode (for testing)
            会自动传播到所有子设备配置 / Automatically propagates to all sub-device configs
            默认值 (Default): False

    安全说明 (Safety Notes):
        max_relative_target 是关键安全参数:
        - 首次使用建议设置为 5 度 / Recommend 5 degrees for first use
        - 熟悉操作后可逐渐增加 / Can gradually increase after familiarization
        - 移除限制前务必确保周围环境安全 / Ensure environment safety before removing limit

        max_relative_target is a critical safety parameter:
        - Recommend 5 degrees for first use
        - Can gradually increase after familiarization
        - Ensure environment safety before removing limit

    使用示例 (Usage Example):
        ```python
        from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig
        from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig

        config = ManipulatorRobotConfig(
            leader_arms={
                "main": DynamixelMotorsBusConfig(
                    port="/dev/ttyUSB0",
                    motors={"joint1": (1, "xl330-m077"), "joint2": (2, "xl330-m077")}
                )
            },
            follower_arms={
                "main": DynamixelMotorsBusConfig(
                    port="/dev/ttyUSB1",
                    motors={"joint1": (1, "xl430-w250"), "joint2": (2, "xl430-w250")}
                )
            },
            cameras={
                "top": OpenCVCameraConfig(camera_index=0, fps=30, width=640, height=480)
            },
            max_relative_target=5.0,  # 安全限制5度 / Safety limit 5 degrees
        )
        ```
    """

    leader_arms: dict[str, MotorsBusConfig] = field(default_factory=lambda: {})
    follower_arms: dict[str, MotorsBusConfig] = field(default_factory=lambda: {})
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {})

    # Optionally limit the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length
    # as the number of motors in your follower arms (assumes all follower arms have the same number of
    # motors).
    max_relative_target: list[float] | float | None = None

    # Optionally set the leader arm in torque mode with the gripper motor set to this angle. This makes it
    # possible to squeeze the gripper and have it spring back to an open position on its own. If None, the
    # gripper is not put in torque mode.
    gripper_open_degree: float | None = None

    mock: bool = False

    def __post_init__(self):
        if self.mock:
            for arm in self.leader_arms.values():
                if not arm.mock:
                    arm.mock = True
            for arm in self.follower_arms.values():
                if not arm.mock:
                    arm.mock = True
            for cam in self.cameras.values():
                if not cam.mock:
                    cam.mock = True

        if self.max_relative_target is not None and isinstance(self.max_relative_target, Sequence):
            for name in self.follower_arms:
                if len(self.follower_arms[name].motors) != len(self.max_relative_target):
                    raise ValueError(
                        f"len(max_relative_target)={len(self.max_relative_target)} but the follower arm with name {name} has "
                        f"{len(self.follower_arms[name].motors)} motors. Please make sure that the "
                        f"`max_relative_target` list has as many parameters as there are motors per arm. "
                        "Note: This feature does not yet work with robots where different follower arms have "
                        "different numbers of motors."
                    )


@RobotConfig.register_subclass("aloha")
@dataclass
class AlohaRobotConfig(ManipulatorRobotConfig):
    # Specific to Aloha, LeRobot comes with default calibration files. Assuming the motors have been
    # properly assembled, no manual calibration step is expected. If you need to run manual calibration,
    # simply update this path to ".cache/calibration/aloha"
    calibration_dir: str = ".cache/calibration/aloha_default"

    # /!\ FOR SAFETY, READ THIS /!\
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    # For Aloha, for every goal position request, motor rotations are capped at 5 degrees by default.
    # When you feel more confident with teleoperation or running the policy, you can extend
    # this safety limit and even removing it by setting it to `null`.
    # Also, everything is expected to work safely out-of-the-box, but we highly advise to
    # first try to teleoperate the grippers only (by commenting out the rest of the motors in this yaml),
    # then to gradually add more motors (by uncommenting), until you can teleoperate both arms fully
    max_relative_target: int | None = 5

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": DynamixelMotorsBusConfig(
                # window_x
                port="/dev/ttyDXL_leader_left",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm430-w350"],
                    "shoulder": [2, "xm430-w350"],
                    "shoulder_shadow": [3, "xm430-w350"],
                    "elbow": [4, "xm430-w350"],
                    "elbow_shadow": [5, "xm430-w350"],
                    "forearm_roll": [6, "xm430-w350"],
                    "wrist_angle": [7, "xm430-w350"],
                    "wrist_rotate": [8, "xl430-w250"],
                    "gripper": [9, "xc430-w150"],
                },
            ),
            "right": DynamixelMotorsBusConfig(
                # window_x
                port="/dev/ttyDXL_leader_right",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm430-w350"],
                    "shoulder": [2, "xm430-w350"],
                    "shoulder_shadow": [3, "xm430-w350"],
                    "elbow": [4, "xm430-w350"],
                    "elbow_shadow": [5, "xm430-w350"],
                    "forearm_roll": [6, "xm430-w350"],
                    "wrist_angle": [7, "xm430-w350"],
                    "wrist_rotate": [8, "xl430-w250"],
                    "gripper": [9, "xc430-w150"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": DynamixelMotorsBusConfig(
                port="/dev/ttyDXL_follower_left",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm540-w270"],
                    "shoulder": [2, "xm540-w270"],
                    "shoulder_shadow": [3, "xm540-w270"],
                    "elbow": [4, "xm540-w270"],
                    "elbow_shadow": [5, "xm540-w270"],
                    "forearm_roll": [6, "xm540-w270"],
                    "wrist_angle": [7, "xm540-w270"],
                    "wrist_rotate": [8, "xm430-w350"],
                    "gripper": [9, "xm430-w350"],
                },
            ),
            "right": DynamixelMotorsBusConfig(
                port="/dev/ttyDXL_follower_right",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm540-w270"],
                    "shoulder": [2, "xm540-w270"],
                    "shoulder_shadow": [3, "xm540-w270"],
                    "elbow": [4, "xm540-w270"],
                    "elbow_shadow": [5, "xm540-w270"],
                    "forearm_roll": [6, "xm540-w270"],
                    "wrist_angle": [7, "xm540-w270"],
                    "wrist_rotate": [8, "xm430-w350"],
                    "gripper": [9, "xm430-w350"],
                },
            ),
        }
    )

    # Troubleshooting: If one of your IntelRealSense cameras freeze during
    # data recording due to bandwidth limit, you might need to plug the camera
    # on another USB hub or PCIe card.
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_high": IntelRealSenseCameraConfig(
                serial_number=128422271347,
                fps=30,
                width=640,
                height=480,
            ),
            "cam_low": IntelRealSenseCameraConfig(
                serial_number=130322270656,
                fps=30,
                width=640,
                height=480,
            ),
            "cam_left_wrist": IntelRealSenseCameraConfig(
                serial_number=218622272670,
                fps=30,
                width=640,
                height=480,
            ),
            "cam_right_wrist": IntelRealSenseCameraConfig(
                serial_number=130322272300,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("koch")
@dataclass
class KochRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/koch"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem585A0085511",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl330-m077"],
                    "shoulder_lift": [2, "xl330-m077"],
                    "elbow_flex": [3, "xl330-m077"],
                    "wrist_flex": [4, "xl330-m077"],
                    "wrist_roll": [5, "xl330-m077"],
                    "gripper": [6, "xl330-m077"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl430-w250"],
                    "shoulder_lift": [2, "xl430-w250"],
                    "elbow_flex": [3, "xl330-m288"],
                    "wrist_flex": [4, "xl330-m288"],
                    "wrist_roll": [5, "xl330-m288"],
                    "gripper": [6, "xl330-m288"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    # ~ Koch specific settings ~
    # Sets the leader arm in torque mode with the gripper motor set to this angle. This makes it possible
    # to squeeze the gripper and have it spring back to an open position on its own.
    gripper_open_degree: float = 35.156

    mock: bool = False


@RobotConfig.register_subclass("koch_bimanual")
@dataclass
class KochBimanualRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/koch_bimanual"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem585A0085511",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl330-m077"],
                    "shoulder_lift": [2, "xl330-m077"],
                    "elbow_flex": [3, "xl330-m077"],
                    "wrist_flex": [4, "xl330-m077"],
                    "wrist_roll": [5, "xl330-m077"],
                    "gripper": [6, "xl330-m077"],
                },
            ),
            "right": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem575E0031751",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl330-m077"],
                    "shoulder_lift": [2, "xl330-m077"],
                    "elbow_flex": [3, "xl330-m077"],
                    "wrist_flex": [4, "xl330-m077"],
                    "wrist_roll": [5, "xl330-m077"],
                    "gripper": [6, "xl330-m077"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl430-w250"],
                    "shoulder_lift": [2, "xl430-w250"],
                    "elbow_flex": [3, "xl330-m288"],
                    "wrist_flex": [4, "xl330-m288"],
                    "wrist_roll": [5, "xl330-m288"],
                    "gripper": [6, "xl330-m288"],
                },
            ),
            "right": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem575E0032081",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl430-w250"],
                    "shoulder_lift": [2, "xl430-w250"],
                    "elbow_flex": [3, "xl330-m288"],
                    "wrist_flex": [4, "xl330-m288"],
                    "wrist_roll": [5, "xl330-m288"],
                    "gripper": [6, "xl330-m288"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    # ~ Koch specific settings ~
    # Sets the leader arm in torque mode with the gripper motor set to this angle. This makes it possible
    # to squeeze the gripper and have it spring back to an open position on its own.
    gripper_open_degree: float = 35.156

    mock: bool = False


@RobotConfig.register_subclass("moss")
@dataclass
class MossRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/moss"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem58760431091",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("so100")
@dataclass
class So100RobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/so100"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem58760431091",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("stretch")
@dataclass
class StretchRobotConfig(RobotConfig):
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "navigation": OpenCVCameraConfig(
                camera_index="/dev/hello-nav-head-camera",
                fps=10,
                width=1280,
                height=720,
                rotation=-90,
            ),
            "head": IntelRealSenseCameraConfig(
                name="Intel RealSense D435I",
                fps=30,
                width=640,
                height=480,
                rotation=90,
            ),
            "wrist": IntelRealSenseCameraConfig(
                name="Intel RealSense D405",
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("lekiwi")
@dataclass
class LeKiwiRobotConfig(RobotConfig):
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # Network Configuration
    ip: str = "192.168.0.193"
    port: int = 5555
    video_port: int = 5556

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "front": OpenCVCameraConfig(
                camera_index="/dev/video0", fps=30, width=640, height=480, rotation=90
            ),
            "wrist": OpenCVCameraConfig(
                camera_index="/dev/video2", fps=30, width=640, height=480, rotation=180
            ),
        }
    )

    calibration_dir: str = ".cache/calibration/lekiwi"

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem585A0077581",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyACM0",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                    "left_wheel": (7, "sts3215"),
                    "back_wheel": (8, "sts3215"),
                    "right_wheel": (9, "sts3215"),
                },
            ),
        }
    )

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "z",
            "rotate_right": "x",
            # Speed control
            "speed_up": "r",
            "speed_down": "f",
            # quit teleop
            "quit": "q",
        }
    )

    mock: bool = False
