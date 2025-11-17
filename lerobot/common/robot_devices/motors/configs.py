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
舵机总线配置模块 (Motor Bus Configuration Module)

功能说明 (Functionality):
    定义舵机总线的配置类,支持不同品牌的舵机。

    Defines configuration classes for motor buses, supporting different motor brands.

配置类型 (Configuration Types):
    - DynamixelMotorsBusConfig: Robotis Dynamixel 舵机配置
    - FeetechMotorsBusConfig: Feetech 舵机配置

使用方式 (Usage):
    ```python
    from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig

    config = DynamixelMotorsBusConfig(
        port="/dev/ttyUSB0",
        motors={
            "shoulder": (1, "xl430-w250"),  # 舵机名称: (ID, 型号) / name: (id, model)
            "elbow": (2, "xl430-w250"),
        }
    )
    ```
"""

import abc
from dataclasses import dataclass

import draccus


@dataclass
class MotorsBusConfig(draccus.ChoiceRegistry, abc.ABC):
    """
    舵机总线配置基类 (Motor Bus Configuration Base Class)

    功能说明 (Functionality):
        所有舵机总线配置的抽象基类,提供类型注册功能。

        Abstract base class for all motor bus configurations, providing type registration.
    """

    @property
    def type(self) -> str:
        """返回配置类型名称 / Return configuration type name"""
        return self.get_choice_name(self.__class__)


@MotorsBusConfig.register_subclass("dynamixel")
@dataclass
class DynamixelMotorsBusConfig(MotorsBusConfig):
    """
    Dynamixel 舵机总线配置 (Dynamixel Motor Bus Configuration)

    属性说明 (Attributes):
        port (str):
            串口路径 / Serial port path
            示例 (Example): "/dev/ttyUSB0", "COM3"

        motors (dict[str, tuple[int, str]]):
            舵机配置字典 / Motor configuration dictionary
            格式 (Format): {舵机名称: (舵机ID, 舵机型号) / {motor_name: (motor_id, motor_model)}
            示例 (Example):
                {
                    "shoulder": (1, "xl430-w250"),
                    "elbow": (2, "xl430-w250"),
                    "gripper": (3, "xl330-m288")
                }

        mock (bool):
            是否使用模拟模式(用于测试) / Whether to use mock mode (for testing)
            默认值 (Default): False
    """

    port: str
    motors: dict[str, tuple[int, str]]
    mock: bool = False


@MotorsBusConfig.register_subclass("feetech")
@dataclass
class FeetechMotorsBusConfig(MotorsBusConfig):
    """
    Feetech 舵机总线配置 (Feetech Motor Bus Configuration)

    属性说明 (Attributes):
        port (str):
            串口路径 / Serial port path
            示例 (Example): "/dev/ttyUSB0", "COM3"

        motors (dict[str, tuple[int, str]]):
            舵机配置字典 / Motor configuration dictionary
            格式 (Format): {舵机名称: (舵机ID, 舵机型号) / {motor_name: (motor_id, motor_model)}
            示例 (Example):
                {
                    "joint1": (1, "sts3215"),
                    "joint2": (2, "sts3215")
                }

        mock (bool):
            是否使用模拟模式(用于测试) / Whether to use mock mode (for testing)
            默认值 (Default): False
    """

    port: str
    motors: dict[str, tuple[int, str]]
    mock: bool = False
