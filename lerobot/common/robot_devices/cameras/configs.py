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
相机配置模块 (Camera Configuration Module)

功能说明 (Functionality):
    定义不同类型相机的配置类,支持 OpenCV 和 Intel RealSense 相机。

    Defines configuration classes for different camera types, supporting OpenCV and Intel RealSense cameras.

配置类型 (Configuration Types):
    - OpenCVCameraConfig: 通用USB相机配置(基于OpenCV) / Generic USB camera config (OpenCV-based)
    - IntelRealSenseCameraConfig: Intel RealSense深度相机配置 / Intel RealSense depth camera config

使用示例 (Usage Example):
    ```python
    # OpenCV 相机 / OpenCV camera
    from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig

    config = OpenCVCameraConfig(
        camera_index=0,  # 相机索引 / Camera index
        fps=30,          # 帧率 / Frame rate
        width=640,       # 分辨率宽度 / Resolution width
        height=480,      # 分辨率高度 / Resolution height
        color_mode="rgb" # 颜色模式 / Color mode
    )

    # RealSense 相机 / RealSense camera
    from lerobot.common.robot_devices.cameras.configs import IntelRealSenseCameraConfig

    config = IntelRealSenseCameraConfig(
        serial_number=128422271347,  # 相机序列号 / Camera serial number
        fps=30,
        width=640,
        height=480,
        use_depth=True  # 启用深度图 / Enable depth map
    )
    ```
"""

import abc
from dataclasses import dataclass

import draccus


@dataclass
class CameraConfig(draccus.ChoiceRegistry, abc.ABC):
    """
    相机配置基类 (Camera Configuration Base Class)

    功能说明 (Functionality):
        所有相机配置的抽象基类,提供类型注册功能。

        Abstract base class for all camera configurations, providing type registration.
    """

    @property
    def type(self) -> str:
        """返回配置类型名称 / Return configuration type name"""
        return self.get_choice_name(self.__class__)


@CameraConfig.register_subclass("opencv")
@dataclass
class OpenCVCameraConfig(CameraConfig):
    """
    OpenCV 相机配置类 (OpenCV Camera Configuration Class)

    功能说明 (Functionality):
        配置基于 OpenCV 的通用 USB 相机。
        适用于大多数 USB 摄像头,包括网络摄像头和工业相机。

        Configures generic USB cameras based on OpenCV.
        Suitable for most USB cameras, including webcams and industrial cameras.

    属性说明 (Attributes):
        camera_index (int):
            相机索引号 / Camera index number
            Linux: 通常对应 /dev/video{index} / Usually corresponds to /dev/video{index}
            Mac/Windows: 从0开始的整数索引 / Integer index starting from 0
            示例 (Example): 0, 1, 2

        fps (int | None):
            帧率(帧/秒) / Frame rate (frames per second)
            None 表示使用相机默认帧率 / None means use camera default
            典型值 (Typical values): 30, 60, 90

        width (int | None):
            图像宽度(像素) / Image width (pixels)
            None 表示使用相机默认分辨率 / None means use camera default
            典型值 (Typical values): 640, 1280, 1920

        height (int | None):
            图像高度(像素) / Image height (pixels)
            None 表示使用相机默认分辨率 / None means use camera default
            典型值 (Typical values): 480, 720, 1080

        color_mode (str):
            颜色通道顺序 / Color channel order
            选项 (Options): "rgb" (红绿蓝) or "bgr" (蓝绿红,OpenCV默认)
            默认值 (Default): "rgb"

        channels (int | None):
            图像通道数 / Number of image channels
            自动设置为3(RGB) / Automatically set to 3 (RGB)

        rotation (int | None):
            图像旋转角度 / Image rotation angle in degrees
            选项 (Options): -90, None, 90, 180
            用于调整相机安装方向 / Used to adjust camera mounting orientation

        mock (bool):
            是否使用模拟模式(用于测试) / Whether to use mock mode (for testing)
            默认值 (Default): False

    测试配置示例 (Tested Configuration Examples):
        For Intel Real Sense D405:
        ```python
        OpenCVCameraConfig(0, 30, 640, 480)   # 30fps, 640x480
        OpenCVCameraConfig(0, 60, 640, 480)   # 60fps, 640x480
        OpenCVCameraConfig(0, 90, 640, 480)   # 90fps, 640x480
        OpenCVCameraConfig(0, 30, 1280, 720)  # 30fps, 1280x720
        ```
    """

    camera_index: int
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"
    channels: int | None = None
    rotation: int | None = None
    mock: bool = False

    def __post_init__(self):
        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )

        self.channels = 3

        if self.rotation not in [-90, None, 90, 180]:
            raise ValueError(f"`rotation` must be in [-90, None, 90, 180] (got {self.rotation})")


@CameraConfig.register_subclass("intelrealsense")
@dataclass
class IntelRealSenseCameraConfig(CameraConfig):
    """
    Intel RealSense 相机配置类 (Intel RealSense Camera Configuration Class)

    功能说明 (Functionality):
        配置 Intel RealSense 深度相机系列。
        支持 RGB 彩色图像和可选的深度图采集。

        Configures Intel RealSense depth camera series.
        Supports RGB color images and optional depth map acquisition.

    属性说明 (Attributes):
        name (str | None):
            相机名称(通常自动检测) / Camera name (usually auto-detected)
            与 serial_number 二选一 / Mutually exclusive with serial_number

        serial_number (int | None):
            相机序列号,用于识别特定相机 / Camera serial number for identifying specific camera
            当连接多个 RealSense 相机时必需 / Required when multiple RealSense cameras connected
            示例 (Example): 128422271347

        fps (int | None):
            帧率(帧/秒) / Frame rate (frames per second)
            None 表示使用相机默认 / None means use camera default
            典型值 (Typical values): 30, 60, 90

        width (int | None):
            图像宽度(像素) / Image width (pixels)
            None 表示使用相机默认 / None means use camera default
            典型值 (Typical values): 640, 1280

        height (int | None):
            图像高度(像素) / Image height (pixels)
            None 表示使用相机默认 / None means use camera default
            典型值 (Typical values): 480, 720

        color_mode (str):
            颜色通道顺序 / Color channel order
            选项 (Options): "rgb" or "bgr"
            默认值 (Default): "rgb"

        channels (int | None):
            图像通道数 / Number of image channels
            自动设置为3(RGB) / Automatically set to 3 (RGB)

        use_depth (bool):
            是否同时采集深度图 / Whether to acquire depth maps
            深度图提供每个像素的距离信息 / Depth map provides distance info for each pixel
            默认值 (Default): False

        force_hardware_reset (bool):
            连接时是否强制硬件重置 / Whether to force hardware reset on connection
            有助于解决相机状态问题 / Helps resolve camera state issues
            默认值 (Default): True

        rotation (int | None):
            图像旋转角度 / Image rotation angle in degrees
            选项 (Options): -90, None, 90, 180

        mock (bool):
            是否使用模拟模式(用于测试) / Whether to use mock mode (for testing)
            默认值 (Default): False

    测试配置示例 (Tested Configuration Examples):
        For Intel Real Sense D405:
        ```python
        IntelRealSenseCameraConfig(128422271347, 30, 640, 480)                  # 基本配置 / Basic
        IntelRealSenseCameraConfig(128422271347, 60, 640, 480)                  # 高帧率 / High FPS
        IntelRealSenseCameraConfig(128422271347, 90, 640, 480)                  # 超高帧率 / Very high FPS
        IntelRealSenseCameraConfig(128422271347, 30, 1280, 720)                 # 高分辨率 / High resolution
        IntelRealSenseCameraConfig(128422271347, 30, 640, 480, use_depth=True)  # 启用深度 / With depth
        IntelRealSenseCameraConfig(128422271347, 30, 640, 480, rotation=90)     # 旋转90度 / Rotated 90°
        ```

    注意事项 (Notes):
        - fps, width, height 必须全部设置或全部为 None
        - name 和 serial_number 只能设置一个

        - fps, width, height must all be set or all be None
        - Only one of name or serial_number can be set
    """

    name: str | None = None
    serial_number: int | None = None
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"
    channels: int | None = None
    use_depth: bool = False
    force_hardware_reset: bool = True
    rotation: int | None = None
    mock: bool = False

    def __post_init__(self):
        # bool is stronger than is None, since it works with empty strings
        if bool(self.name) and bool(self.serial_number):
            raise ValueError(
                f"One of them must be set: name or serial_number, but {self.name=} and {self.serial_number=} provided."
            )

        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )

        self.channels = 3

        at_least_one_is_not_none = self.fps is not None or self.width is not None or self.height is not None
        at_least_one_is_none = self.fps is None or self.width is None or self.height is None
        if at_least_one_is_not_none and at_least_one_is_none:
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them, "
                f"but {self.fps=}, {self.width=}, {self.height=} were provided."
            )

        if self.rotation not in [-90, None, 90, 180]:
            raise ValueError(f"`rotation` must be in [-90, None, 90, 180] (got {self.rotation})")
