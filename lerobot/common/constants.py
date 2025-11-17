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
全局常量定义模块 (Global Constants Definition Module)

功能说明 (Functionality):
    定义 LeRobot 库中使用的所有全局常量,包括:
    - 数据集键名称 (Dataset key names)
    - 文件和目录名称 (File and directory names)
    - 缓存路径配置 (Cache path configuration)

    Defines all global constants used throughout the LeRobot library, including:
    - Dataset key names
    - File and directory names
    - Cache path configuration
"""

import os
from pathlib import Path

from huggingface_hub.constants import HF_HOME

# ================================================================================
# 数据集键名称常量 (Dataset Key Name Constants)
# ================================================================================
# 这些常量定义了数据集中各个数据字段的标准键名
# These constants define the standard key names for various data fields in datasets

OBS_ENV = "observation.environment_state"  # 环境状态观测键 / Environment state observation key
                                           # 类型 (Type): str
                                           # 用于存储环境的全局状态信息
                                           # Used to store global environment state information

OBS_ROBOT = "observation.state"           # 机器人状态观测键 / Robot state observation key
                                          # 类型 (Type): str
                                          # 用于存储机器人的关节位置、速度等状态
                                          # Used to store robot joint positions, velocities, etc.

OBS_IMAGE = "observation.image"           # 单个图像观测键 / Single image observation key
                                          # 类型 (Type): str
                                          # 用于存储单个相机的图像数据
                                          # Used to store image data from a single camera

OBS_IMAGES = "observation.images"         # 多个图像观测键 / Multiple images observation key
                                          # 类型 (Type): str
                                          # 用于存储多个相机的图像数据
                                          # Used to store image data from multiple cameras

ACTION = "action"                         # 动作键 / Action key
                                          # 类型 (Type): str
                                          # 用于存储机器人执行的动作数据
                                          # Used to store action data executed by the robot

# ================================================================================
# 文件和目录名称常量 (File and Directory Name Constants)
# ================================================================================
# 这些常量定义了训练、检查点和模型存储的标准文件/目录结构
# These constants define the standard file/directory structure for training, checkpoints, and model storage

CHECKPOINTS_DIR = "checkpoints"           # 检查点目录名 / Checkpoints directory name
                                          # 类型 (Type): str
                                          # 存储训练过程中保存的所有检查点
                                          # Stores all checkpoints saved during training

LAST_CHECKPOINT_LINK = "last"             # 最新检查点链接名 / Latest checkpoint link name
                                          # 类型 (Type): str
                                          # 指向最新检查点的符号链接或引用
                                          # Symbolic link or reference to the latest checkpoint

PRETRAINED_MODEL_DIR = "pretrained_model" # 预训练模型目录名 / Pretrained model directory name
                                          # 类型 (Type): str
                                          # 存储预训练模型权重和配置
                                          # Stores pretrained model weights and configuration

TRAINING_STATE_DIR = "training_state"     # 训练状态目录名 / Training state directory name
                                          # 类型 (Type): str
                                          # 存储训练状态(优化器、调度器、随机数生成器等)
                                          # Stores training state (optimizer, scheduler, RNG, etc.)

RNG_STATE = "rng_state.safetensors"       # 随机数生成器状态文件名 / RNG state filename
                                          # 类型 (Type): str
                                          # 保存随机数生成器状态以确保可重现性
                                          # Saves RNG state for reproducibility

TRAINING_STEP = "training_step.json"      # 训练步数文件名 / Training step filename
                                          # 类型 (Type): str
                                          # 保存当前训练步数
                                          # Saves current training step number

OPTIMIZER_STATE = "optimizer_state.safetensors"  # 优化器状态文件名 / Optimizer state filename
                                                 # 类型 (Type): str
                                                 # 保存优化器状态(动量、梯度等)
                                                 # Saves optimizer state (momentum, gradients, etc.)

OPTIMIZER_PARAM_GROUPS = "optimizer_param_groups.json"  # 优化器参数组文件名 / Optimizer param groups filename
                                                        # 类型 (Type): str
                                                        # 保存优化器参数组配置
                                                        # Saves optimizer parameter group configuration

SCHEDULER_STATE = "scheduler_state.json"  # 学习率调度器状态文件名 / Scheduler state filename
                                          # 类型 (Type): str
                                          # 保存学习率调度器状态
                                          # Saves learning rate scheduler state

# ================================================================================
# 缓存目录配置 (Cache Directory Configuration)
# ================================================================================
# 配置 LeRobot 数据和模型的本地缓存位置
# Configures the local cache location for LeRobot data and models

# 默认缓存路径:在 HuggingFace Home 目录下创建 lerobot 子目录
# Default cache path: creates a lerobot subdirectory under HuggingFace Home
# 类型 (Type): Path
default_cache_path = Path(HF_HOME) / "lerobot"

# LeRobot 主目录:可通过环境变量 HF_LEROBOT_HOME 自定义
# LeRobot home directory: can be customized via HF_LEROBOT_HOME environment variable
# 类型 (Type): Path
# 默认值 (Default): ~/.cache/huggingface/lerobot (在大多数系统上)
# 可通过设置环境变量 HF_LEROBOT_HOME 来覆盖
# Can be overridden by setting the HF_LEROBOT_HOME environment variable
HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", default_cache_path)).expanduser()

# 检查是否使用了已弃用的环境变量
# Check for deprecated environment variable usage
if "LEROBOT_HOME" in os.environ:
    raise ValueError(
        f"You have a 'LEROBOT_HOME' environment variable set to '{os.getenv('LEROBOT_HOME')}'.\n"
        "'LEROBOT_HOME' is deprecated, please use 'HF_LEROBOT_HOME' instead."
        # 提示:LEROBOT_HOME 已弃用,请改用 HF_LEROBOT_HOME
        # Note: LEROBOT_HOME is deprecated, please use HF_LEROBOT_HOME instead
    )
