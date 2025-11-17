#!/usr/bin/env python

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
LeRobot 库的主初始化文件
Main initialization file for the LeRobot library

功能说明 (Functionality):
    这个文件包含了LeRobot库中所有可用的环境、数据集和策略的列表,用于反映库的当前状态。
    为了保持轻量级并确保快速访问这些变量,我们不导入所有依赖项。
    This file contains lists of available environments, datasets, and policies to reflect
    the current state of the LeRobot library. We do not want to import all dependencies,
    but keep it lightweight to ensure fast access to these variables.

使用示例 (Example):
    ```python
        import lerobot
        # 打印所有可用的环境 / Print all available environments
        print(lerobot.available_envs)
        # 打印每个环境的可用任务 / Print available tasks per environment
        print(lerobot.available_tasks_per_env)
        # 打印所有可用的数据集 / Print all available datasets
        print(lerobot.available_datasets)
        # 打印每个环境的可用数据集 / Print available datasets per environment
        print(lerobot.available_datasets_per_env)
        # 打印所有真实世界数据集 / Print all real-world datasets
        print(lerobot.available_real_world_datasets)
        # 打印所有可用的策略 / Print all available policies
        print(lerobot.available_policies)
        # 打印每个环境的可用策略 / Print available policies per environment
        print(lerobot.available_policies_per_env)
        # 打印所有可用的机器人 / Print all available robots
        print(lerobot.available_robots)
        # 打印所有可用的相机 / Print all available cameras
        print(lerobot.available_cameras)
        # 打印所有可用的电机 / Print all available motors
        print(lerobot.available_motors)
    ```

实现新组件的步骤 (Steps for implementing new components):

    实现可用 LeRobotDataset 加载的新数据集时:
    When implementing a new dataset loadable with LeRobotDataset:
    - 更新 `lerobot/__init__.py` 中的 `available_datasets_per_env`
      Update `available_datasets_per_env` in `lerobot/__init__.py`

    实现新环境时 (例如 `gym_aloha`):
    When implementing a new environment (e.g. `gym_aloha`):
    - 更新 `lerobot/__init__.py` 中的 `available_tasks_per_env` 和 `available_datasets_per_env`
      Update `available_tasks_per_env` and `available_datasets_per_env` in `lerobot/__init__.py`

    实现新策略类时 (例如 `DiffusionPolicy`):
    When implementing a new policy class (e.g. `DiffusionPolicy`):
    - 更新 `lerobot/__init__.py` 中的 `available_policies` 和 `available_policies_per_env`
      Update `available_policies` and `available_policies_per_env` in `lerobot/__init__.py`
    - 设置必需的 `name` 类属性
      Set the required `name` class attribute
    - 通过导入新的策略类来更新 `tests/test_available.py` 中的变量
      Update variables in `tests/test_available.py` by importing your new Policy class
"""

import itertools

# 导入版本信息 / Import version information
from lerobot.__version__ import __version__  # noqa: F401

# TODO(rcadene): 改进策略和环境的命名。目前,`available_policies` 中的项目既指向 yaml 文件又指向建模名称。
# `available_envs` 也类似,既指向 yaml 文件又指向环境名称。应该让这些区别更明显。
# TODO(rcadene): Improve policies and envs. As of now, an item in `available_policies`
# refers to a yaml file AND a modeling name. Same for `available_envs` which refers to
# a yaml file AND a environment name. The difference should be more obvious.

# ================================================================================
# 可用环境和任务 (Available Environments and Tasks)
# ================================================================================
# 每个环境对应的可用任务列表
# Dictionary mapping each environment to its list of available tasks
# 类型: Dict[str, List[str]]
# Type: Dict[str, List[str]]
# 键 (Keys): 环境名称 (environment name)
# 值 (Values): 任务名称列表 (list of task names)
available_tasks_per_env = {
    "aloha": [  # ALOHA 双臂机器人环境 / ALOHA bimanual robot environment
        "AlohaInsertion-v0",      # 插入任务 / Insertion task
        "AlohaTransferCube-v0",   # 立方体转移任务 / Cube transfer task
    ],
    "pusht": ["PushT-v0"],  # PushT 推动任务环境 / PushT pushing task environment
    "xarm": ["XarmLift-v0"],  # XArm 机械臂抓取任务 / XArm manipulation lifting task
}

# 从 available_tasks_per_env 的键中提取所有可用的环境名称
# Extract all available environment names from keys of available_tasks_per_env
# 类型: List[str] - 环境名称列表 / List of environment names
available_envs = list(available_tasks_per_env.keys())

# ================================================================================
# 可用数据集 (Available Datasets)
# ================================================================================
# 每个环境对应的可用数据集列表
# Dictionary mapping each environment to its list of available datasets
# 类型: Dict[str, List[str]]
# 键 (Keys): 环境名称 (environment name)
# 值 (Values): 数据集名称列表,格式为 "lerobot/dataset_name"
#             (list of dataset names in format "lerobot/dataset_name")
available_datasets_per_env = {
    "aloha": [
        "lerobot/aloha_sim_insertion_human",
        "lerobot/aloha_sim_insertion_scripted",
        "lerobot/aloha_sim_transfer_cube_human",
        "lerobot/aloha_sim_transfer_cube_scripted",
        "lerobot/aloha_sim_insertion_human_image",
        "lerobot/aloha_sim_insertion_scripted_image",
        "lerobot/aloha_sim_transfer_cube_human_image",
        "lerobot/aloha_sim_transfer_cube_scripted_image",
    ],
    # TODO(alexander-soare): 添加 "lerobot/pusht_keypoints"。目前无法添加因为与测试耦合太紧密。
    # TODO(alexander-soare): Add "lerobot/pusht_keypoints". Right now we can't because this is too tightly
    # coupled with tests.
    "pusht": [  # PushT 环境数据集 / PushT environment datasets
        "lerobot/pusht",        # 状态表示的数据集 / State representation dataset
        "lerobot/pusht_image"   # 图像表示的数据集 / Image representation dataset
    ],
    "xarm": [
        "lerobot/xarm_lift_medium",
        "lerobot/xarm_lift_medium_replay",
        "lerobot/xarm_push_medium",
        "lerobot/xarm_push_medium_replay",
        "lerobot/xarm_lift_medium_image",
        "lerobot/xarm_lift_medium_replay_image",
        "lerobot/xarm_push_medium_image",
        "lerobot/xarm_push_medium_replay_image",
    ],
}

# ================================================================================
# 真实世界数据集 (Real-World Datasets)
# ================================================================================
# 从真实机器人硬件收集的数据集列表
# List of datasets collected from real robot hardware
# 类型: List[str] - 数据集名称列表,格式为 "lerobot/dataset_name"
# Type: List[str] - List of dataset names in format "lerobot/dataset_name"
# 这些数据集包含来自各种研究机构和真实机器人平台的数据
# These datasets contain data from various research institutions and real robot platforms
available_real_world_datasets = [
    "lerobot/aloha_mobile_cabinet",
    "lerobot/aloha_mobile_chair",
    "lerobot/aloha_mobile_elevator",
    "lerobot/aloha_mobile_shrimp",
    "lerobot/aloha_mobile_wash_pan",
    "lerobot/aloha_mobile_wipe_wine",
    "lerobot/aloha_static_battery",
    "lerobot/aloha_static_candy",
    "lerobot/aloha_static_coffee",
    "lerobot/aloha_static_coffee_new",
    "lerobot/aloha_static_cups_open",
    "lerobot/aloha_static_fork_pick_up",
    "lerobot/aloha_static_pingpong_test",
    "lerobot/aloha_static_pro_pencil",
    "lerobot/aloha_static_screw_driver",
    "lerobot/aloha_static_tape",
    "lerobot/aloha_static_thread_velcro",
    "lerobot/aloha_static_towel",
    "lerobot/aloha_static_vinh_cup",
    "lerobot/aloha_static_vinh_cup_left",
    "lerobot/aloha_static_ziploc_slide",
    "lerobot/umi_cup_in_the_wild",
    "lerobot/unitreeh1_fold_clothes",
    "lerobot/unitreeh1_rearrange_objects",
    "lerobot/unitreeh1_two_robot_greeting",
    "lerobot/unitreeh1_warehouse",
    "lerobot/nyu_rot_dataset",
    "lerobot/utokyo_saytap",
    "lerobot/imperialcollege_sawyer_wrist_cam",
    "lerobot/utokyo_xarm_bimanual",
    "lerobot/tokyo_u_lsmo",
    "lerobot/utokyo_pr2_opening_fridge",
    "lerobot/cmu_franka_exploration_dataset",
    "lerobot/cmu_stretch",
    "lerobot/asu_table_top",
    "lerobot/utokyo_pr2_tabletop_manipulation",
    "lerobot/utokyo_xarm_pick_and_place",
    "lerobot/ucsd_kitchen_dataset",
    "lerobot/austin_buds_dataset",
    "lerobot/dlr_sara_grid_clamp",
    "lerobot/conq_hose_manipulation",
    "lerobot/columbia_cairlab_pusht_real",
    "lerobot/dlr_sara_pour",
    "lerobot/dlr_edan_shared_control",
    "lerobot/ucsd_pick_and_place_dataset",
    "lerobot/berkeley_cable_routing",
    "lerobot/nyu_franka_play_dataset",
    "lerobot/austin_sirius_dataset",
    "lerobot/cmu_play_fusion",
    "lerobot/berkeley_gnm_sac_son",
    "lerobot/nyu_door_opening_surprising_effectiveness",
    "lerobot/berkeley_fanuc_manipulation",
    "lerobot/jaco_play",
    "lerobot/viola",
    "lerobot/kaist_nonprehensile",
    "lerobot/berkeley_mvp",
    "lerobot/uiuc_d3field",
    "lerobot/berkeley_gnm_recon",
    "lerobot/austin_sailor_dataset",
    "lerobot/utaustin_mutex",
    "lerobot/roboturk",
    "lerobot/stanford_hydra_dataset",
    "lerobot/berkeley_autolab_ur5",
    "lerobot/stanford_robocook",
    "lerobot/toto",
    "lerobot/fmb",
    "lerobot/droid_100",
    "lerobot/berkeley_rpt",
    "lerobot/stanford_kuka_multimodal_dataset",
    "lerobot/iamlab_cmu_pickup_insert",
    "lerobot/taco_play",
    "lerobot/berkeley_gnm_cory_hall",
    "lerobot/usc_cloth_sim",
]

# 合并并排序所有可用的数据集(包括仿真和真实世界)
# Merge and sort all available datasets (including simulation and real-world)
# 类型: List[str] - 排序后的唯一数据集名称列表
# Type: List[str] - Sorted list of unique dataset names
# 使用 itertools.chain 合并环境数据集和真实世界数据集,使用 set 去重,最后排序
# Uses itertools.chain to merge environment and real-world datasets, set for deduplication, then sorts
available_datasets = sorted(
    set(itertools.chain(*available_datasets_per_env.values(), available_real_world_datasets))
)

# ================================================================================
# 可用策略 (Available Policies)
# ================================================================================
# 列出 `lerobot/common/policies` 中所有可用的策略
# Lists all available policies from `lerobot/common/policies`
# 类型: List[str] - 策略名称列表
# Type: List[str] - List of policy names
available_policies = [
    "act",        # ACT (Action Chunking Transformer) - 动作分块转换器策略
    "diffusion",  # Diffusion Policy - 扩散策略
    "tdmpc",      # TDMPC (Temporal Difference Model Predictive Control) - 时序差分模型预测控制
    "vqbet",      # VQ-BeT (Vector Quantized Behavior Transformer) - 矢量量化行为转换器
]

# ================================================================================
# 可用机器人 (Available Robots)
# ================================================================================
# 列出 `lerobot/common/robot_devices/robots` 中所有可用的机器人配置
# Lists all available robot configurations from `lerobot/common/robot_devices/robots`
# 类型: List[str] - 机器人配置名称列表
# Type: List[str] - List of robot configuration names
available_robots = [
    "koch",            # Koch 单臂机器人 / Koch single-arm robot
    "koch_bimanual",   # Koch 双臂机器人 / Koch bimanual robot
    "aloha",           # ALOHA 双臂机器人系统 / ALOHA bimanual robot system
    "so100",           # SO-100 低成本机械臂 / SO-100 low-cost robot arm
    "moss",            # MOSS 机器人平台 / MOSS robot platform
]

# ================================================================================
# 可用相机 (Available Cameras)
# ================================================================================
# 列出 `lerobot/common/robot_devices/cameras` 中所有支持的相机类型
# Lists all supported camera types from `lerobot/common/robot_devices/cameras`
# 类型: List[str] - 相机类型名称列表
# Type: List[str] - List of camera type names
available_cameras = [
    "opencv",         # OpenCV 标准相机接口 / OpenCV standard camera interface
    "intelrealsense", # Intel RealSense 深度相机 / Intel RealSense depth camera
]

# ================================================================================
# 可用电机 (Available Motors)
# ================================================================================
# 列出 `lerobot/common/robot_devices/motors` 中所有支持的电机类型
# Lists all supported motor types from `lerobot/common/robot_devices/motors`
# 类型: List[str] - 电机类型名称列表
# Type: List[str] - List of motor type names
available_motors = [
    "dynamixel",  # Dynamixel 舵机系列 / Dynamixel servo motor series
    "feetech",    # Feetech 舵机系列 / Feetech servo motor series
]

# ================================================================================
# 每个环境的可用策略 (Available Policies per Environment)
# ================================================================================
# 键和值都指向 yaml 配置文件
# Keys and values refer to yaml configuration files
# 类型: Dict[str, List[str]]
# 键 (Keys): 环境名称 (environment name)
# 值 (Values): 该环境适用的策略名称列表 (list of policy names applicable to that environment)
available_policies_per_env = {
    "aloha": ["act"],                     # ALOHA 环境使用 ACT 策略
    "pusht": ["diffusion", "vqbet"],      # PushT 环境支持扩散和 VQ-BeT 策略
    "xarm": ["tdmpc"],                    # XArm 环境使用 TDMPC 策略
    "koch_real": ["act_koch_real"],       # 真实 Koch 机器人的 ACT 配置
    "aloha_real": ["act_aloha_real"],     # 真实 ALOHA 机器人的 ACT 配置
}

# ================================================================================
# 组合数据结构 (Combination Data Structures)
# ================================================================================
# 以下变量用于测试和迭代所有可能的组合
# The following variables are used for testing and iterating over all possible combinations

# 环境-任务对列表 / List of environment-task pairs
# 类型: List[Tuple[str, str]] - [(环境名, 任务名), ...]
# Type: List[Tuple[str, str]] - [(env_name, task_name), ...]
env_task_pairs = [(env, task) for env, tasks in available_tasks_per_env.items() for task in tasks]

# 环境-数据集对列表 / List of environment-dataset pairs
# 类型: List[Tuple[str, str]] - [(环境名, 数据集名), ...]
# Type: List[Tuple[str, str]] - [(env_name, dataset_name), ...]
env_dataset_pairs = [
    (env, dataset) for env, datasets in available_datasets_per_env.items() for dataset in datasets
]

# 环境-数据集-策略三元组列表 / List of environment-dataset-policy triplets
# 类型: List[Tuple[str, str, str]] - [(环境名, 数据集名, 策略名), ...]
# Type: List[Tuple[str, str, str]] - [(env_name, dataset_name, policy_name), ...]
# 用于组合测试:在特定环境中使用特定数据集训练特定策略
# Used for combination testing: training a specific policy with a specific dataset in a specific environment
env_dataset_policy_triplets = [
    (env, dataset, policy)
    for env, datasets in available_datasets_per_env.items()
    for dataset in datasets
    for policy in available_policies_per_env[env]
]
