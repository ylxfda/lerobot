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
LeRobot 数据集核心模块 (LeRobot Dataset Core Module)

功能说明 (Functionality):
    这是 LeRobot 库的核心数据集模块,提供了机器人学习数据集的完整管理功能。
    主要功能包括:
    1. 从 HuggingFace Hub 或本地加载数据集
    2. 支持视频压缩存储(MP4)和高效解码
    3. Episode(回合)级别的数据组织和索引
    4. 时序帧选择和多帧观测支持
    5. 数据统计计算和归一化
    6. 实时数据采集和在线缓冲
    7. 与 PyTorch DataLoader 的无缝集成

    This is the core dataset module of the LeRobot library, providing complete management
    functionality for robot learning datasets. Main features include:
    1. Loading datasets from HuggingFace Hub or local storage
    2. Support for video compression (MP4) and efficient decoding
    3. Episode-level data organization and indexing
    4. Temporal frame selection and multi-frame observation support
    5. Data statistics computation and normalization
    6. Real-time data collection and online buffering
    7. Seamless integration with PyTorch DataLoader

主要类 (Main Classes):
    LeRobotDatasetMetadata: 数据集元数据管理类 / Dataset metadata management class
    LeRobotDataset: 主数据集类,支持离线和在线模式 / Main dataset class supporting offline and online modes

数据格式 (Data Format):
    - Parquet 文件: 存储结构化数据(状态、动作、元数据)
    - MP4 视频: 存储图像观测(高压缩比)
    - JSON 文件: 存储元数据、任务信息、统计数据
    - Parquet files: Store structured data (states, actions, metadata)
    - MP4 videos: Store image observations (high compression ratio)
    - JSON files: Store metadata, task information, statistics

版本信息 (Version):
    CODEBASE_VERSION: v2.1 - 当前代码库版本 / Current codebase version
"""

import contextlib
import logging
import shutil
from pathlib import Path
from typing import Callable

import datasets
import numpy as np
import packaging.version
import PIL.Image
import torch
import torch.utils
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.errors import RevisionNotFoundError

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.common.datasets.image_writer import AsyncImageWriter, write_image
from lerobot.common.datasets.utils import (
    DEFAULT_FEATURES,
    DEFAULT_IMAGE_PATH,
    INFO_PATH,
    TASKS_PATH,
    append_jsonlines,
    backward_compatible_episodes_stats,
    check_delta_timestamps,
    check_timestamps_sync,
    check_version_compatibility,
    create_empty_dataset_info,
    create_lerobot_dataset_card,
    embed_images,
    get_delta_indices,
    get_episode_data_index,
    get_features_from_robot,
    get_hf_features_from_features,
    get_safe_version,
    hf_transform_to_torch,
    is_valid_version,
    load_episodes,
    load_episodes_stats,
    load_info,
    load_stats,
    load_tasks,
    validate_episode_buffer,
    validate_frame,
    write_episode,
    write_episode_stats,
    write_info,
    write_json,
)
from lerobot.common.datasets.video_utils import (
    VideoFrame,
    decode_video_frames,
    encode_video_frames,
    get_safe_default_codec,
    get_video_info,
)
from lerobot.common.robot_devices.robots.utils import Robot

# 当前代码库版本 / Current codebase version
# 用于版本兼容性检查和数据集格式验证
# Used for version compatibility checking and dataset format validation
CODEBASE_VERSION = "v2.1"


class LeRobotDatasetMetadata:
    """
    LeRobot 数据集元数据管理类 (LeRobot Dataset Metadata Management Class)

    功能说明 (Functionality):
        管理数据集的元数据,包括版本信息、任务定义、Episode 索引和统计数据。
        负责从本地或 HuggingFace Hub 加载和同步元数据。

        Manages dataset metadata including version information, task definitions,
        episode indices, and statistics. Handles loading and syncing metadata
        from local storage or HuggingFace Hub.

    主要属性 (Main Attributes):
        repo_id (str): HuggingFace 仓库 ID / HuggingFace repository ID
        revision (str): 数据集版本/分支 / Dataset version/branch
        root (Path): 数据集本地根目录 / Local root directory of dataset
        info (dict): 数据集基本信息(名称、作者、FPS等) / Basic dataset info
        tasks (list): 任务列表 / List of tasks
        episodes (list): Episode 元数据列表 / List of episode metadata
        stats (dict): 聚合统计数据 / Aggregated statistics
        episodes_stats (dict): 每个 Episode 的统计数据 / Per-episode statistics

    使用示例 (Usage Example):
        ```python
        # 创建元数据对象并从 Hub 加载
        # Create metadata object and load from Hub
        metadata = LeRobotDatasetMetadata(
            repo_id="lerobot/pusht",
            root="/path/to/local/cache",
            revision="v2.1"
        )

        # 访问数据集信息
        # Access dataset information
        print(metadata.info['fps'])  # 帧率 / Frame rate
        print(metadata.episodes)     # Episode 列表 / Episode list
        ```
    """
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
    ):
        """
        初始化数据集元数据对象 (Initialize Dataset Metadata Object)

        参数说明 (Parameters):
            repo_id (str): HuggingFace 数据集仓库 ID,格式: "组织/数据集名"
                          例如: "lerobot/pusht"
                          HuggingFace dataset repository ID, format: "org/dataset_name"
                          e.g., "lerobot/pusht"

            root (str | Path | None): 数据集本地存储根目录
                                     如果为 None,使用默认缓存路径: HF_LEROBOT_HOME / repo_id
                                     Local storage root directory for dataset
                                     If None, uses default cache path: HF_LEROBOT_HOME / repo_id

            revision (str | None): 数据集版本标签或分支名
                                  如果为 None,使用当前代码库版本(CODEBASE_VERSION)
                                  Dataset version tag or branch name
                                  If None, uses current codebase version (CODEBASE_VERSION)

            force_cache_sync (bool): 是否强制从远程仓库同步缓存
                                    True 时会重新下载元数据
                                    Whether to force cache sync from remote repository
                                    When True, metadata will be re-downloaded

        内部流程 (Internal Flow):
            1. 设置仓库 ID、版本和根目录路径
            2. 尝试从本地加载元数据
            3. 如果本地不存在或强制同步,从 Hub 下载元数据
            4. 加载所有元数据(info, tasks, episodes, stats)
        """
        # 保存基本配置 / Save basic configuration
        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        try:
            # 如果强制同步,跳过本地加载 / If force sync, skip local loading
            if force_cache_sync:
                raise FileNotFoundError
            # 尝试从本地加载元数据 / Try to load metadata from local storage
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            # 本地不存在,需要从远程下载 / Local not found, need to download from remote
            if is_valid_version(self.revision):
                # 获取安全的版本号(处理版本兼容性) / Get safe version (handle version compatibility)
                self.revision = get_safe_version(self.repo_id, self.revision)

            # 创建元数据目录 / Create metadata directory
            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            # 从仓库拉取元数据文件 / Pull metadata files from repository
            self.pull_from_repo(allow_patterns="meta/")
            # 加载已下载的元数据 / Load downloaded metadata
            self.load_metadata()

    def load_metadata(self):
        """
        加载所有元数据文件 (Load All Metadata Files)

        功能说明 (Functionality):
            从本地存储加载数据集的所有元数据,包括:
            1. info: 基本信息(FPS、机器人类型、任务等)
            2. tasks: 任务定义列表
            3. episodes: Episode 元数据(索引、长度等)
            4. stats/episodes_stats: 统计数据(均值、标准差等)

            Loads all dataset metadata from local storage, including:
            1. info: Basic information (FPS, robot type, tasks, etc.)
            2. tasks: Task definition list
            3. episodes: Episode metadata (indices, lengths, etc.)
            4. stats/episodes_stats: Statistics (mean, std, etc.)

        版本兼容性 (Version Compatibility):
            v2.1 之前: 使用聚合的 stats 文件
            v2.1 及之后: 使用 per-episode 的 episodes_stats 文件
            Before v2.1: Uses aggregated stats file
            v2.1 and later: Uses per-episode episodes_stats file

        副作用 (Side Effects):
            设置以下实例属性:
            - self.info: dict
            - self.tasks: list
            - self.task_to_task_index: dict
            - self.episodes: list
            - self.stats: dict
            - self.episodes_stats: dict
        """
        # 加载基本信息 / Load basic information
        self.info = load_info(self.root)
        # 检查版本兼容性 / Check version compatibility
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        # 加载任务定义 / Load task definitions
        self.tasks, self.task_to_task_index = load_tasks(self.root)
        # 加载 Episode 索引 / Load episode indices
        self.episodes = load_episodes(self.root)

        # 根据版本加载统计数据 / Load statistics based on version
        if self._version < packaging.version.parse("v2.1"):
            # 旧版本:加载聚合统计并转换为 per-episode 格式
            # Old version: Load aggregated stats and convert to per-episode format
            self.stats = load_stats(self.root)
            self.episodes_stats = backward_compatible_episodes_stats(self.stats, self.episodes)
        else:
            # 新版本:加载 per-episode 统计并聚合
            # New version: Load per-episode stats and aggregate
            self.episodes_stats = load_episodes_stats(self.root)
            self.stats = aggregate_stats(list(self.episodes_stats.values()))

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        """
        从 HuggingFace Hub 拉取数据集文件 (Pull Dataset Files from HuggingFace Hub)

        功能说明 (Functionality):
            使用 HuggingFace Hub 的 snapshot_download 功能下载数据集文件到本地。
            支持模式匹配来选择性下载特定文件。

            Uses HuggingFace Hub's snapshot_download to download dataset files locally.
            Supports pattern matching for selective file downloading.

        参数说明 (Parameters):
            allow_patterns (list[str] | str | None): 允许下载的文件模式列表
                                                     例如: ["meta/", "data/"] 只下载这些目录
                                                     List of file patterns to allow for download
                                                     e.g., ["meta/", "data/"] downloads only these dirs

            ignore_patterns (list[str] | str | None): 忽略的文件模式列表
                                                      例如: ["*.mp4"] 跳过所有视频文件
                                                      List of file patterns to ignore
                                                      e.g., ["*.mp4"] skips all video files

        返回值 (Returns):
            None: 文件直接下载到 self.root 目录
                  Files are downloaded directly to self.root directory

        使用示例 (Usage Example):
            # 只下载元数据 / Download only metadata
            metadata.pull_from_repo(allow_patterns="meta/")

            # 下载所有数据但跳过视频 / Download all data but skip videos
            metadata.pull_from_repo(ignore_patterns="*.mp4")
        """
        snapshot_download(
            self.repo_id,              # 仓库 ID / Repository ID
            repo_type="dataset",       # 仓库类型:数据集 / Repository type: dataset
            revision=self.revision,    # 版本/分支 / Version/branch
            local_dir=self.root,       # 本地保存目录 / Local save directory
            allow_patterns=allow_patterns,    # 允许的模式 / Allowed patterns
            ignore_patterns=ignore_patterns,  # 忽略的模式 / Ignored patterns
        )

    @property
    def _version(self) -> packaging.version.Version:
        """
        获取创建此数据集时使用的代码库版本 (Get Codebase Version Used to Create This Dataset)

        返回值 (Returns):
            packaging.version.Version: 解析后的版本对象,例如 Version('2.1')
                                      Parsed version object, e.g., Version('2.1')
        """
        return packaging.version.parse(self.info["codebase_version"])

    def get_data_file_path(self, ep_index: int) -> Path:
        """
        获取指定 Episode 的数据文件路径 (Get Data File Path for Specified Episode)

        参数说明 (Parameters):
            ep_index (int): Episode 索引号,从 0 开始
                           Episode index number, starting from 0

        返回值 (Returns):
            Path: Parquet 数据文件的完整路径
                 例如: "data/chunk-000/episode_000042.parquet"
                 Full path to the Parquet data file
                 e.g., "data/chunk-000/episode_000042.parquet"

        实现逻辑 (Implementation Logic):
            1. 根据 Episode 索引计算所属的 chunk 编号
            2. 使用模板字符串替换占位符生成文件路径
        """
        # 计算 Episode 所属的 chunk / Calculate chunk for this episode
        ep_chunk = self.get_episode_chunk(ep_index)
        # 格式化路径字符串 / Format path string
        fpath = self.data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        """
        获取指定 Episode 和相机的视频文件路径 (Get Video File Path for Specified Episode and Camera)

        参数说明 (Parameters):
            ep_index (int): Episode 索引号,从 0 开始
                           Episode index number, starting from 0

            vid_key (str): 视频/相机键名,例如 "observation.images.top"
                          Video/camera key name, e.g., "observation.images.top"

        返回值 (Returns):
            Path: MP4 视频文件的完整路径
                 例如: "videos/chunk-000/observation.images.top_episode_000042.mp4"
                 Full path to the MP4 video file
                 e.g., "videos/chunk-000/observation.images.top_episode_000042.mp4"

        实现逻辑 (Implementation Logic):
            1. 根据 Episode 索引计算所属的 chunk 编号
            2. 使用模板字符串替换占位符(chunk, video_key, episode_index)生成文件路径
        """
        # 计算 Episode 所属的 chunk / Calculate chunk for this episode
        ep_chunk = self.get_episode_chunk(ep_index)
        # 格式化视频路径字符串 / Format video path string
        fpath = self.video_path.format(episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index)
        return Path(fpath)

    def get_episode_chunk(self, ep_index: int) -> int:
        """
        计算 Episode 所属的 chunk 编号 (Calculate Chunk Number for Episode)

        功能说明 (Functionality):
            为了管理大型数据集,Episodes 被分组到多个 chunks 中。
            每个 chunk 包含固定数量的 Episodes (由 chunks_size 定义)。

            To manage large datasets, episodes are grouped into chunks.
            Each chunk contains a fixed number of episodes (defined by chunks_size).

        参数说明 (Parameters):
            ep_index (int): Episode 索引号,从 0 开始
                           Episode index number, starting from 0

        返回值 (Returns):
            int: Chunk 编号,从 0 开始
                例如: 如果 chunks_size=100, ep_index=250 -> 返回 2
                Chunk number, starting from 0
                e.g., if chunks_size=100, ep_index=250 -> returns 2

        示例 (Example):
            chunks_size = 100  # 每个 chunk 包含 100 个 episodes
            get_episode_chunk(0)   -> 0    # Episode 0 在 chunk 0
            get_episode_chunk(99)  -> 0    # Episode 99 在 chunk 0
            get_episode_chunk(100) -> 1    # Episode 100 在 chunk 1
            get_episode_chunk(250) -> 2    # Episode 250 在 chunk 2
        """
        return ep_index // self.chunks_size

    @property
    def data_path(self) -> str:
        """
        Parquet 数据文件的路径模板 (Path Template for Parquet Data Files)

        返回值 (Returns):
            str: 可格式化的路径字符串,包含占位符如 {episode_chunk}, {episode_index}
                例如: "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
                Formattable path string with placeholders like {episode_chunk}, {episode_index}
                e.g., "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        """
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        """
        视频文件的路径模板 (Path Template for Video Files)

        返回值 (Returns):
            str | None: 可格式化的视频路径字符串,或 None(如果数据集不包含视频)
                       例如: "videos/chunk-{episode_chunk:03d}/{video_key}_episode_{episode_index:06d}.mp4"
                       Formattable video path string, or None if dataset contains no videos
                       e.g., "videos/chunk-{episode_chunk:03d}/{video_key}_episode_{episode_index:06d}.mp4"
        """
        return self.info["video_path"]

    @property
    def robot_type(self) -> str | None:
        """
        记录数据集时使用的机器人类型 (Robot Type Used in Recording This Dataset)

        返回值 (Returns):
            str | None: 机器人类型标识符,例如 "aloha", "koch", "so100"
                       或 None(如果未指定)
                       Robot type identifier, e.g., "aloha", "koch", "so100"
                       or None if not specified
        """
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        """
        数据采集时使用的帧率 (Frames Per Second Used During Data Collection)

        返回值 (Returns):
            int: 每秒帧数,例如 30, 50, 60
                通常用于时序对齐和播放
                Frames per second, e.g., 30, 50, 60
                Typically used for temporal alignment and playback
        """
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        """
        数据集包含的所有特征定义 (All Feature Definitions Contained in the Dataset)

        返回值 (Returns):
            dict[str, dict]: 特征字典,键为特征名,值为特征元数据
                           例如: {
                               "observation.state": {
                                   "dtype": "float32",
                                   "shape": [14],
                                   "names": ["joint_0", "joint_1", ...]
                               },
                               "action": {"dtype": "float32", "shape": [14]},
                               "observation.images.top": {"dtype": "video", "shape": [3, 480, 640]}
                           }
                           Feature dictionary, keys are feature names, values are feature metadata
        """
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        """
        以图像格式存储的视觉模态的键名列表 (Keys to Access Visual Modalities Stored as Images)

        返回值 (Returns):
            list[str]: 图像特征键名列表
                      例如: ["observation.image"] (如果图像存储为 PNG/JPEG)
                      List of image feature key names
                      e.g., ["observation.image"] (if images stored as PNG/JPEG)

        注意 (Note):
            大多数现代数据集使用视频格式(MP4)存储图像以节省空间
            Most modern datasets use video format (MP4) to store images for space efficiency
        """
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        """
        以视频格式存储的视觉模态的键名列表 (Keys to Access Visual Modalities Stored as Videos)

        返回值 (Returns):
            list[str]: 视频特征键名列表
                      例如: ["observation.images.top", "observation.images.wrist"]
                      List of video feature key names
                      e.g., ["observation.images.top", "observation.images.wrist"]

        注意 (Note):
            视频格式(MP4)比独立图像文件更节省存储空间,是推荐的图像存储方式
            Video format (MP4) is more space-efficient than individual image files
        """
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        """
        所有视觉模态的键名列表(不考虑存储方法) (Keys to Access Visual Modalities Regardless of Storage Method)

        返回值 (Returns):
            list[str]: 所有相机/图像特征键名列表,包括 image 和 video 类型
                      例如: ["observation.images.top", "observation.images.wrist", "observation.image"]
                      List of all camera/image feature key names, including both image and video types
                      e.g., ["observation.images.top", "observation.images.wrist", "observation.image"]
        """
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> dict[str, list | dict]:
        """
        向量模态各维度的名称 (Names of the Various Dimensions of Vector Modalities)

        返回值 (Returns):
            dict[str, list | dict]: 特征名到维度名称的映射
                                   例如: {
                                       "observation.state": ["joint_0", "joint_1", ..., "joint_13"],
                                       "action": ["joint_0", "joint_1", ..., "joint_13"]
                                   }
                                   Mapping from feature names to dimension names
                                   Useful for understanding what each dimension represents
        """
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        """
        不同特征的形状 (Shapes for the Different Features)

        返回值 (Returns):
            dict: 特征名到形状元组的映射
                 例如: {
                     "observation.state": (14,),          # 14维状态向量 / 14-dim state vector
                     "action": (14,),                     # 14维动作向量 / 14-dim action vector
                     "observation.images.top": (3, 480, 640)  # RGB图像 / RGB image
                 }
                 Mapping from feature names to shape tuples
        """
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        """
        可用的 Episode 总数 (Total Number of Episodes Available)

        返回值 (Returns):
            int: Episode 总数,例如 500
                Total number of episodes, e.g., 500
        """
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        """
        数据集中保存的帧总数 (Total Number of Frames Saved in This Dataset)

        返回值 (Returns):
            int: 所有 Episodes 的总帧数,例如 50000
                Total frames across all episodes, e.g., 50000
        """
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        """
        数据集中执行的不同任务总数 (Total Number of Different Tasks Performed in This Dataset)

        返回值 (Returns):
            int: 任务类型数量,例如 1 (单任务) 或 5 (多任务)
                Number of task types, e.g., 1 (single task) or 5 (multi-task)
        """
        return self.info["total_tasks"]

    @property
    def total_chunks(self) -> int:
        """
        Chunk 总数(Episode 组) (Total Number of Chunks - Groups of Episodes)

        返回值 (Returns):
            int: Chunk 数量,例如 如果有 500 episodes 且 chunks_size=100,则返回 5
                Number of chunks, e.g., if 500 episodes with chunks_size=100, returns 5
        """
        return self.info["total_chunks"]

    @property
    def chunks_size(self) -> int:
        """
        每个 Chunk 的最大 Episode 数 (Max Number of Episodes Per Chunk)

        返回值 (Returns):
            int: Chunk 大小,通常为 100 或 1000
                用于将大型数据集分割成可管理的部分
                Chunk size, typically 100 or 1000
                Used to split large datasets into manageable parts
        """
        return self.info["chunks_size"]

    def get_task_index(self, task: str) -> int | None:
        """
        根据自然语言任务描述获取任务索引 (Get Task Index from Natural Language Task Description)

        功能说明 (Functionality):
            给定一个任务的自然语言描述,如果该任务已存在于数据集中,返回其 task_index,
            否则返回 None。用于多任务数据集中的任务查找。

            Given a task in natural language, returns its task_index if the task already
            exists in the dataset, otherwise return None. Used for task lookup in multi-task datasets.

        参数说明 (Parameters):
            task (str): 任务的自然语言描述
                       例如: "Pick up the cube", "Insert the peg"
                       Natural language task description
                       e.g., "Pick up the cube", "Insert the peg"

        返回值 (Returns):
            int | None: 任务索引(从 0 开始),如果任务不存在则返回 None
                       Task index (starting from 0), or None if task doesn't exist
        """
        return self.task_to_task_index.get(task, None)

    def add_task(self, task: str):
        """
        Given a task in natural language, add it to the dictionary of tasks.
        """
        if task in self.task_to_task_index:
            raise ValueError(f"The task '{task}' already exists and can't be added twice.")

        task_index = self.info["total_tasks"]
        self.task_to_task_index[task] = task_index
        self.tasks[task_index] = task
        self.info["total_tasks"] += 1

        task_dict = {
            "task_index": task_index,
            "task": task,
        }
        append_jsonlines(task_dict, self.root / TASKS_PATH)

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length

        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1

        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] += len(self.video_keys)
        if len(self.video_keys) > 0:
            self.update_video_info()

        write_info(self.info, self.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)

        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.root)

    def update_video_info(self) -> None:
        """
        Warning: this function writes info from first episode videos, implicitly assuming that all videos have
        been encoded the same way. Also, this means it assumes the first episode exists.
        """
        for key in self.video_keys:
            if not self.features[key].get("info", None):
                video_path = self.root / self.get_video_file_path(ep_index=0, vid_key=key)
                self.info["features"][key]["info"] = get_video_info(video_path)

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Total episodes: '{self.total_episodes}',\n"
            f"    Total frames: '{self.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        root: str | Path | None = None,
        robot: Robot | None = None,
        robot_type: str | None = None,
        features: dict | None = None,
        use_videos: bool = True,
    ) -> "LeRobotDatasetMetadata":
        """Creates metadata for a LeRobotDataset."""
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        obj.root.mkdir(parents=True, exist_ok=False)

        if robot is not None:
            features = get_features_from_robot(robot, use_videos)
            robot_type = robot.robot_type
            if not all(cam.fps == fps for cam in robot.cameras.values()):
                logging.warning(
                    f"Some cameras in your {robot.robot_type} robot don't have an fps matching the fps of your dataset."
                    "In this case, frames from lower fps cameras will be repeated to fill in the blanks."
                )
        elif features is None:
            raise ValueError(
                "Dataset features must either come from a Robot or explicitly passed upon creation."
            )
        else:
            # TODO(aliberts, rcadene): implement sanity check for features
            features = {**features, **DEFAULT_FEATURES}

            # check if none of the features contains a "/" in their names,
            # as this would break the dict flattening in the stats computation, which uses '/' as separator
            for key in features:
                if "/" in key:
                    raise ValueError(f"Feature names should not contain '/'. Found '/' in feature '{key}'.")

            features = {**features, **DEFAULT_FEATURES}

        obj.tasks, obj.task_to_task_index = {}, {}
        obj.episodes_stats, obj.stats, obj.episodes = {}, {}, {}
        obj.info = create_empty_dataset_info(CODEBASE_VERSION, fps, robot_type, features, use_videos)
        if len(obj.video_keys) > 0 and not use_videos:
            raise ValueError()
        write_json(obj.info, obj.root / INFO_PATH)
        obj.revision = None
        return obj


class LeRobotDataset(torch.utils.data.Dataset):
    """
    LeRobot 主数据集类 (LeRobot Main Dataset Class)

    功能说明 (Functionality):
        这是 LeRobot 的核心数据集类,继承自 PyTorch 的 Dataset。
        提供完整的机器人学习数据集管理功能,包括:
        - 从 HuggingFace Hub 或本地加载数据集
        - 支持视频压缩和高效解码
        - 时序帧选择(用于多帧策略)
        - 图像变换和数据增强
        - Episode 级别的数据索引
        - 与 PyTorch DataLoader 无缝集成

        This is the core dataset class of LeRobot, inheriting from PyTorch's Dataset.
        Provides complete robot learning dataset management functionality, including:
        - Loading datasets from HuggingFace Hub or local storage
        - Video compression and efficient decoding support
        - Temporal frame selection (for multi-frame policies)
        - Image transforms and data augmentation
        - Episode-level data indexing
        - Seamless integration with PyTorch DataLoader

    主要属性 (Main Attributes):
        repo_id (str): 数据集仓库 ID / Dataset repository ID
        root (Path): 本地存储根目录 / Local storage root directory
        meta (LeRobotDatasetMetadata): 元数据对象 / Metadata object
        hf_dataset (datasets.Dataset): HuggingFace Dataset 对象 / HuggingFace Dataset object
        stats (dict): 数据统计信息(均值、标准差等) / Data statistics (mean, std, etc.)
        episode_data_index (dict): Episode 到帧索引的映射 / Mapping from episodes to frame indices

    使用示例 (Usage Example):
        ```python
        # 从 Hub 加载数据集
        # Load dataset from Hub
        dataset = LeRobotDataset("lerobot/pusht")

        # 只加载特定 episodes
        # Load only specific episodes
        dataset = LeRobotDataset("lerobot/pusht", episodes=[0, 1, 2])

        # 添加图像变换
        # Add image transforms
        from torchvision.transforms import v2
        transforms = v2.Compose([
            v2.RandomCrop(size=(84, 84)),
            v2.RandomHorizontalFlip(p=0.5),
        ])
        dataset = LeRobotDataset("lerobot/pusht", image_transforms=transforms)

        # 用于 DataLoader
        # Use with DataLoader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4
        )
        ```
    """
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        """
        初始化 LeRobotDataset 对象 (Initialize LeRobotDataset Object)

        ========================================================================
        使用模式 (Usage Modes)
        ========================================================================

        根据不同的使用场景,有 2 种实例化模式:
        2 modes are available for instantiating this class, depending on 2 different use cases:

        模式 1: 数据集已存在 (Mode 1: Dataset Already Exists)
        ------------------------------------------------
            1a. 在本地磁盘的 'root' 文件夹中
                - 典型场景:您在本地录制了数据集,可能还未推送到 Hub
                - 使用 'root' 参数实例化将直接从磁盘加载数据集
                - 可以离线工作(无需网络连接)

                On your local disk in the 'root' folder. This is typically the case when you recorded your
                dataset locally and you may or may not have pushed it to the hub yet. Instantiating this class
                with 'root' will load your dataset directly from disk. This can happen while you're offline (no
                internet connection).

            1b. 在 Hugging Face Hub 上,地址为 https://huggingface.co/datasets/{repo_id}
                - 本地磁盘 'root' 文件夹中不存在
                - 使用 'repo_id' 实例化将从 Hub 下载数据集
                - 要求数据集符合 codebase_version v2.0
                - 如果数据集是旧格式创建的,需要先转换
                  (使用 lerobot/common/datasets/v2/convert_dataset_v1_to_v2.py)

                On the Hugging Face Hub at the address https://huggingface.co/datasets/{repo_id} and not on
                your local disk in the 'root' folder. Instantiating this class with this 'repo_id' will download
                the dataset from that address and load it, pending your dataset is compliant with
                codebase_version v2.0. If your dataset has been created before this new format, you will be
                prompted to convert it using our conversion script from v1.6 to v2.0, which you can find at
                lerobot/common/datasets/v2/convert_dataset_v1_to_v2.py.

        模式 2: 数据集不存在 (Mode 2: Dataset Doesn't Exist)
        --------------------------------------------------
            - 本地磁盘或 Hub 上都不存在
            - 可以使用 'create' 类方法创建空的 LeRobotDataset
            - 用途:录制新数据集或将现有数据集转换为 LeRobotDataset 格式

            Your dataset doesn't already exists (either on local disk or on the Hub): you can create an empty
            LeRobotDataset with the 'create' classmethod. This can be used for recording a dataset or port an
            existing dataset to the LeRobotDataset format.


        ========================================================================
        文件结构 (File Structure)
        ========================================================================

        从文件角度来看,LeRobotDataset 封装了 3 个主要部分:
        In terms of files, LeRobotDataset encapsulates 3 main things:

            1. 元数据 (Metadata):
                - info: 包含数据集的各种信息,如形状、键名、FPS等
                        Contains various information about the dataset like shapes, keys, fps etc.
                - stats: 存储不同模态的数据集统计信息,用于归一化
                        Stores the dataset statistics of the different modalities for normalization
                - tasks: 包含数据集每个任务的提示词,可用于任务条件训练
                        Contains the prompts for each task of the dataset, which can be used for
                        task-conditioned training

            2. hf_dataset (来自 datasets.Dataset):
                从 parquet 文件读取所有数值数据
                Reads any values from parquet files

            3. videos (可选):
                加载帧以与 parquet 文件中的数据同步
                Frames are loaded to be synchronous with data from parquet files

        典型的 LeRobotDataset 目录结构:
        A typical LeRobotDataset looks like this from its root path:
        .
        ├── data
        │   ├── chunk-000
        │   │   ├── episode_000000.parquet
        │   │   ├── episode_000001.parquet
        │   │   ├── episode_000002.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── episode_001000.parquet
        │   │   ├── episode_001001.parquet
        │   │   ├── episode_001002.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes.jsonl
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.jsonl
        └── videos
            ├── chunk-000
            │   ├── observation.images.laptop
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            │   ├── observation.images.phone
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            ├── chunk-001
            └── ...

        注意事项 (Notes):
            这种基于文件的结构设计尽可能通用。按 Episode 分割文件,
            允许更细粒度地控制使用和下载哪些 Episodes。
            数据集的结构完全由 info.json 文件描述,可以在下载实际数据之前
            轻松下载或直接在 Hub 上查看。
            使用的文件类型非常简单,不需要复杂工具即可读取,
            仅使用 .parquet、.json 和 .mp4 文件(以及 README 的 .md 文件)。

            Note that this file-based structure is designed to be as versatile as possible. The files are split by
            episodes which allows a more granular control over which episodes one wants to use and download. The
            structure of the dataset is entirely described in the info.json file, which can be easily downloaded
            or viewed directly on the hub before downloading any actual data. The type of files used are very
            simple and do not need complex tools to be read, it only uses .parquet, .json and .mp4 files (and .md
            for the README).

        ========================================================================
        参数说明 (Parameters)
        ========================================================================

        Args:
            repo_id (str):
                用于获取数据集的仓库 ID。本地存储路径为 root/repo_id。
                例如: "lerobot/pusht", "lerobot/aloha_sim_insertion_human"
                This is the repo id that will be used to fetch the dataset. Locally, the dataset
                will be stored under root/repo_id.

            root (Path | None, optional):
                用于下载/写入文件的本地目录。也可以通过设置 LEROBOT_HOME 环境变量指向不同位置。
                默认值: '~/.cache/huggingface/lerobot'
                Local directory to use for downloading/writing files. You can also
                set the LEROBOT_HOME environment variable to point to a different location. Defaults to
                '~/.cache/huggingface/lerobot'.

            episodes (list[int] | None, optional):
                如果指定,则只加载此列表中 episode_index 指定的 Episodes。
                例如: [0, 1, 2, 10, 20] 只加载这 5 个 episodes
                默认值: None (加载所有 episodes)
                If specified, this will only load episodes specified by
                their episode_index in this list. Defaults to None.

            image_transforms (Callable | None, optional):
                可以传入 torchvision.transforms.v2 的标准 v2 图像变换,
                将应用于所有视觉模态(无论来自视频还是图像)。
                例如: RandomCrop, RandomHorizontalFlip, ColorJitter等
                默认值: None (不应用变换)
                You can pass standard v2 image transforms from
                torchvision.transforms.v2 here which will be applied to visual modalities (whether they come
                from videos or images). Defaults to None.

            delta_timestamps (dict[list[float]] | None, optional):
                时序偏移字典,用于多帧观测。
                键为特征名,值为相对于当前帧的时间偏移列表(秒)。
                例如: {"observation.images.top": [-0.1, 0.0, 0.1]} 表示获取前一帧、当前帧、后一帧
                用于时序策略,如 ACT 需要多帧历史观测
                默认值: None (只获取当前帧)
                Temporal offset dictionary for multi-frame observations.
                Keys are feature names, values are lists of time offsets (in seconds) relative to current frame.
                e.g., {"observation.images.top": [-0.1, 0.0, 0.1]} gets previous, current, and next frame
                Used for temporal policies like ACT that need multi-frame history observations
                Defaults to None.

            tolerance_s (float, optional):
                时间戳同步容差(秒)。用于确保数据时间戳实际与 fps 值同步。
                在数据集初始化时,确保每个时间戳与下一个的间隔为 1/fps ± tolerance_s。
                这也适用于从视频文件解码的帧。还用于检查 `delta_timestamps` 是否为 1/fps 的倍数。
                默认值: 1e-4 (0.0001秒)
                Tolerance in seconds used to ensure data timestamps are actually in
                sync with the fps value. It is used at the init of the dataset to make sure that each
                timestamps is separated to the next by 1/fps +/- tolerance_s. This also applies to frames
                decoded from video files. It is also used to check that `delta_timestamps` (when provided) are
                multiples of 1/fps. Defaults to 1e-4.

            revision (str, optional):
                可选的 Git 修订版本 ID,可以是分支名、标签或提交哈希。
                例如: "main", "v2.0", "abc123def"
                默认值: 当前代码库版本标签
                An optional Git revision id which can be a branch name, a tag, or a
                commit hash. Defaults to current codebase version tag.

            force_cache_sync (bool, optional):
                是否先同步和刷新本地文件。
                如果为 True 且文件已在本地缓存中,会更快。
                但是,加载的文件可能与 Hub 上的版本不同步,特别是如果指定了 'revision'。
                默认值: False
                Flag to sync and refresh local files first. If True and files
                are already present in the local cache, this will be faster. However, files loaded might not
                be in sync with the version on the hub, especially if you specified 'revision'. Defaults to
                False.

            download_videos (bool, optional):
                是否下载视频。
                注意:设为 True 但视频文件已在本地磁盘上时,不会重新下载。
                默认值: True
                Flag to download the videos. Note that when set to True but the
                video files are already present on local disk, they won't be downloaded again. Defaults to
                True.

            video_backend (str | None, optional):
                用于解码视频的后端。
                默认: 如果平台可用则使用 'torchcodec',否则使用 'pyav'
                其他选项: 'pyav' (Torchvision 使用的解码器,曾是默认选项)
                         'video_reader' (Torchvision 的另一个解码器)
                Video backend to use for decoding videos. Defaults to torchcodec when available on the platform;
                otherwise, defaults to 'pyav'. You can also use the 'pyav' decoder used by Torchvision, which
                used to be the default option, or 'video_reader' which is another decoder of Torchvision.
        """
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.delta_indices = None

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)

        # Load actual data
        try:
            if force_cache_sync:
                raise FileNotFoundError
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        # Check timestamps
        timestamps = torch.stack(self.hf_dataset["timestamp"]).numpy()
        episode_indices = torch.stack(self.hf_dataset["episode_index"]).numpy()
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def push_to_hub(
        self,
        branch: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        tag_version: bool = True,
        push_videos: bool = True,
        private: bool = False,
        allow_patterns: list[str] | str | None = None,
        upload_large_folder: bool = False,
        **card_kwargs,
    ) -> None:
        ignore_patterns = ["images/"]
        if not push_videos:
            ignore_patterns.append("videos/")

        hub_api = HfApi()
        hub_api.create_repo(
            repo_id=self.repo_id,
            private=private,
            repo_type="dataset",
            exist_ok=True,
        )
        if branch:
            hub_api.create_branch(
                repo_id=self.repo_id,
                branch=branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )

        upload_kwargs = {
            "repo_id": self.repo_id,
            "folder_path": self.root,
            "repo_type": "dataset",
            "revision": branch,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        if upload_large_folder:
            hub_api.upload_large_folder(**upload_kwargs)
        else:
            hub_api.upload_folder(**upload_kwargs)

        if not hub_api.file_exists(self.repo_id, REPOCARD_NAME, repo_type="dataset", revision=branch):
            card = create_lerobot_dataset_card(
                tags=tags, dataset_info=self.meta.info, license=license, **card_kwargs
            )
            card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=branch)

        if tag_version:
            with contextlib.suppress(RevisionNotFoundError):
                hub_api.delete_tag(self.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
            hub_api.create_tag(self.repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    def download_episodes(self, download_videos: bool = True) -> None:
        """Downloads the dataset from the given 'repo_id' at the provided version. If 'episodes' is given, this
        will only download those episodes (selected by their episode_index). If 'episodes' is None, the whole
        dataset will be downloaded. Thanks to the behavior of snapshot_download, if the files are already present
        in 'local_dir', they won't be downloaded again.
        """
        # TODO(rcadene, aliberts): implement faster transfer
        # https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads
        files = None
        ignore_patterns = None if download_videos else "videos/"
        if self.episodes is not None:
            files = self.get_episodes_file_paths()

        self.pull_from_repo(allow_patterns=files, ignore_patterns=ignore_patterns)

    def get_episodes_file_paths(self) -> list[Path]:
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files

        return fpaths

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        if self.episodes is None:
            path = str(self.root / "data")
            hf_dataset = load_dataset("parquet", data_dir=path, split="train")
        else:
            files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def create_hf_dataset(self) -> datasets.Dataset:
        features = get_hf_features_from_features(self.features)
        ft_dict = {col: [] for col in features}
        hf_dataset = datasets.Dataset.from_dict(ft_dict, features=features, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        return len(self.hf_dataset) if self.hf_dataset is not None else self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_features(self) -> datasets.Features:
        """Features of the hf_dataset."""
        if self.hf_dataset is not None:
            return self.hf_dataset.features
        else:
            return get_hf_features_from_features(self.features)

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset.select(query_indices[key])["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        return {
            key: torch.stack(self.hf_dataset.select(q_idx)[key])
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys
        }

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)

        return item

    def _add_padding_keys(self, item: dict, padding: dict[str, list[bool]]) -> dict:
        for key, val in padding.items():
            item[key] = torch.BoolTensor(val)
        return item

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]

        return item

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    def create_episode_buffer(self, episode_index: int | None = None) -> dict:
        current_ep_idx = self.meta.total_episodes if episode_index is None else episode_index
        ep_buffer = {}
        # size and task are special cases that are not in self.features
        ep_buffer["size"] = 0
        ep_buffer["task"] = []
        for key in self.features:
            ep_buffer[key] = current_ep_idx if key == "episode_index" else []
        return ep_buffer

    def _get_image_file_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        fpath = DEFAULT_IMAGE_PATH.format(
            image_key=image_key, episode_index=episode_index, frame_index=frame_index
        )
        return self.root / fpath

    def _save_image(self, image: torch.Tensor | np.ndarray | PIL.Image.Image, fpath: Path) -> None:
        if self.image_writer is None:
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            write_image(image, fpath)
        else:
            self.image_writer.save_image(image=image, fpath=fpath)

    def add_frame(self, frame: dict) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        # Add frame features to episode_buffer
        for key in frame:
            if key == "task":
                # Note: we associate the task in natural language to its task index during `save_episode`
                self.episode_buffer["task"].append(frame["task"])
                continue

            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            if self.features[key]["dtype"] in ["image", "video"]:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"], image_key=key, frame_index=frame_index
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_image(frame[key], img_path)
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

    def save_episode(self, episode_data: dict | None = None) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = compute_episode_stats(episode_buffer, self.features)

        if len(self.meta.video_keys) > 0:
            video_paths = self.encode_episode_videos(episode_index)
            for key in self.meta.video_keys:
                episode_buffer[key] = video_paths[key]

        # `meta.save_episode` be executed after encoding the videos
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        # delete images
        img_dir = self.root / "images"
        if img_dir.is_dir():
            shutil.rmtree(self.root / "images")

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()

    def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
        episode_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(episode_dict, features=self.hf_features, split="train")
        ep_dataset = embed_images(ep_dataset)
        self.hf_dataset = concatenate_datasets([self.hf_dataset, ep_dataset])
        self.hf_dataset.set_transform(hf_transform_to_torch)
        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)

    def clear_episode_buffer(self) -> None:
        episode_index = self.episode_buffer["episode_index"]
        if self.image_writer is not None:
            for cam_key in self.meta.camera_keys:
                img_dir = self._get_image_file_path(
                    episode_index=episode_index, image_key=cam_key, frame_index=0
                ).parent
                if img_dir.is_dir():
                    shutil.rmtree(img_dir)

        # Reset the buffer
        self.episode_buffer = self.create_episode_buffer()

    def start_image_writer(self, num_processes: int = 0, num_threads: int = 4) -> None:
        if isinstance(self.image_writer, AsyncImageWriter):
            logging.warning(
                "You are starting a new AsyncImageWriter that is replacing an already existing one in the dataset."
            )

        self.image_writer = AsyncImageWriter(
            num_processes=num_processes,
            num_threads=num_threads,
        )

    def stop_image_writer(self) -> None:
        """
        Whenever wrapping this dataset inside a parallelized DataLoader, this needs to be called first to
        remove the image_writer in order for the LeRobotDataset object to be pickleable and parallelized.
        """
        if self.image_writer is not None:
            self.image_writer.stop()
            self.image_writer = None

    def _wait_image_writer(self) -> None:
        """Wait for asynchronous image writer to finish."""
        if self.image_writer is not None:
            self.image_writer.wait_until_done()

    def encode_videos(self) -> None:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        for ep_idx in range(self.meta.total_episodes):
            self.encode_episode_videos(ep_idx)

    def encode_episode_videos(self, episode_index: int) -> dict:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        video_paths = {}
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            video_paths[key] = str(video_path)
            if video_path.is_file():
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue
            img_dir = self._get_image_file_path(
                episode_index=episode_index, image_key=key, frame_index=0
            ).parent
            encode_video_frames(img_dir, video_path, self.fps, overwrite=True)

        return video_paths

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        root: str | Path | None = None,
        robot: Robot | None = None,
        robot_type: str | None = None,
        features: dict | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from scratch in order to record data."""
        obj = cls.__new__(cls)
        obj.meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot=robot,
            robot_type=robot_type,
            features=features,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        return obj


class MultiLeRobotDataset(torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`s.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.root = Path(root) if root else HF_LEROBOT_HOME
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 0.0001)
        # Construct the underlying datasets passing everything but `transform` and `delta_timestamps` which
        # are handled by this class.
        self._datasets = [
            LeRobotDataset(
                repo_id,
                root=self.root / repo_id,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=self.tolerances_s[repo_id],
                download_videos=download_videos,
                video_backend=video_backend,
            )
            for repo_id in repo_ids
        ]

        # Disable any data keys that are not common across all of the datasets. Note: we may relax this
        # restriction in future iterations of this class. For now, this is necessary at least for being able
        # to use PyTorch's default DataLoader collate function.
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            logging.warning(
                f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                "other datasets."
            )
            self.disabled_features.update(extra_keys)

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        # TODO(rcadene, aliberts): We should not perform this aggregation for datasets
        # with multiple robots of different ranges. Instead we should have one normalization
        # per robot.
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets])

    @property
    def repo_id_to_index(self):
        """Return a mapping from dataset repo_id to a dataset index automatically created by this class.

        This index is incorporated as a data key in the dictionary returned by `__getitem__`.
        """
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def repo_index_to_id(self):
        """Return the inverse mapping if repo_id_to_index."""
        return {v: k for k, v in self.repo_id_to_index}

    @property
    def fps(self) -> int:
        """Frames per second used during data collection.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info["fps"]

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.

        Returns False if it only loads images from png files.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            features.update({k: v for k, v in dataset.hf_features.items() if k not in self.disabled_features})
        return features

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image, VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        """Keys to access video frames that requires to be decoded into images.

        Note: It is empty if the dataset contains images only,
        or equal to `self.cameras` if the dataset contains videos only,
        or can even be a subset of `self.cameras` in a case of a mixed image/video dataset.
        """
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        # Determine which dataset to get an item from based on the index.
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: '{self.repo_ids}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )
