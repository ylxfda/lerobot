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
训练流程配置模块 (Training Pipeline Configuration Module)

功能说明 (Functionality):
    定义完整训练流程的配置类 TrainPipelineConfig。
    这是训练脚本的核心配置,包含:
    - 数据集配置 (Dataset configuration)
    - 环境配置 (Environment configuration)
    - 策略配置 (Policy configuration)
    - 优化器和调度器 (Optimizer and scheduler)
    - 训练超参数 (Training hyperparameters)
    - 评估设置 (Evaluation settings)
    - 日志和检查点 (Logging and checkpointing)

    Defines the complete training pipeline configuration class TrainPipelineConfig.
    This is the core configuration for training scripts, including:
    - Dataset configuration
    - Environment configuration
    - Policy configuration
    - Optimizer and scheduler
    - Training hyperparameters
    - Evaluation settings
    - Logging and checkpointing

使用方式 (Usage):
    1. 通过 YAML 文件配置 / Configure via YAML file
    2. 通过命令行参数覆盖 / Override via command-line arguments
    3. 从检查点恢复训练 / Resume training from checkpoint

配置文件名称 (Configuration File Name):
    train_config.json - 保存在输出目录中的完整训练配置
                       Complete training configuration saved in output directory
"""

import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot.common import envs
from lerobot.common.optim import OptimizerConfig
from lerobot.common.optim.schedulers import LRSchedulerConfig
from lerobot.common.utils.hub import HubMixin
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig

# 训练配置文件名 / Training configuration file name
TRAIN_CONFIG_NAME = "train_config.json"


@dataclass
class TrainPipelineConfig(HubMixin):
    """
    完整训练流程配置类 (Complete Training Pipeline Configuration Class)

    功能说明 (Functionality):
        封装训练脚本所需的所有配置参数。
        支持从 YAML 文件加载、命令行覆盖和从检查点恢复。

        Encapsulates all configuration parameters needed for training scripts.
        Supports loading from YAML files, command-line overrides, and checkpoint resumption.

    配置字段 (Configuration Fields):
        dataset (DatasetConfig): 数据集配置 / Dataset configuration
        env (EnvConfig | None): 环境配置(用于在线评估) / Environment config (for online eval)
        policy (PreTrainedConfig | None): 策略模型配置 / Policy model configuration
        output_dir (Path | None): 输出目录路径 / Output directory path
        job_name (str | None): 任务名称 / Job name
        resume (bool): 是否从检查点恢复 / Whether to resume from checkpoint
        seed (int): 随机种子 / Random seed
        num_workers (int): DataLoader工作进程数 / Number of DataLoader workers
        batch_size (int): 批次大小 / Batch size
        steps (int): 总训练步数 / Total training steps
        eval_freq (int): 评估频率 / Evaluation frequency
        log_freq (int): 日志记录频率 / Logging frequency
        save_checkpoint (bool): 是否保存检查点 / Whether to save checkpoints
        save_freq (int): 检查点保存频率 / Checkpoint saving frequency
        optimizer (OptimizerConfig): 优化器配置 / Optimizer configuration
        scheduler (LRSchedulerConfig): 学习率调度器配置 / LR scheduler configuration
        eval (EvalConfig): 评估配置 / Evaluation configuration
        wandb (WandBConfig): Weights & Biases配置 / W&B configuration
    """

    dataset: DatasetConfig  # 数据集配置 / Dataset configuration

    env: envs.EnvConfig | None = None  # 环境配置(可选) / Environment configuration (optional)

    policy: PreTrainedConfig | None = None  # 策略配置 / Policy configuration

    # 输出目录:保存所有运行输出的位置
    # 如果使用相同的 dir 运行另一个训练会话,其内容将被覆盖,除非设置 resume=true
    # Output directory: where to save all run outputs
    # If running another session with same dir, contents will be overwritten unless resume=true
    output_dir: Path | None = None

    job_name: str | None = None  # 任务名称 / Job name

    # 恢复选项:设为 true 以恢复之前的运行
    # 需要确保 output_dir 是包含至少一个检查点的现有运行目录
    # 恢复时,默认使用检查点中的配置,而不是命令行提供的配置
    # Resume option: set to true to resume a previous run
    # Need to ensure output_dir is existing run directory with at least one checkpoint
    # When resuming, uses configuration from checkpoint by default
    resume: bool = False

    # 随机种子:用于训练(如模型初始化、数据集洗牌)和评估环境
    # Random seed: used for training (model init, dataset shuffling) and eval envs
    seed: int | None = 1000

    # DataLoader 工作进程数 / Number of DataLoader workers
    num_workers: int = 4

    batch_size: int = 8          # 批次大小 / Batch size
    steps: int = 100_000         # 总训练步数 / Total training steps
    eval_freq: int = 20_000      # 评估频率(步) / Evaluation frequency (steps)
    log_freq: int = 200          # 日志记录频率(步) / Logging frequency (steps)
    save_checkpoint: bool = True # 是否保存检查点 / Whether to save checkpoints

    # 检查点保存频率:每 save_freq 训练迭代保存一次,以及最后一步后保存
    # Checkpoint save frequency: saves every save_freq iterations and after last step
    save_freq: int = 20_000

    use_policy_training_preset: bool = True  # 使用策略训练预设 / Use policy training preset

    optimizer: OptimizerConfig | None = None      # 优化器配置 / Optimizer configuration
    scheduler: LRSchedulerConfig | None = None    # 学习率调度器配置 / LR scheduler configuration
    eval: EvalConfig = field(default_factory=EvalConfig)      # 评估配置 / Evaluation configuration
    wandb: WandBConfig = field(default_factory=WandBConfig)   # W&B配置 / W&B configuration

    def __post_init__(self):
        self.checkpoint_path = None

    def validate(self):
        # HACK: We parse again the cli args here to get the pretrained paths if there was some.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            # Only load the policy config
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        elif self.resume:
            # The entire train config is already loaded, we just need to get the checkpoint dir
            config_path = parser.parse_arg("config_path")
            if not config_path:
                raise ValueError(
                    f"A config_path is expected when resuming a run. Please specify path to {TRAIN_CONFIG_NAME}"
                )
            if not Path(config_path).resolve().exists():
                raise NotADirectoryError(
                    f"{config_path=} is expected to be a local path. "
                    "Resuming from the hub is not supported for now."
                )
            policy_path = Path(config_path).parent
            self.policy.pretrained_path = policy_path
            self.checkpoint_path = policy_path.parent

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type}"
            else:
                self.job_name = f"{self.env.type}_{self.policy.type}"

        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume is {self.resume}. "
                f"Please change your output directory so that {self.output_dir} is not overwritten."
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        if isinstance(self.dataset.repo_id, list):
            raise NotImplementedError("LeRobotMultiDataset is not currently implemented.")

        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when the policy presets are not used.")
        elif self.use_policy_training_preset and not self.resume:
            self.optimizer = self.policy.get_optimizer_preset()
            self.scheduler = self.policy.get_scheduler_preset()

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

    def to_dict(self) -> dict:
        return draccus.encode(self)

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / TRAIN_CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: Type["TrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> "TrainPipelineConfig":
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
            else:
                print(f"{TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        elif Path(model_id).is_file():
            config_file = model_id
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=TRAIN_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{TRAIN_CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        cli_args = kwargs.pop("cli_args", [])
        cfg = draccus.parse(cls, config_file, args=cli_args)

        return cfg
