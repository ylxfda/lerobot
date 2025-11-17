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
训练脚本 (Training Script)

功能说明 (Functionality):
    LeRobot 的主训练脚本,用于训练模仿学习策略。
    Main training script for LeRobot, used to train imitation learning policies.

核心功能 (Core Features):
    1. 数据加载 (Data Loading): 从数据集加载训练数据
    2. 模型训练 (Model Training): 训练策略网络
    3. 梯度裁剪 (Gradient Clipping): 防止梯度爆炸
    4. 混合精度训练 (Mixed Precision): AMP 加速训练
    5. 学习率调度 (LR Scheduling): 动态调整学习率
    6. 检查点保存 (Checkpointing): 定期保存模型状态
    7. 在线评估 (Online Evaluation): 在仿真环境中评估策略
    8. 日志记录 (Logging): 记录训练指标到 WandB

训练流程 (Training Pipeline):
    1. 加载配置和数据集 / Load configuration and dataset
    2. 创建策略模型 / Create policy model
    3. 设置优化器和调度器 / Setup optimizer and scheduler
    4. 训练循环 / Training loop:
       a. 从数据集采样批次 / Sample batch from dataset
       b. 前向传播计算损失 / Forward pass to compute loss
       c. 反向传播更新参数 / Backward pass to update parameters
       d. 记录训练指标 / Log training metrics
       e. 定期评估和保存 / Periodic evaluation and checkpoint saving
    5. 保存最终模型 / Save final model

使用示例 (Usage Example):
    ```bash
    # 基本训练 / Basic training
    python lerobot/scripts/train.py \\
        dataset.repo_id=lerobot/pusht \\
        policy=diffusion \\
        output_dir=outputs/pusht_diffusion

    # 恢复训练 / Resume training
    python lerobot/scripts/train.py \\
        config_path=outputs/pusht_diffusion/checkpoints/last/train_config.json \\
        resume=true

    # 使用配置文件 / Use configuration file
    python lerobot/scripts/train.py --config-path=./configs/default.yaml
    ```

命令行参数 (Command-line Arguments):
    通过 draccus 自动从 TrainPipelineConfig 生成
    支持通过点号访问嵌套配置,如 policy.chunk_size=16
    Automatically generated from TrainPipelineConfig via draccus
    Supports dot notation for nested configs, e.g., policy.chunk_size=16
"""

import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    执行单次策略更新步骤 (Perform Single Policy Update Step)

    功能说明 (Functionality):
        执行完整的训练迭代:前向传播、反向传播、梯度裁剪、参数更新。
        支持混合精度训练和梯度缩放。

        Performs complete training iteration: forward pass, backward pass,
        gradient clipping, parameter update. Supports mixed precision and gradient scaling.

    参数说明 (Parameters):
        train_metrics (MetricsTracker):
            训练指标跟踪器,用于记录损失和其他指标
            Training metrics tracker for logging loss and other metrics

        policy (PreTrainedPolicy):
            要训练的策略模型 / Policy model to train
            形状 (Shape): 模型参数 / Model parameters

        batch (Any):
            训练批次数据,通常包含观测和动作
            Training batch data, typically contains observations and actions
            结构 (Structure): {"observation.state": Tensor, "action": Tensor, ...}

        optimizer (Optimizer):
            PyTorch 优化器(如 Adam, AdamW) / PyTorch optimizer (e.g., Adam, AdamW)

        grad_clip_norm (float):
            梯度裁剪的最大范数,防止梯度爆炸
            Maximum norm for gradient clipping, prevents gradient explosion
            典型值 (Typical value): 1.0, 10.0

        grad_scaler (GradScaler):
            梯度缩放器,用于混合精度训练
            Gradient scaler for mixed precision training

        lr_scheduler (Optional):
            学习率调度器,每步更新学习率 / LR scheduler, updates LR each step

        use_amp (bool):
            是否使用自动混合精度训练 / Whether to use automatic mixed precision
            默认值 (Default): False

        lock (Optional):
            多进程训练时的锁机制 / Lock for multi-process training

    返回值 (Returns):
        tuple[MetricsTracker, dict]:
            - 更新后的指标跟踪器 / Updated metrics tracker
            - 输出字典,包含额外信息 / Output dict with additional info

    训练步骤 (Training Steps):
        1. 前向传播:计算损失 / Forward pass: compute loss
        2. 反向传播:计算梯度 / Backward pass: compute gradients
        3. 梯度裁剪:限制梯度大小 / Gradient clipping: limit gradient magnitude
        4. 参数更新:应用梯度 / Parameter update: apply gradients
        5. 学习率更新:调整学习率 / LR update: adjust learning rate
    """
    start_time = time.perf_counter()
    # 获取模型设备(CPU/CUDA) / Get model device (CPU/CUDA)
    device = get_device_from_parameters(policy)

    # 设置为训练模式(启用 dropout 等) / Set to training mode (enable dropout, etc.)
    policy.train()

    # 前向传播:使用混合精度(如果启用) / Forward pass: use mixed precision (if enabled)
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        # 计算损失和输出字典 / Compute loss and output dictionary
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # 缩放损失并反向传播 / Scale loss and backpropagate
    grad_scaler.scale(loss).backward()

    # 在梯度裁剪之前取消缩放优化器参数的梯度
    # Unscale gradients of optimizer's assigned params **prior to gradient clipping**
    grad_scaler.unscale_(optimizer)

    # 梯度裁剪:限制梯度范数以防止梯度爆炸
    # Gradient clipping: limit gradient norm to prevent gradient explosion
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,  # 如果梯度包含 inf/NaN,不抛出错误 / Don't error if grad contains inf/NaN
    )

    # 优化器步骤:梯度已经取消缩放,scaler.step 不会再次取消缩放
    # 但如果梯度包含 inf/NaN,仍会跳过 optimizer.step()
    # Optimizer step: gradients already unscaled, scaler.step won't unscale again
    # But still skips optimizer.step() if gradients contain infs or NaNs
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)

    # 更新下一次迭代的缩放因子 / Update scale for next iteration
    grad_scaler.update()

    # 清零梯度,准备下一次迭代 / Zero gradients for next iteration
    optimizer.zero_grad()

    # 每个批次更新学习率(而不是每个 epoch)
    # Step through learning rate scheduler at every batch (instead of epoch)
    if lr_scheduler is not None:
        lr_scheduler.step()

    # 如果策略有 update 方法,调用它
    # 例如 TDMPC 中的指数移动平均(EMA)更新
    # If policy has update method, call it
    # e.g., Exponential Moving Average (EMA) update in TDMPC
    if has_method(policy, "update"):
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if eval_env:
        eval_env.close()
    logging.info("End of training")


if __name__ == "__main__":
    init_logging()
    train()
