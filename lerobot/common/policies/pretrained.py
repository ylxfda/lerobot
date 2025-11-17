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
预训练策略基类模块 (Pretrained Policy Base Class Module)

功能说明 (Functionality):
    定义所有策略模型的基类 PreTrainedPolicy。
    提供统一的接口用于:
    - 策略模型的保存和加载 (Save and load policy models)
    - HuggingFace Hub 集成 (HuggingFace Hub integration)
    - 配置管理 (Configuration management)
    - Safetensors 格式序列化 (Safetensors format serialization)

    所有具体策略(ACT, Diffusion, TDMPC等)都继承自这个基类。
    All concrete policies (ACT, Diffusion, TDMPC, etc.) inherit from this base class.

主要类 (Main Classes):
    PreTrainedPolicy: 策略模型的抽象基类 / Abstract base class for policy models

使用模式 (Usage Pattern):
    1. 子类必须定义 config_class 和 name 类属性
    2. 实现 forward() 方法定义前向传播
    3. 使用 from_pretrained() 加载预训练模型
    4. 使用 save_pretrained() 保存模型到 Hub

    1. Subclasses must define config_class and name class attributes
    2. Implement forward() method to define forward pass
    3. Use from_pretrained() to load pretrained models
    4. Use save_pretrained() to save models to Hub
"""

import abc
import logging
import os
from pathlib import Path
from typing import Type, TypeVar

import packaging
import safetensors
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_model as load_model_as_safetensor
from safetensors.torch import save_model as save_model_as_safetensor
from torch import Tensor, nn

from lerobot.common.utils.hub import HubMixin
from lerobot.configs.policies import PreTrainedConfig

# 类型变量,用于类型提示 / Type variable for type hints
T = TypeVar("T", bound="PreTrainedPolicy")

# 默认的策略模型卡片模板 / Default policy model card template
# 用于在推送模型到 Hub 时生成 README.md
# Used to generate README.md when pushing model to Hub
DEFAULT_POLICY_CARD = """
---
# 模型卡片元数据参考规范 / For reference on model card metadata spec
# https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# 文档/指南 / Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

此策略使用 [LeRobot](https://github.com/huggingface/lerobot) 推送到 Hub:
This policy has been pushed to the Hub using [LeRobot](https://github.com/huggingface/lerobot):
- Docs: {{ docs_url | default("[More Information Needed]", true) }}
"""


class PreTrainedPolicy(nn.Module, HubMixin, abc.ABC):
    """
    策略模型的基类 (Base Class for Policy Models)

    功能说明 (Functionality):
        所有 LeRobot 策略模型的抽象基类。继承自:
        - nn.Module: PyTorch 模型基类
        - HubMixin: 提供 HuggingFace Hub 集成功能
        - abc.ABC: 抽象基类,强制子类实现特定方法

        Abstract base class for all LeRobot policy models. Inherits from:
        - nn.Module: PyTorch model base class
        - HubMixin: Provides HuggingFace Hub integration
        - abc.ABC: Abstract base class, enforces subclass implementation

    必需的类属性 (Required Class Attributes):
        config_class (Type[PreTrainedConfig]): 对应的配置类
                                              例如: ACTConfig, DiffusionConfig
                                              Corresponding configuration class
                                              e.g., ACTConfig, DiffusionConfig

        name (str): 策略名称,用于识别和加载
                   例如: "act", "diffusion", "tdmpc"
                   Policy name for identification and loading
                   e.g., "act", "diffusion", "tdmpc"

    主要方法 (Main Methods):
        __init__(config): 初始化策略 / Initialize policy
        forward(batch): 前向传播(抽象方法,子类必须实现) / Forward pass (abstract, must implement)
        from_pretrained(path): 从预训练权重加载 / Load from pretrained weights
        save_pretrained(path): 保存模型到目录/Hub / Save model to directory/Hub

    使用示例 (Usage Example):
        ```python
        # 定义新策略 / Define new policy
        class MyPolicy(PreTrainedPolicy):
            config_class = MyPolicyConfig
            name = "my_policy"

            def forward(self, batch):
                # 实现前向传播 / Implement forward pass
                return output

        # 从预训练模型加载 / Load from pretrained
        policy = MyPolicy.from_pretrained("username/my_policy")

        # 保存模型 / Save model
        policy.save_pretrained("./my_policy")
        ```
    """

    # 类属性:子类必须重写 / Class attributes: must be overridden by subclasses
    config_class: None  # 配置类类型 / Configuration class type
    name: None          # 策略名称 / Policy name

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PreTrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "config_class", None):
            raise TypeError(f"Class {cls.__name__} must define 'config_class'")
        if not getattr(cls, "name", None):
            raise TypeError(f"Class {cls.__name__} must define 'name'")

    def _save_pretrained(self, save_directory: Path) -> None:
        self.config._save_pretrained(save_directory)
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))

    @classmethod
    def from_pretrained(
        cls: Type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        """
        The policy is set in evaluation mode by default using `policy.eval()` (dropout modules are
        deactivated). To train it, you should first set it back in training mode with `policy.train()`.
        """
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e

        policy.to(config.device)
        policy.eval()
        return policy

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        if packaging.version.parse(safetensors.__version__) < packaging.version.parse("0.4.3"):
            load_model_as_safetensor(model, model_file, strict=strict)
            if map_location != "cpu":
                logging.warning(
                    "Loading model weights on other devices than 'cpu' is not supported natively in your version of safetensors."
                    " This means that the model is loaded on 'cpu' first and then copied to the device."
                    " This leads to a slower loading time."
                    " Please update safetensors to version 0.4.3 or above for improved performance."
                )
                model.to(map_location)
        else:
            safetensors.torch.load_model(model, model_file, strict=strict, device=map_location)
        return model

    # def generate_model_card(self, *args, **kwargs) -> ModelCard:
    #     card = ModelCard.from_template(
    #         card_data=self._hub_mixin_info.model_card_data,
    #         template_str=self._hub_mixin_info.model_card_template,
    #         repo_url=self._hub_mixin_info.repo_url,
    #         docs_url=self._hub_mixin_info.docs_url,
    #         **kwargs,
    #     )
    #     return card

    @abc.abstractmethod
    def get_optim_params(self) -> dict:
        """
        Returns the policy-specific parameters dict to be passed on to the optimizer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """To be called whenever the environment is reset.

        Does things like clearing caches.
        """
        raise NotImplementedError

    # TODO(aliberts, rcadene): split into 'forward' and 'compute_loss'?
    @abc.abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """_summary_

        Args:
            batch (dict[str, Tensor]): _description_

        Returns:
            tuple[Tensor, dict | None]: The loss and potentially other information. Apart from the loss which
                is a Tensor, all other items should be logging-friendly, native Python types.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Return one action to run in the environment (potentially in batch mode).

        When the model uses a history of observations, or outputs a sequence of actions, this method deals
        with caching.
        """
        raise NotImplementedError
