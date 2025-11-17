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
版本管理模块 (Version Management Module)

功能说明 (Functionality):
    此模块启用 `lerobot.__version__` 属性,允许用户查询已安装的 LeRobot 库版本。
    This module enables the `lerobot.__version__` attribute, allowing users to query
    the installed version of the LeRobot library.

变量说明 (Variable Description):
    __version__ (str): LeRobot 库的版本字符串,例如 "2.1.0"
                      如果包未安装(开发模式),则返回 "unknown"
                      The version string of the LeRobot library, e.g., "2.1.0"
                      Returns "unknown" if package is not installed (development mode)
"""

from importlib.metadata import PackageNotFoundError, version

try:
    # 尝试从已安装的包元数据中获取版本信息
    # Try to get version information from installed package metadata
    __version__ = version("lerobot")
except PackageNotFoundError:
    # 如果包未安装(例如在开发模式下),设置版本为 "unknown"
    # If package is not installed (e.g., in development mode), set version to "unknown"
    __version__ = "unknown"
