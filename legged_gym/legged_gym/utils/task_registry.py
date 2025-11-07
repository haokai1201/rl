# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """根据注册的名称或提供的配置文件创建环境。

        参数:
            name (string): 已注册环境的名称
            args (Args, 可选): Isaac Gym命令行参数。如果为None，则会调用get_args()。默认为None
            env_cfg (Dict, 可选): 用于覆盖已注册配置的环境配置文件。默认为None

        异常:
            ValueError: 如果没有与'name'对应的已注册环境则抛出错误

        返回:
            isaacgym.VecTaskPython: 创建的环境实例
            Dict: 对应的配置文件
        """
        # 如果没有传入args参数，则获取命令行参数
        if args is None:
            args = get_args()
        # 检查是否存在具有该名称的已注册环境
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"名称为: {name} 的任务未被注册")
        # 如果未提供配置，则加载默认配置
        if env_cfg is None:
            # 加载配置文件
            env_cfg, _ = self.get_cfgs(name)
        # 从args覆盖配置(如果已指定)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        # 解析模拟参数(首先转换为字典)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        # 创建并返回环境实例
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """ 创建训练算法，可以基于已注册的名称或提供的配置文件。

        参数:
            env (isaacgym.VecTaskPython): 要训练的环境（TODO：从算法内部移除）
            name (string, 可选): 已注册环境的名称。如果为None，则使用配置文件。默认为None。
            args (Args, 可选): Isaac Gym命令行参数。如果为None，则调用get_args()。默认为None。
            train_cfg (Dict, 可选): 训练配置文件。如果为None，则使用'name'获取配置文件。默认为None。
            log_root (str, 可选): Tensorboard的日志目录。设置为'None'可避免记录日志（例如在测试时）。
                                  日志将保存在<log_root>/<date_time>_<run_name>中。默认为"default"=<path_to_LEGGED_GYM>/logs/<experiment_name>。

        异常:
            ValueError: 如果'name'和'train_cfg'都未提供则抛出错误
            Warning: 如果同时提供了'name'和'train_cfg'，则忽略'name'

        返回:
            PPO: 创建的算法
            Dict: 对应的配置文件
        """
        # 如果没有传入args参数，则获取命令行参数
        if args is None:
            args = get_args()
        
        # 如果传入了配置文件则使用它们，否则从名称加载
        if train_cfg is None:
            if name is None:
                raise ValueError("必须提供'name'或'train_cfg'中的一个")
            # 加载配置文件
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"已提供'train_cfg' -> 忽略'name={name}'")
        
        # 从args覆盖配置（如果已指定）
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        # 根据log_root参数设置日志目录
        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        
        # 将配置转换为字典并创建runner实例
        train_cfg_dict = class_to_dict(train_cfg)
        runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
        
        # 在创建新log_dir之前保存恢复路径
        resume = train_cfg.runner.resume
        if resume:
            # 加载之前训练的模型
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"从以下位置加载模型: {resume_path}")
            runner.load(resume_path)
        
        return runner, train_cfg

# make global task registry
task_registry = TaskRegistry()