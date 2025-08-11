# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
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

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
    to_torch, get_axis_params, torch_rand_float, tensor_clamp, quat_apply  
from isaacgymenvs.tasks.base.vec_task import VecTask


class AllegroHandDown(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg

        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["block", "egg", "pen"]

        self.ignore_z = (self.object_type == "pen")

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml"
        }

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get("assetFileNameBlock", self.asset_files_dict["block"])
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg", self.asset_files_dict["egg"])
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get("assetFileNamePen", self.asset_files_dict["pen"])

        # can be "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["full_no_vel", "full", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")

        print("Obs type:", self.obs_type)

        self.num_obs_dict = {
            "full_no_vel": 50,
            "full": 72,
            "full_state": 88
        }

        self.up_axis = 'z'

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 88

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = 16

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # original settings.
        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # Added for the rotation task.
        # ==== 自转轴与固定正交基（每个 env 一次构造，整段 episode 复用） ====
        axis_k = self.cfg["env"].get("axisK", [0,0,1])
        self.k_world = to_torch(axis_k, device=self.device, dtype=torch.float).repeat((self.num_envs,1))
        self.k_world = self.k_world / (self.k_world.norm(dim=-1, keepdim=True) + 1e-9)

        a = to_torch([1,0,0], device=self.device, dtype=torch.float).repeat((self.num_envs,1))
        need_swap = (torch.abs((a * self.k_world).sum(-1)) > 0.9)
        a[need_swap] = to_torch([0,1,0], device=self.device, dtype=torch.float)
        self.v1_basis = torch.nn.functional.normalize(torch.cross(self.k_world, a, dim=-1), dim=-1)
        self.v2_basis = torch.nn.functional.normalize(torch.cross(self.k_world, self.v1_basis, dim=-1), dim=-1)

        # ==== 目标角/阈值 ====
        self.use_theta_goal = bool(self.cfg["env"].get("useThetaGoal", False))
        if self.use_theta_goal:
            self.theta_goal = torch.deg2rad(torch.full((self.num_envs,), float(self.cfg["env"].get("thetaGoalDeg", 180.0)), device=self.device))
            self.delta_theta_tol = torch.deg2rad(torch.full((self.num_envs,), float(self.cfg["env"].get("deltaThetaTolDeg", 10.0)), device=self.device))
        else:
            self.theta_goal = torch.zeros(self.num_envs, device=self.device)
            self.delta_theta_tol = torch.zeros(self.num_envs, device=self.device)

        self.delta_xy_tol  = float(self.cfg["env"].get("deltaXYTol", 0.02))
        self.p_bar         = float(self.cfg["env"].get("pushAwayThresh", 0.07))
        self.z_min         = float(self.cfg["env"].get("zMin", 0.70))
        self.success_hold_steps = int(self.cfg["env"].get("successHoldSteps", 15))
        self.table_force_thr    = float(self.cfg["env"].get("tableForceThr", 5.0))

        # ==== 权重 ====
        W = self.cfg["env"].get("w", {})
        self.w_rot    = float(W.get("rot",    3.0))
        self.w_err    = float(W.get("err",    1.5))
        self.w_succ   = float(W.get("succ",   5.0))
        self.w_dist   = float(W.get("dist",   0.0))
        self.w_push   = float(W.get("push",   1.0))
        self.w_table  = float(W.get("table",  0.0))
        self.w_fall   = float(W.get("fall",   2.0))
        self.w_work   = float(W.get("work",   1e-3))
        self.w_torque = float(W.get("torque", 1e-4))
        self.w_pen    = float(W.get("pen",    0.5))
        self.w_dev    = float(W.get("dev",    1.0))

        # === 新的奖励开关/系数（可在 YAML 里覆盖）===
        R = self.cfg["env"].get("reward", {})
        # 2) 指尖接近度（仅当 <0 时生效，惩罚距离）
        self.ftip_reward_scale    = float(R.get("ftipRewardScale", 0.0))    # 例如 -1.0
        # 3) 能耗（功率）惩罚
        self.energy_scale         = float(R.get("energyScale", 0.0))        # >0 生效
        self.clip_energy_reward   = int(R.get("clipEnergyReward", 0))       # 0/1
        self.energy_upper_bound   = float(R.get("energyUpperBound", 50.0))  # 裁剪上界
        # 4) 桌面接触惩罚
        self.penalize_tb_contact  = int(R.get("penalizeTbContact", 1))      # 0/1
        self.tb_cf_scale          = float(R.get("tbContactScale", 0.5))     # >0 生效
        self.tb_cf_thr            = float(R.get("tbContactThr", 0.2))   

        if self.penalize_tb_contact:
            _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
            self.net_contact_force = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
            
            env0 = self.envs[0]
            table_handle0 = self.gym.find_actor_handle(env0, "table")
            # 列出该 actor 的刚体名称（对 box 资产一般只有 1 个刚体）
            tb_body_names = self.gym.get_actor_rigid_body_names(env0, table_handle0)
            assert len(tb_body_names) >= 1, "table actor 没找到刚体"
            tb_first_body = tb_body_names[0]  # 只有一个时就取第一个

            # 拿“env 域”的刚体索引（可直接切 net_contact_force）
            self.table_body_index = self.gym.find_actor_rigid_body_index(
                env0, table_handle0, tb_first_body, gymapi.DOMAIN_ENV
            )
            print(f"[table] body_name={tb_first_body}, env_rb_index={self.table_body_index}")

            # 注意：下面这个视图每步前要 refresh_net_contact_force_tensor
            self.table_contact_force = self.net_contact_force[:, self.table_body_index, :]  # [N,3]
        
        # ==== 状态缓存 ====
        self.theta_buf       = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)  # 累计角 Θ
        self.obj_xy_ref      = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.last_object_rot = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.succ_hold_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # 单步推进裁剪：c1 = 1.5*dt（再 cap 一下）
        self.c1 = min(1.5 * self.dt, 0.25)

        # ==== 指尖 body 索引（固定为你给的四个） ====
        self.fingertip_names = ["link_3_tip", "link_7_tip", "link_11_tip", "link_15_tip"]
        self._cache_fingertip_indices()  # 函数见下

        # ==== 获取 net contact force tensor（用于 r_table） ====
        contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.net_contact_forces = gymtorch.wrap_tensor(contact_tensor).view(self.num_envs, -1, 3)


        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
        #     sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        #     self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

             dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
             self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # create some wrapper tensors for different slices
        # self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print("Num dofs: ", self.num_dofs)

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        
        pos0 = self.shadow_hand_default_dof_pos  # 这是在 _create_envs 里算好的 q_init
        # print("pos0(default):", pos0.tolist())
        self.prev_targets[:, :self.num_shadow_hand_dofs] = pos0
        self.cur_targets[:,  :self.num_shadow_hand_dofs] = pos0
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        self.global_indices = torch.arange(self.num_envs * 4, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                           * torch.rand(self.num_envs, device=self.device) + torch.log(self.force_prob_range[1]))

        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        self.fingertip_names = ["link_3_tip", "link_7_tip", "link_11_tip", "link_15_tip"]
        self.fingertip_actor_indices = [21, 16, 11, 6] 
        self._cache_fingertip_indices()
        
        
    

    def _cache_fingertip_indices(self):
        # # 只缓存 env 级索引（用于 rigid_body_states / net_contact_forces）
        # self.tip_actor_ids = []
        # self.tip_env_ids   = []
        # if len(self.envs) == 0:
        #     return
        # env0  = self.envs[0]
        # hand0 = self.allegro_hands[0]
        # env_offset = self._table_rb_count

        # print("\n[Fingertips mapping by name]")
        # for nm in self.fingertip_names:  # ["link_3_tip", "link_7_tip", "link_11_tip", "link_15_tip"]
        #     h_tip = self.gym.find_actor_rigid_body_handle(env0, hand0, nm)
        #     assert h_tip >= 0, f"Tip body not found by name: {nm}"
        #     print(f"  {nm:<14} -> actor[{h_tip:02d}] -> env[{env_offset + h_tip:02d}]")
        #     self.tip_actor_ids.append(h_tip)
        #     self.tip_env_ids.append(env_offset + h_tip)

        # self.tip_body_ids_tensor = to_torch(self.tip_env_ids, dtype=torch.long, device=self.device)

        self.tip_actor_ids = list(self.fingertip_actor_indices)

        # env 级（用于 self.rigid_body_states / net_contact_forces）
        # 当前创建顺序是：table -> hand -> object -> goal
        # 所以 hand 的 env 索引 = table 刚体数 + hand 的 actor 索引
        if not hasattr(self, "_table_rb_count"):
            # 确保在 _create_envs() 里先设置了 self._table_rb_count
            raise RuntimeError("self._table_rb_count 未设置；请确认在 _create_envs() 里已赋值")

        tip_env_ids = [self._table_rb_count + i for i in self.tip_actor_ids]
        self.tip_env_ids_tensor = to_torch(tip_env_ids, dtype=torch.long, device=self.device)

       
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # ===== 默认资产路径（可被 cfg 覆盖）=====
        # 手
        hand_root = "/home/chen/IsaacGymEnvs/assets/urdf/allegro_hand_v4/allegro_hand_description"
        hand_file = "allegro_hand_right.urdf"
        # 桌子
        table_root = "/home/chen/IsaacGymEnvs/assets/urdf"
        table_file = "table_small.urdf"
        # 物体（block/egg/pen 从 repo assets 根下取）
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        object_asset_file = self.asset_files_dict[self.object_type]

                # initial joint 
        self.hand_qpos_init_override = {
            "joint_0": -0.0771,
            "joint_1": 0.6144, 
            "joint_2": 0.6420, 
            "joint_3": 0.6009,
            "joint_4": 0.0,
            "joint_5": 0.6885,
            "joint_6": 0.6990,
            "joint_7": 0.5725,
            "joint_8": 0.0,
            "joint_9": 0.8274, 
            "joint_10": 0.6709, 
            "joint_11": 0.3643,
            "joint_12": 1.2536,
            "joint_13": 0.5940,
            "joint_14": -0.1373,
            "joint_15": 0.5759,
        }

        # 允许 YAML 覆盖
        if "asset" in self.cfg["env"]:
            hand_root  = self.cfg["env"]["asset"].get("handRoot", hand_root)
            hand_file  = self.cfg["env"]["asset"].get("handFile", hand_file)
            # table_root = self.cfg["env"]["asset"].get("tableRoot", table_root)
            table_file = self.cfg["env"]["asset"].get("tableFile", table_file)
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            object_asset_file = self.cfg["env"]["asset"].get("objectFile", object_asset_file)

        # ===== 资产选项（GPU PhysX 更稳的配置）=====
        hand_opts = gymapi.AssetOptions()
        hand_opts.flip_visual_attachments = False
        hand_opts.fix_base_link = True
        hand_opts.collapse_fixed_joints = False # True
        hand_opts.disable_gravity = False
        hand_opts.thickness = 0.001
        hand_opts.angular_damping = 0.01
        hand_opts.default_dof_drive_mode = gymapi.DOF_MODE_POS
        if self.physics_engine == gymapi.SIM_PHYSX:
            hand_opts.use_physx_armature = True
        hand_opts.vhacd_enabled = True
        hand_opts.vhacd_params = gymapi.VhacdParams()

        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_opts.use_mesh_materials = True
        table_opts.vhacd_enabled = True
        # ===== 新：直接用盒子资产来当桌子 =====
        # 你的尺寸：0.12 x 0.12 x 0.5（单位：米）
        table_sx, table_sy, table_sz = 0.12, 0.12, 0.5
        table_asset = self.gym.create_box(self.sim, table_sx, table_sy, table_sz, table_opts)

        obj_opts = gymapi.AssetOptions()
        obj_opts.disable_gravity = False
        obj_opts.vhacd_enabled = True
        # goal 用同一份选项
        goal_opts = gymapi.AssetOptions()
        goal_opts.disable_gravity = True
        goal_opts.vhacd_enabled = True

        # ===== 加载资产（注意：先设选项后加载）=====
        allegro_hand_asset = self.gym.load_asset(self.sim, hand_root,  hand_file,  hand_opts)
        # table_asset        = self.gym.load_asset(self.sim, table_root, table_file, table_opts)
        object_asset       = self.gym.load_asset(self.sim, asset_root, object_asset_file, obj_opts)
        goal_asset         = self.gym.load_asset(self.sim, asset_root, object_asset_file, goal_opts)

        # ===== Hand DOF/刚体/形状数等 =====
        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(allegro_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(allegro_hand_asset)
        self.num_shadow_hand_dofs   = self.gym.get_asset_dof_count(allegro_hand_asset)
        print("Num dofs: ", self.num_shadow_hand_dofs)
        self.num_shadow_hand_actuators = self.num_shadow_hand_dofs
        self.actuated_dof_indices = [i for i in range(self.num_shadow_hand_dofs)]

        # DOF 属性（保持你之前的）
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(allegro_hand_asset)
        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos  = []
        self.shadow_hand_dof_default_vel  = []
        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)
            shadow_hand_dof_props['effort'][i]    = 0.5
            shadow_hand_dof_props['stiffness'][i] = 3.0
            shadow_hand_dof_props['damping'][i]   = 0.1
            shadow_hand_dof_props['friction'][i]  = 0.01
            shadow_hand_dof_props['armature'][i]  = 0.001

        self.actuated_dof_indices      = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos  = to_torch(self.shadow_hand_dof_default_pos,  device=self.device)
        self.shadow_hand_dof_default_vel  = to_torch(self.shadow_hand_dof_default_vel,  device=self.device)

        # set the init joint settings 

        # ===== 位姿：桌子 / 物体 / 手（palm-down） / 目标 =====
        # table_h = 0.72  # 按你的 table_small.urdf 实际高度调整；若URDF原点在桌面面心，则把下面的 /2 改为 = table_h
        # self.table_height = table_h
        # table_pose = gymapi.Transform()
        # table_pose.p = gymapi.Vec3(0.0, 0.0, table_h / 2.0)

        # 桌子原点在几何中心 => 表面高度 = 中心z + 高度的一半
        table_center_z = table_sz * 0.5  # 0.25
        table_top_z    = table_center_z + table_sz * 0.0  # 表面就是 center 上方一半，但我们把中心就放在 0.25 方便推导
        # 为简单：把“中心”放在 z=0.25，等价于“表面”在 z=0.5
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, table_center_z)  # 0.25

        self.table_height = table_sz  #* 0.5  # = 0.5（桌面 z）

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.0, 0.0, self.table_height + 0.03)
        object_start_pose.r = gymapi.Quat(0, 0, 0, 1)

        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(-0.08, 0.0, self.table_height + 0.14)
        shadow_hand_start_pose.r = gymapi.Quat.from_euler_zyx(0.0, np.pi/2, 0.0)  # 手心朝 -Z

        self.goal_displacement = gymapi.Vec3(-0.2, -0.1, 0.12)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement
        goal_start_pose.r = object_start_pose.r

        # ===== aggregate 尺寸（手 + 物体 + 目标 + 桌子）=====
        hand_rb  = self.gym.get_asset_rigid_body_count(allegro_hand_asset)
        hand_sh  = self.gym.get_asset_rigid_shape_count(allegro_hand_asset)
        obj_rb   = self.gym.get_asset_rigid_body_count(object_asset)
        obj_sh   = self.gym.get_asset_rigid_shape_count(object_asset)
        table_rb = self.gym.get_asset_rigid_body_count(table_asset)
        table_sh = self.gym.get_asset_rigid_shape_count(table_asset)

        # 保存 table 刚体数，用作 hand 在 env 的索引偏移
        self._table_rb_count = table_rb
        self._printed_maps = False       # 只打印一次，防止刷屏

        max_agg_bodies = hand_rb + obj_rb + obj_rb + table_rb
        max_agg_shapes = hand_sh + obj_sh + obj_sh + table_sh

        # ===== 容器 =====
        self.allegro_hands = []
        self.envs = []
        self.object_init_state = []
        self.hand_start_states = []
        self.hand_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.table_indices = []

        # 在我们定义的创建顺序（table→hand→object→goal）下，物体刚体在 env 内的起始偏移
        # table 的刚体先放，再 hand，再 object：
        object_rb_offset = table_rb + hand_rb
        self.object_rb_handles = list(range(object_rb_offset, object_rb_offset + obj_rb))

        # ===== 创建各个 env =====
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # 1) 桌子
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0, 0)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            for sp in shape_props:
                sp.friction = 0.5        # 你的 <friction value="1.0"/>
                sp.restitution = 0.5
                sp.rolling_friction = 0.1
                sp.torsion_friction = 0.1
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, shape_props)
            # table_id = self.gym.get_actor_index(env_ptr, table_actor, gymapi.DOMAIN_SIM)
            # self.table_indices.append(table_id)

            # 2) 手
            allegro_hand_actor = self.gym.create_actor(env_ptr, allegro_hand_asset, shadow_hand_start_pose, "hand", i, -1, 0)
            self.gym.set_actor_dof_properties(env_ptr, allegro_hand_actor, shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, allegro_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)
            self.hand_start_states.append([
                shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z, shadow_hand_start_pose.r.w,
                0, 0, 0, 0, 0, 0
            ])
            self.gym.enable_actor_dof_force_sensors(env_ptr, allegro_hand_actor)
            self.hand_override_info = [
                (self.gym.find_actor_dof_handle(env_ptr, allegro_hand_actor, name),
                self.hand_qpos_init_override[name]) for name in self.hand_qpos_init_override
            ]

            # # —— 用名字找到四个 tip（当前 env 的 hand）并上红色（可视+碰撞）——
            # name_handles_this_env = []
            # self.fingertip_names = ["link_3_tip", "link_7_tip", "link_11_tip", "link_15_tip"]
            # for nm in self.fingertip_names:
            #     h_tip = self.gym.find_actor_rigid_body_handle(env_ptr, allegro_hand_actor, nm)
            #     assert h_tip >= 0, f"Tip body not found by name in this env: {nm}"
            #     name_handles_this_env.append(h_tip)
            #     for mesh_type in (gymapi.MESH_VISUAL, gymapi.MESH_COLLISION):
            #         self.gym.set_rigid_body_color(
            #             env_ptr, allegro_hand_actor, h_tip, mesh_type, gymapi.Vec3(0.95, 0.2, 0.2)
            #         )           
            self.fingertip_actor_indices = [21, 16, 11, 6]

            # --- 给 4 个 tip 上红色，方便在 viewer 里确认 ---
            for b in self.fingertip_actor_indices:
                self.gym.set_rigid_body_color(
                    env_ptr, allegro_hand_actor, b, gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.3, 0.95)
                )
            self._need_verify_once = True

            # 3) 物体
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            self.object_init_state.append([
                object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                0, 0, 0, 0, 0, 0
            ])

            # 4) 目标（可视化）
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            if self.object_type != "block":
                self.gym.set_rigid_body_color(env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(env_ptr, goal_handle,   0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.allegro_hands.append(allegro_hand_actor)

            if i == 0 and not self._printed_maps:
                tip_env_ids = [self._table_rb_count + j for j in self.fingertip_actor_indices]
                print(f"[FINGERTIP] actor_ids={self.fingertip_actor_indices} -> env_ids={tip_env_ids}")


        # ===== 结尾：转换为张量/缓存 =====
        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)


        # === 在 _create_envs(...) 的最后，把名字->索引的映射和 q_init 写入仿真 ===
        env0  = self.envs[0]
        hand0 = self.allegro_hands[0]

        idx_val_pairs = []
        for name, val in self.hand_qpos_init_override.items():
            idx = self.gym.find_actor_dof_handle(env0, hand0, name)
            # print(f"[init map] {name} -> {idx}")
            assert idx >= 0, f"DOF name not found: {name} (检查 URDF 名字是否匹配)"
            idx_val_pairs.append((idx, val))

        q_init = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        for idx, val in idx_val_pairs:
            q_init[idx] = val
        q_init = torch.clamp(q_init, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)

        # 作为“默认 reset 姿态”，只存一份在类里，后续 reset 用它
        self.shadow_hand_default_dof_pos = q_init.clone()

        # 用 per-actor API 直接设置每个环境里“手”的初始关节状态（此时还没有 GPU 张量视图）
        init_states = np.zeros(self.num_shadow_hand_dofs, dtype=gymapi.DofState.dtype)
        init_states['pos'] = q_init.detach().cpu().numpy()
        init_states['vel'] = 0.0

        for env, hand in zip(self.envs, self.allegro_hands):
            self.gym.set_actor_dof_states(env, hand, init_states, gymapi.STATE_ALL)
    
    def _print_actor_body_map(self, asset, actor, env_ptr, env_offset, tag):
        n = self.gym.get_asset_rigid_body_count(asset)
        print(f"\n[{tag}] asset_rb_count={n}, env_offset={env_offset}")
        for j in range(n):
            nm = self.gym.get_asset_rigid_body_name(asset, j)
            h  = self.gym.find_actor_rigid_body_handle(env_ptr, actor, nm)  # actor 内索引
            print(f"  asset[{j:02d}] {nm:<16} -> actor[{h:02d}] -> env[{env_offset + h:02d}]")
            
    def compute_reward(self, actions):
        # 刷新接触力张量（每步都要）
        self.gym.refresh_net_contact_force_tensor(self.sim)

        if hasattr(self, "tip_env_ids_tensor") and self.tip_env_ids_tensor.numel() == 4:
            tips_pos = self.rigid_body_states[:, self.tip_env_ids_tensor, 0:3]  # 用 env 级索引
            tip_count = 4
        else:
            tips_pos = torch.zeros((self.num_envs, 1, 3), dtype=torch.float, device=self.device)
            tip_count = 0

        # 估计“来自桌面”的法向力（近似）：物体各刚体的 +Z 净接触力求和，
        # 并在“靠近桌面”时才计入（减少手指接触的干扰）
        obj_forces = self.net_contact_forces[:, self.object_rb_handles, :]  # [N, n_obj_rb, 3]
        fn_z = torch.clamp(obj_forces[..., 2], min=0.0).sum(dim=1)          # [N]
        near_table = (self.object_pos[:, 2] < (self.table_height + 0.03)).float()
        fn_est_z = fn_z * near_table                                        # [N]
        # print(f"[DEBUG] Table contact force: {self.table_contact_force}")
        # —— 桌面合力向量（用于 tb_contact 惩罚），table 的刚体在 env 中最前面 —— 
        # tb_vec = self.net_contact_forces[:, :self._table_rb_count, :].sum(dim=1)  # [N,3]
        # —— 桌面接触强度（逐刚体绝对值，避免相互抵消）——
        # table_cf: [N, n_tb, 3] 只取“桌子”的刚体
        # table_cf = self.net_contact_forces[:, self.table_env_rb_ids_tensor, :]

        # # L1 强度：先按轴取绝对值，再跨 (刚体, 轴) 求和 -> [N]
        # tb_strength = table_cf.abs().sum(dim=[1, 2])

        # # 二值接触：thr 来自 YAML
        # in_contact = tb_strength > self.tb_cf_thr

        # 关节量（若无 dof_force_tensor，则传 0 并置 has_tau=0）
        qdot = self.shadow_hand_dof_vel
        if hasattr(self, "dof_force_tensor"):
            tau = self.dof_force_tensor
            has_tau = 1
        else:
            tau = torch.zeros_like(qdot)
            has_tau = 0

        # # 用 JIT 奖励计算
        # (rew, resets_base, theta_out, succ_hold_out,
        # theta_step, dev_angle, r_rot, r_dist, r_push, r_table, r_fall) = compute_spin_reward_axis_jit(
        #     self.object_pos, self.object_rot, self.last_object_rot,
        #     self.k_world, self.v1_basis, self.v2_basis,
        #     tips_pos, tip_count,
        #     qdot, tau, has_tau,
        #     self.theta_buf, self.obj_xy_ref,
        #     float(self.c1), float(self.p_bar), float(self.z_min),
        #     int(self.use_theta_goal), self.theta_goal, self.delta_theta_tol, float(self.delta_xy_tol),
        #     float(self.w_succ), float(self.w_rot), float(self.w_err), float(self.w_dist), float(self.w_push),
        #     float(self.w_table), float(self.w_fall), float(self.w_work), float(self.w_torque), float(self.w_pen), float(self.w_dev),
        #     fn_est_z, float(self.table_force_thr),
        #     succ_hold_in=self.succ_hold_count, hold_steps=int(self.success_hold_steps),
        #     table_height=float(self.table_height)
        # )
        # —— 用 JIT 奖励计算 —— 
        (rew, resets_base, theta_out, succ_hold_out,
         theta_step, dev_angle, r_rot, r_push, r_table, r_fall,
         r_ftip, r_energy, r_tb) = compute_spin_reward_axis_jit(
            self.object_pos, self.object_rot, self.last_object_rot,
            self.k_world, self.v1_basis, self.v2_basis,
            tips_pos, tip_count,
            qdot, tau, has_tau,
            self.theta_buf, self.obj_xy_ref,
            float(self.c1), float(self.p_bar), float(self.z_min),
            int(self.use_theta_goal), self.theta_goal, self.delta_theta_tol, float(self.delta_xy_tol),
            float(self.w_succ), float(self.w_rot), float(self.w_err), float(self.w_dist), float(self.w_push),
            float(self.w_table), float(self.w_fall), float(self.w_work), float(self.w_torque), float(self.w_pen), float(self.w_dev),
            fn_est_z, float(self.table_force_thr),
            succ_hold_in=self.succ_hold_count, hold_steps=int(self.success_hold_steps),
            table_height=float(self.table_height),
            # === 新增参数 ===
            ftip_reward_scale=float(self.ftip_reward_scale),
            energy_scale=float(self.energy_scale),
            clip_energy_reward=int(self.clip_energy_reward),
            energy_upper_bound=float(self.energy_upper_bound),
            penalize_tb_contact=int(self.penalize_tb_contact),
            tb_strength=self.table_contact_force if self.penalize_tb_contact else None, # tb_strength, # float(table_cf),                         # [N,3]
            tb_cf_thr=float(self.tb_cf_thr),
            tb_cf_scale=float(self.tb_cf_scale),
        )

        # 超时并入 reset
        timed_out = self.progress_buf >= self.max_episode_length - 1
        resets = torch.where(timed_out, torch.ones_like(resets_base), resets_base)

        # 写回 buffer
        self.rew_buf[:]   = rew
        self.reset_buf[:] = resets.float()

        # 维护状态
        self.theta_buf[:]        = theta_out
        self.succ_hold_count[:]  = succ_hold_out
        self.last_object_rot[:]  = self.object_rot

        # 统计项（TensorBoard）
        self.extras["theta_step"]  = theta_step.mean()
        self.extras["Theta_total"] = self.theta_buf.mean()
        self.extras["dev_angle"]   = dev_angle.mean()
        self.extras["r_rot"]       = r_rot.mean()
        self.extras["r_push"]      = r_push.mean()
        self.extras["r_table"]     = r_table.mean()
        self.extras["r_fall"]      = r_fall.mean()

        self.extras["r_ftip"]   = r_ftip.mean()
        self.extras["r_energy"] = r_energy.mean()
        self.extras["r_tb"]     = r_tb.mean()
        # self.extras["tb_strength"] = tb_strength.mean()
        # self.extras["tb_contact_rate"] = in_contact.float().mean()
    
    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        if self.obs_type == "full_no_vel":
            self.compute_full_observations(True)
        elif self.obs_type == "full":
            self.compute_full_observations()
        elif self.obs_type == "full_state":
             self.compute_full_state()
        else:
            print("Unknown observations type!")

        if self.asymmetric_obs:
            self.compute_full_state(True)

    def compute_full_observations(self, no_vel=False):
        if no_vel:
            self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                                   self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)

            self.obs_buf[:, 16:23] = self.object_pose
            self.obs_buf[:, 23:30] = self.goal_pose
            self.obs_buf[:, 30:34] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            self.obs_buf[:, 34:50] = self.actions
            
        else:
            self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                                   self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            self.obs_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel

            # 2*16 = 32 -16
            self.obs_buf[:, 32:39] = self.object_pose
            self.obs_buf[:, 39:42] = self.object_linvel
            self.obs_buf[:, 42:45] = self.vel_obs_scale * self.object_angvel

            self.obs_buf[:, 45:52] = self.goal_pose
            self.obs_buf[:, 52:56] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            # self.obs_buf[:, 56:72] = self.actions
            act_start = 56
            self.obs_buf[:, act_start:act_start + self.num_actions] = self.actions

    def compute_full_state(self, asymm_obs=False):
        if asymm_obs:
            self.states_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                                      self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            self.states_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
            self.states_buf[:, 2*self.num_shadow_hand_dofs:3*self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

            obj_obs_start = 3*self.num_shadow_hand_dofs  # 48
            self.states_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
            self.states_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
            self.states_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

            goal_obs_start = obj_obs_start + 13  # 61
            self.states_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
            self.states_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            fingertip_obs_start = goal_obs_start + 11  # 72

            # obs_end = 96 + 65 + 30 = 191
            # obs_total = obs_end + num_actions = 72 + 16 = 88
            obs_end = fingertip_obs_start
            self.states_buf[:, obs_end:obs_end + self.num_actions] = self.actions
        else:
            self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                                      self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            self.obs_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
            self.obs_buf[:, 2*self.num_shadow_hand_dofs:3*self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

            obj_obs_start = 3*self.num_shadow_hand_dofs  # 48
            self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
            self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
            self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

            goal_obs_start = obj_obs_start + 13  # 61
            self.obs_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
            self.obs_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            fingertip_obs_start = goal_obs_start + 11  # 72

            # obs_end = 96 + 65 + 30 = 191
            # obs_total = obs_end + num_actions = 72 + 16 = 88
            obs_end = fingertip_obs_start #+ num_ft_states + num_ft_force_torques
            self.obs_buf[:, obs_end:obs_end + self.num_actions] = self.actions

    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    def reset_idx(self, env_ids, goal_env_ids):
        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        if self.object_type == "pen":
            rand_angle_y = torch.tensor(0.3)
            new_object_rot = randomize_rotation_pen(rand_floats[:, 3], rand_floats[:, 4], rand_angle_y,
                                                    self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids], self.z_unit_tensor[env_ids])

        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.goal_object_indices[goal_env_ids]]).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset random force probabilities
        self.random_force_prob[env_ids] = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                                    * torch.rand(len(env_ids), device=self.device) + torch.log(self.force_prob_range[1]))

        # reset shadow hand
        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (rand_floats[:, 5:5+self.num_shadow_hand_dofs] + 1)

        pos = self.shadow_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta
        self.shadow_hand_dof_pos[env_ids, :] = pos
        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_shadow_hand_dofs:5+self.num_shadow_hand_dofs*2]


        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        # 直接用我们在 _create_envs 末尾得到的 q_init
        pos = self.shadow_hand_default_dof_pos.unsqueeze(0).repeat(len(env_ids), 1)

        # 写到视图（会同步到底层 self.dof_state）
        self.shadow_hand_dof_pos[env_ids, :] = pos
        self.shadow_hand_dof_vel[env_ids, :] = 0.0

        # 同步 position target（用于位置控制/相对控制的基线）
        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids,  :self.num_shadow_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)

        # 推给仿真：先 State（硬写位置/速度），再 Target（控制目标）
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(hand_indices), len(env_ids)
        )
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.cur_targets),
            gymtorch.unwrap_tensor(hand_indices), len(env_ids)
        )

        # 本地张量与仿真同步（可选但建议）
        self.gym.refresh_dof_state_tensor(self.sim)


        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

        # rotation ref and count reset.
        self.obj_xy_ref[env_ids] = self.root_state_tensor[self.object_indices[env_ids], 0:2]
        self.theta_buf[env_ids] = 0.0
        self.last_object_rot[env_ids] = self.root_state_tensor[self.object_indices[env_ids], 3:7]
        self.succ_hold_count[env_ids] = 0.0

        # print("after reset q[0]:", self.shadow_hand_dof_pos[env_ids[0], :].tolist())



    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)

        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        just_reset = None
        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)
            just_reset = env_ids  # 记录刚 reset 的 id

        self.actions = actions.clone().to(self.device)
        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions,
                                                                   self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        # —— 关键：对刚 reset 的 env，本帧不要改 target（避免覆盖）
        if just_reset is not None and len(just_reset) > 0:
            self.cur_targets[just_reset[:, None], self.actuated_dof_indices] = \
                self.prev_targets[just_reset[:, None], self.actuated_dof_indices]

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        # self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)

            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape, device=self.device) * self.object_rb_masses * self.force_scale

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)

    def post_physics_step(self):
        if getattr(self, "_need_verify_once", False):
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            i = 0  # 只看第0个环境
            env_offset = self._table_rb_count

            # 名字得到的 actor/env 索引
            name_actor_ids = self.tip_actor_ids
            name_env_ids   = [env_offset + a for a in name_actor_ids]

            # 你那组“人工验证”的 actor/env 索引
            dbg_actor_ids = self.fingertip_actor_indices
            dbg_env_ids   = [env_offset + a for a in dbg_actor_ids] if dbg_actor_ids else []

            if dbg_env_ids:
                for k, nm in enumerate(self.fingertip_names):
                    # 为了可比，按 k 取（假设顺序对应）。若顺序不对应，你可以在 cfg 里按相同顺序填写 debug 索引
                    a_name = name_actor_ids[k]
                    e_name = name_env_ids[k]
                    a_dbg  = dbg_actor_ids[k]
                    e_dbg  = dbg_env_ids[k]

                    p_name = self.rigid_body_states[i, e_name, 0:3]
                    p_dbg  = self.rigid_body_states[i, e_dbg,  0:3]
                    d = torch.norm(p_name - p_dbg).item()
                    print(f"[VERIFY Δp] {nm:<12} name_actor={a_name:02d} dbg_actor={a_dbg:02d} |Δp|={d:.6f} m")
                    print(f"[DEBUG] Rotation weights={self.w_rot}, table weights={self.penalize_tb_contact}, {self.tb_cf_scale}")
                    # print(f"[DEBUG] Net Contact force {self.net_contact_force}")
                    # print(f"[DEBUG] Table contact force: {self.table_contact_force}")
                    # 在 viewer 里画十字：name(红)、dbg(蓝)
                    L=0.02
                    def cross(p, col):
                        lines = [
                            p[0]-L, p[1],   p[2],  p[0]+L, p[1],   p[2],
                            p[0],   p[1]-L, p[2],  p[0],   p[1]+L, p[2],
                            p[0],   p[1],   p[2]-L,p[0],   p[1],   p[2]+L,
                        ]
                        self.gym.add_lines(self.viewer, self.envs[i], 3, lines, col)
                    cross(self.rigid_body_states[i, e_name, 0:3].cpu().numpy(), [0.95,0.2,0.2])  # 红：名字
                    cross(self.rigid_body_states[i, e_dbg,  0:3].cpu().numpy(), [0.2,0.3,0.95])  # 蓝：索引

            self._need_verify_once = False

        # print(f"[DEBUG] Net Contact force {self.net_contact_force}")
        # print(f"[DEBUG] Table contact force: {self.table_contact_force}")
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                targetx = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                targety = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                targetz = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.goal_pos[i].cpu().numpy() + self.goal_displacement_tensor.cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetz[0], targetz[1], targetz[2]], [0.1, 0.1, 0.85])

                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])

#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_spin_reward_axis_jit(
    obj_pos: torch.Tensor,            # [N,3]
    obj_rot: torch.Tensor,            # [N,4] (x,y,z,w)
    last_obj_rot: torch.Tensor,       # [N,4]
    k_world: torch.Tensor,            # [N,3]
    v1_basis: torch.Tensor,           # [N,3]
    v2_basis: torch.Tensor,           # [N,3]
    tip_pos: torch.Tensor,            # [N,tip,3]
    tip_count: int,
    qdot: torch.Tensor,               # [N,ndof]
    tau: torch.Tensor,                # [N,ndof]
    has_tau: int,
    theta_buf_in: torch.Tensor,       # [N]
    obj_xy_ref: torch.Tensor,         # [N,2]
    c1_clip: float, p_bar: float, z_min: float,
    use_theta_goal: int, theta_goal: torch.Tensor, delta_theta_tol: torch.Tensor, delta_xy_tol: float,
    w_succ: float, w_rot: float, w_err: float, w_dist: float, w_push: float,
    w_table: float, w_fall: float, w_work: float, w_torque: float, w_pen: float, w_dev: float,
    fn_est_z: torch.Tensor,           # [N] 估计 +Z 桌面力（近桌才有值）
    f_thr: float,
    succ_hold_in: torch.Tensor,       # [N]
    hold_steps: int,
    table_height: float,
    # === 新增：ftip/energy/tb_contact ===
    ftip_reward_scale: float,
    energy_scale: float,
    clip_energy_reward: int,
    energy_upper_bound: float,
    penalize_tb_contact: int,
    tb_strength: torch.Tensor,
    # tb_strength: torch.Tensor,             # [N,3] 桌面合力向量
    tb_cf_thr: float,
    tb_cf_scale: float,
):
    # 相对旋转推进
    q_rel = quat_mul(obj_rot, quat_conjugate(last_obj_rot))
    v1_new = quat_apply(q_rel, v1_basis)

    c1 = (v1_new * v1_basis).sum(-1)
    c2 = (v1_new * v2_basis).sum(-1)
    c3 = (v1_new * k_world ).sum(-1)

    c3c = torch.clamp(c3, -1.0, 1.0)
    dev_angle = torch.abs(0.5 * torch.pi - torch.acos(c3c))

    inner = torch.clamp(c1, -1.0, 1.0)
    theta_step = torch.sign(c2 + 1e-12) * torch.acos(inner)
    theta_buf = theta_buf_in + theta_step

    r_rot = torch.clamp(theta_step, -c1_clip, c1_clip)
    r_dev = -dev_angle

    # 目标角/成功维持
    if use_theta_goal == 1:
        err = torch.abs(theta_goal - theta_buf)
        succ_mask = (err < delta_theta_tol) & (torch.norm(obj_pos[:, :2] - obj_xy_ref, p=2, dim=-1) < delta_xy_tol)
        r_err  = -err
        r_succ = succ_mask.float()
        succ_hold = torch.where(succ_mask, succ_hold_in + 1.0, torch.zeros_like(succ_hold_in))
        resets_succ = (succ_hold >= hold_steps).float()
    else:
        r_err = torch.zeros_like(theta_buf)
        r_succ = torch.zeros_like(theta_buf)
        succ_hold = torch.zeros_like(succ_hold_in)
        resets_succ = torch.zeros_like(theta_buf)

    # # —— (旧) 指尖贴近奖励（可把 w_dist 设为 0 以停用）——
    # if tip_count > 0:
    #     d = torch.norm(tip_pos - obj_pos[:, None, :], p=2, dim=-1)  # [N,tip]
    #     r_dist = torch.clamp(1.0 / (1e-3 + d), 0.0, 5.0).mean(dim=-1)
    # else:
    #     r_dist = torch.zeros_like(theta_buf)

    # —— Push-away 约束 —— 
    xy_disp = torch.norm(obj_pos[:, :2] - obj_xy_ref, p=2, dim=-1)
    r_push = -(xy_disp - p_bar).clamp_min(0.0)

    # —— 掉落/桌面力项（原有）——
    r_fall  = -(obj_pos[:, 2] < z_min).float()
    r_table = -torch.clamp(fn_est_z - f_thr, min=0.0)

    # —— 力矩与功率 —— 
    if has_tau == 1:
        r_work   = - (tau.abs() * qdot.abs()).sum(-1)        # 旧：瞬时功率（保留给 w_work=0/非0 的兼容）
        r_torque = - (tau ** 2).sum(-1)
    else:
        r_work   = torch.zeros_like(theta_buf)
        r_torque = torch.zeros_like(theta_buf)

    # —— 新增：2) ftip 距离惩罚（仅 scale<0 生效）——
    if (tip_count > 0) and (ftip_reward_scale < 0.0):
        d_ftip = torch.norm(tip_pos - obj_pos[:, None, :], p=2, dim=-1).mean(dim=-1)  # [N]
        r_ftip = ftip_reward_scale * d_ftip
    else:
        r_ftip = torch.zeros_like(theta_buf)

    # —— 新增：3) 能耗（功率）惩罚 —— 
    if (has_tau == 1) and (energy_scale > 0.0):
        power = (qdot.abs() * tau.abs()).sum(dim=-1)  # [N]
        if clip_energy_reward == 1:
            power = torch.clamp(power, max=energy_upper_bound)
        r_energy = - energy_scale * power
    else:
        r_energy = torch.zeros_like(theta_buf)

    # —— 新增：4) 桌面接触惩罚 —— 
    if penalize_tb_contact == 1 and tb_cf_scale > 0.0:
        in_contact = torch.abs(tb_strength).sum(-1) > tb_cf_thr
        r_tb = - tb_cf_scale * in_contact.to(theta_buf.dtype)
    else:
        r_tb = torch.zeros_like(theta_buf)

    # 非指尖用力占位
    r_pen = torch.zeros_like(theta_buf)

    # —— 汇总（注意：r_ftip / r_energy / r_tb 均已带系数）——
    rew = (
        w_succ   * r_succ   +
        w_rot    * r_rot    +
        w_err    * r_err    +
        # w_dist   * r_dist   +
        w_push   * r_push   +
        w_table  * r_table  +
        w_fall   * r_fall   +
        w_work   * r_work   +   # 若改用 r_energy，可把 w_work 设 0
        w_torque * r_torque +
        w_pen    * r_pen    +
        w_dev    * r_dev    +
        r_ftip + r_energy + r_tb
    )

    resets_base = torch.max(resets_succ, (obj_pos[:,2] < z_min).float())

    return rew, resets_base, theta_buf, succ_hold, theta_step, dev_angle, r_rot, r_push, r_table, r_fall, r_ftip, r_energy, r_tb #, r_dist,



# @torch.jit.script
# def compute_spin_reward_axis_jit(
#     obj_pos: torch.Tensor,            # [N,3]
#     obj_rot: torch.Tensor,            # [N,4] (x,y,z,w)
#     last_obj_rot: torch.Tensor,       # [N,4]
#     k_world: torch.Tensor,            # [N,3]
#     v1_basis: torch.Tensor,           # [N,3]
#     v2_basis: torch.Tensor,           # [N,3]
#     tip_pos: torch.Tensor,            # [N,tip,3]
#     tip_count: int,
#     qdot: torch.Tensor,               # [N,ndof]
#     tau: torch.Tensor,                # [N,ndof]
#     has_tau: int,
#     theta_buf_in: torch.Tensor,       # [N]
#     obj_xy_ref: torch.Tensor,         # [N,2]
#     c1_clip: float, p_bar: float, z_min: float,
#     use_theta_goal: int, theta_goal: torch.Tensor, delta_theta_tol: torch.Tensor, delta_xy_tol: float,
#     w_succ: float, w_rot: float, w_err: float, w_dist: float, w_push: float,
#     w_table: float, w_fall: float, w_work: float, w_torque: float, w_pen: float, w_dev: float,
#     fn_est_z: torch.Tensor,           # [N] 估计的 +Z 桌面法向力
#     f_thr: float,
#     succ_hold_in: torch.Tensor,       # [N] 当前成功维持计数
#     hold_steps: int,
#     table_height: float
# ):
#     # 相对旋转
#     q_rel = quat_mul(obj_rot, quat_conjugate(last_obj_rot))  # [N,4]
#     v1_new = quat_apply(q_rel, v1_basis)                     # [N,3]

#     # 基坐标
#     c1 = (v1_new * v1_basis).sum(-1)  # 与 v1
#     c2 = (v1_new * v2_basis).sum(-1)  # 与 v2
#     c3 = (v1_new * k_world ).sum(-1)  # 与 k

#     # 偏轴 dev = |pi/2 - acos(c3)|
#     c3c = torch.clamp(c3, -1.0, 1.0)
#     dev_angle = torch.abs(0.5 * torch.pi - torch.acos(c3c))

#     # 单步有符号角
#     inner = torch.clamp(c1, -1.0, 1.0)
#     theta_step = torch.sign(c2 + 1e-12) * torch.acos(inner)

#     theta_buf = theta_buf_in + theta_step

#     # 旋转推进（裁剪防爆）
#     r_rot = torch.clamp(theta_step, -c1_clip, c1_clip)
#     r_dev = -dev_angle

#     # 目标角/成功 shaping + 持续 N 步
#     if use_theta_goal == 1:
#         err = torch.abs(theta_goal - theta_buf)
#         # succ_mask = (err < delta_theta_tol) & ((obj_pos[:, :2] - obj_xy_ref).norm(dim=-1) < delta_xy_tol)
#         succ_mask = (err < delta_theta_tol) & (torch.norm(obj_pos[:, :2] - obj_xy_ref, p=2, dim=-1) < delta_xy_tol)
#         r_err  = -err
#         r_succ = succ_mask.float()
#         succ_hold = torch.where(succ_mask, succ_hold_in + 1.0, torch.zeros_like(succ_hold_in))
#         resets_succ = (succ_hold >= hold_steps).float()
#     else:
#         r_err  = torch.zeros_like(theta_buf)
#         r_succ = torch.zeros_like(theta_buf)
#         succ_hold = torch.zeros_like(succ_hold_in)
#         resets_succ = torch.zeros_like(theta_buf)

#     # 指尖贴近
#     if tip_count > 0:
#         # d = (tip_pos - obj_pos[:, None, :]).norm(dim=-1)         # [N,tip]
#         d = torch.norm(tip_pos - obj_pos[:, None, :], p=2, dim=-1)
#         rdist_each = 1.0 / (1e-3 + d)
#         rdist_each = torch.clamp(rdist_each, 0.0, 5.0)
#         r_dist = rdist_each.mean(dim=-1)
#     else:
#         r_dist = torch.zeros_like(theta_buf)

#     # 推离/走位
#     # xy_disp = (obj_pos[:, :2] - obj_xy_ref).norm(dim=-1)
#     xy_disp = torch.norm(obj_pos[:, :2] - obj_xy_ref, p=2, dim=-1)
#     r_push = -(xy_disp - p_bar).clamp_min(0.0)

#     # 掉落
#     r_fall = -(obj_pos[:, 2] < z_min).float()

#     # 桌面接触（仅在靠近桌面时计入，已在 python 侧做 near_table 过滤，这里再安全阈）
#     r_table = -torch.clamp(fn_est_z - f_thr, min=0.0)

#     # 做功能量/力矩
#     if has_tau == 1:
#         r_work   = - (tau.abs() * qdot.abs()).sum(-1)
#         r_torque = - (tau ** 2).sum(-1)
#     else:
#         r_work   = torch.zeros_like(theta_buf)
#         r_torque = torch.zeros_like(theta_buf)

#     # 非指尖用力占位
#     r_pen = torch.zeros_like(theta_buf)

#     # 汇总
#     rew = (
#         w_succ   * r_succ   +
#         w_rot    * r_rot    +
#         w_err    * r_err    +
#         w_dist   * r_dist   +
#         w_push   * r_push   +
#         w_table  * r_table  +
#         w_fall   * r_fall   +
#         w_work   * r_work   +
#         w_torque * r_torque +
#         w_pen    * r_pen    +
#         w_dev    * r_dev
#     )

#     # resets（成功保持N步 或 掉落），超时在 python 侧合并
#     resets_base = torch.max(resets_succ, (obj_pos[:,2] < z_min).float())

#     return rew, resets_base, theta_buf, succ_hold, theta_step, dev_angle, r_rot, r_dist, r_push, r_table, r_fall

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot
