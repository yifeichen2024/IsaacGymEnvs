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


class ShadowHandDown(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
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
        assert self.object_type in ["block", "cylinder", "cylinder_axis", "cylinder_corner", "egg", "pen"]

        self.ignore_z = (self.object_type == "pen")

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "cylinder": "urdf/objects/cylinder.urdf",
            "cylinder_corner": "urdf/objects/cylinder_corner.urdf",
            "cylinder_axis": "urdf/objects/cylinder_axis.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml"
        }

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get("assetFileNameBlock", self.asset_files_dict["block"])
            self.asset_files_dict["cylinder"] = self.cfg["env"]["asset"].get("assetFileNameCylinder", self.asset_files_dict["cylinder"])
            self.asset_files_dict["cylinder_corner"] = self.cfg["env"]["asset"].get("assetFileNameCylinderCorner", self.asset_files_dict["cylinder_corner"])
            self.asset_files_dict["cylinder_axis"] = self.cfg["env"]["asset"].get("assetFileNameCylinderAxis", self.asset_files_dict["cylinder_axis"])
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg", self.asset_files_dict["egg"])
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get("assetFileNamePen", self.asset_files_dict["pen"])

        # can be "openai", "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["openai", "full_no_vel", "full", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")

        print("Obs type:", self.obs_type)

        self.num_obs_dict = {
            "openai": 42,
            "full_no_vel": 77,
            "full": 157,
            "full_state": 211
        }

        self.up_axis = 'z'

        self.fingertips = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal", "robot0:thdistal"]
        self.num_fingertips = len(self.fingertips)

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 211

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = 20

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

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

        # TODO added for the new reward computeation 
        # ==== 自转轴/正交基 ====
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
        self.z_min         = float(self.cfg["env"].get("zMin",  self.table_height if hasattr(self,'table_height') else 0.70))
        self.success_hold_steps = int(self.cfg["env"].get("successHoldSteps", 15))
        self.table_force_thr    = float(self.cfg["env"].get("tableForceThr", 5.0))

        # ==== 权重 ====
        W = self.cfg["env"].get("w", {})
        self.w_rot    = float(W.get("rot",    3.0))
        self.w_err    = float(W.get("err",    1.5))
        self.w_succ   = float(W.get("succ",   5.0))
        self.w_dist   = float(W.get("dist",   0.0))   # 不用时设 0
        self.w_push   = float(W.get("push",   1.0))
        self.w_table  = float(W.get("table",  0.0))
        self.w_fall   = float(W.get("fall",   2.0))
        self.w_work   = float(W.get("work",   1e-3))
        self.w_torque = float(W.get("torque", 1e-4))
        self.w_pen    = float(W.get("pen",    0.5))
        self.w_dev    = float(W.get("dev",    1.0))

        # ==== 扩展奖励项 ====
        R = self.cfg["env"].get("reward", {})
        self.ftip_reward_scale    = float(R.get("ftipRewardScale", 0.0))     # 负值生效
        self.energy_scale         = float(R.get("energyScale", 0.0))         # >0 生效
        self.clip_energy_reward   = int(R.get("clipEnergyReward", 0))
        self.energy_upper_bound   = float(R.get("energyUpperBound", 50.0))
        self.penalize_tb_contact  = int(R.get("penalizeTbContact", 1))
        self.tb_cf_scale          = float(R.get("tbContactScale", 0.5))
        self.tb_cf_thr            = float(R.get("tbContactThr", 0.2))

        # ==== 运行时缓存 ====
        self.theta_buf       = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.obj_xy_ref      = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.last_object_rot = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.succ_hold_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.c1 = min(1.5 * self.dt, 0.25)

        # ==== 接触力张量 + 桌体刚体索引 ====
        nf = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.net_contact_forces = gymtorch.wrap_tensor(nf).view(self.num_envs, -1, 3)

        env0 = self.envs[0]
        table_handle0 = self.gym.find_actor_handle(env0, "table")
        tb_body_names = self.gym.get_actor_rigid_body_names(env0, table_handle0)
        assert len(tb_body_names) >= 1, "table actor 没找到刚体"
        tb_first_body = tb_body_names[0]
        self.table_body_index = self.gym.find_actor_rigid_body_index(env0, table_handle0, tb_first_body, gymapi.DOMAIN_ENV)
        self.table_contact_force = self.net_contact_forces[:, self.table_body_index, :]  # [N,3]

        # ==== 指尖 env 级索引（Shadow 自带 5 个指尖名）====
        self._cache_fingertip_indices()  # 定义见下



        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
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
        self.dt = self.cfg["sim"]["dt"]
        self.up_axis_idx = 2 if self.up_axis == 'z' else 1 # index of up axis: Y=1, Z=2

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)


    def _cache_fingertip_indices(self):
        # 需要在 _create_envs 里已设置 self._table_rb_count
        assert hasattr(self, "_table_rb_count"), "_table_rb_count 未设置"
        # 把 asset 级指尖名映射到当前 env 的 actor/body 索引，再转为 env 级索引
        env0  = self.envs[0]
        hand0 = self.shadow_hands[0]

        tip_actor_ids = []
        for nm in self.fingertips:  # ["robot0:ffdistal", ..., "robot0:thdistal"]
            h_tip = self.gym.find_actor_rigid_body_handle(env0, hand0, nm)
            assert h_tip >= 0, f"未找到指尖刚体: {nm}"
            tip_actor_ids.append(h_tip)

        tip_env_ids = [self._table_rb_count + idx for idx in tip_actor_ids]
        self.tip_env_ids_tensor = to_torch(tip_env_ids, dtype=torch.long, device=self.device)
        

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets'))
        shadow_hand_asset_file = os.path.normpath("mjcf/open_ai_assets/hand/shadow_hand.xml")
        if "asset" in self.cfg["env"]:
            shadow_hand_asset_file = os.path.normpath(self.cfg["env"]["asset"].get("assetFileName", shadow_hand_asset_file))
        object_asset_file = self.asset_files_dict[self.object_type]

        # hand 资产（沿用你现有的选项与 DOF/肌腱设置）
        hand_opts = gymapi.AssetOptions()
        hand_opts.flip_visual_attachments = False
        hand_opts.fix_base_link = True
        hand_opts.collapse_fixed_joints = True
        hand_opts.disable_gravity = True
        hand_opts.thickness = 0.001
        hand_opts.angular_damping = 0.01
        if self.physics_engine == gymapi.SIM_PHYSX:
            hand_opts.use_physx_armature = True
        hand_opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, hand_opts)
        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs   = self.gym.get_asset_dof_count(shadow_hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)
        self.num_shadow_hand_tendons   = self.gym.get_asset_tendon_count(shadow_hand_asset)

         # tendon set up
        limit_stiffness = 30
        t_damping = 0.1
        relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_asset)

        for i in range(self.num_shadow_hand_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(shadow_hand_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping
        self.gym.set_asset_tendon_properties(shadow_hand_asset, tendon_props)

        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)
        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos  = []
        self.shadow_hand_dof_default_vel  = []
        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

        self.actuated_dof_indices = to_torch(
            [self.gym.find_asset_dof_index(shadow_hand_asset,
            self.gym.get_asset_actuator_joint_name(shadow_hand_asset, i))
            for i in range(self.num_shadow_hand_actuators)],
            dtype=torch.long, device=self.device
        )
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos  = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel  = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]

        # create fingertip force sensors, if needed
        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(shadow_hand_asset, ft_handle, sensor_pose)


        # === 新增：桌子资产（用 box），并与 Allegro 一致的位姿/尺寸 ===
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_opts.use_mesh_materials = True
        table_opts.vhacd_enabled = True
        table_sx, table_sy, table_sz = 0.12, 0.12, 0.5
        table_asset = self.gym.create_box(self.sim, table_sx, table_sy, table_sz, table_opts)

        # 物体/目标资产选项：与 Allegro 对齐（物体有重力，目标无重力）
        obj_opts = gymapi.AssetOptions()
        obj_opts.disable_gravity = False
        obj_opts.vhacd_enabled = True
        goal_opts = gymapi.AssetOptions()
        goal_opts.disable_gravity = True
        goal_opts.vhacd_enabled = True

        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, obj_opts)
        goal_asset   = self.gym.load_asset(self.sim, asset_root, object_asset_file, goal_opts)

        # === 位姿：桌子 / 物体 / 手（palm-down） / 目标 ===
        table_center_z = table_sz * 0.5
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, table_center_z)
        self.table_height = table_sz

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.0, 0.0, self.table_height + 0.035)
        object_start_pose.r = gymapi.Quat.from_euler_zyx(np.pi/2, 0.0, 0.0)

        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(-0.36, 0.0, self.table_height + 0.128)
        shadow_hand_start_pose.r = gymapi.Quat.from_euler_zyx(np.pi, 0.0, np.pi*3/2)

        self.goal_displacement = gymapi.Vec3(-0.2, -0.4, 0.12)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement
        goal_start_pose.r = object_start_pose.r

        # === 聚合尺寸（含桌子）并缓存 table 刚体数作为偏移 ===
        hand_rb  = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        hand_sh  = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        obj_rb   = self.gym.get_asset_rigid_body_count(object_asset)
        obj_sh   = self.gym.get_asset_rigid_shape_count(object_asset)
        table_rb = self.gym.get_asset_rigid_body_count(table_asset)
        table_sh = self.gym.get_asset_rigid_shape_count(table_asset)

        self._table_rb_count = table_rb
        max_agg_bodies = hand_rb + obj_rb + obj_rb + table_rb
        max_agg_shapes = hand_sh + obj_sh + obj_sh + table_sh

        # 容器
        self.shadow_hands = []
        self.envs = []
        self.object_init_state = []
        self.hand_start_states = []
        self.hand_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.table_indices = []

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]

        # 物体刚体在 env 张量中的偏移（桌子在最前，然后手，再物体）
        object_rb_offset = table_rb + hand_rb
        self.object_rb_handles = list(range(object_rb_offset, object_rb_offset + obj_rb))

        # === 创建各个 env：桌子 -> 手 -> 物体 -> 目标 ===
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # 1) 桌子
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0, 0)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            for sp in shape_props:
                sp.friction = 0.5
                sp.restitution = 0.5
                sp.rolling_friction = 0.1
                sp.torsion_friction = 0.1
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, shape_props)

            # 2) 手
            shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, -1, 0)
            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)
            self.hand_start_states.append([
                shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z, shadow_hand_start_pose.r.w,
                0, 0, 0, 0, 0, 0
            ])
            if self.obs_type == "full_state" or self.asymmetric_obs:
                self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)

            # 3) 物体
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            self.object_init_state.append([
                object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                0, 0, 0, 0, 0, 0
            ])

            # 4) 目标
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            if self.object_type != "block":
                self.gym.set_rigid_body_color(env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(env_ptr, goal_handle,   0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)

        # 与原逻辑兼容：质量缓存等
        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = to_torch([prop.mass for prop in object_rb_props], dtype=torch.float, device=self.device)

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]
    
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses  = to_torch(self.object_rb_masses,  dtype=torch.float, device=self.device)

        self.hand_indices      = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices    = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)


    def compute_reward(self, actions):
        # 刷新接触力
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # 指尖位置 [N, tip, 3]
        if hasattr(self, "tip_env_ids_tensor") and self.tip_env_ids_tensor.numel() > 0:
            tips_pos = self.rigid_body_states[:, self.tip_env_ids_tensor, 0:3]
            tip_count = tips_pos.shape[1]
        else:
            tips_pos = torch.zeros((self.num_envs, 1, 3), dtype=torch.float, device=self.device)
            tip_count = 0

        # 估计“来自桌面”的 +Z 法向力（只在接近桌面计入）
        obj_forces = self.net_contact_forces[:, self.object_rb_handles, :]  # [N, n_obj_rb, 3]
        fn_z = torch.clamp(obj_forces[..., 2], min=0.0).sum(dim=1)          # [N]
        near_table = (self.object_pos[:, 2] < (self.table_height + 0.03)).float()
        

        fn_est_z = fn_z * near_table                                        # [N]

        # 关节量
        qdot = self.shadow_hand_dof_vel
        if self.obs_type == "full_state" or self.asymmetric_obs:
            tau = self.dof_force_tensor
            has_tau = 1
        else:
            tau = torch.zeros_like(qdot)
            has_tau = 0

        # JIT 奖励
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
            # 扩展项
            ftip_reward_scale=float(self.ftip_reward_scale),
            energy_scale=float(self.energy_scale),
            clip_energy_reward=int(self.clip_energy_reward),
            energy_upper_bound=float(self.energy_upper_bound),
            penalize_tb_contact=int(self.penalize_tb_contact),
            tb_strength=self.table_contact_force if self.penalize_tb_contact else torch.zeros_like(self.net_contact_forces[:,0,:]),
            tb_cf_thr=float(self.tb_cf_thr),
            tb_cf_scale=float(self.tb_cf_scale),
        )

        # 合并超时
        timed_out = self.progress_buf >= self.max_episode_length - 1
        resets = torch.where(timed_out, torch.ones_like(resets_base), resets_base)

        # 写回
        self.rew_buf[:]   = rew
        self.reset_buf[:] = resets.float()

        # 维护状态
        self.theta_buf[:]       = theta_out
        self.succ_hold_count[:] = succ_hold_out
        self.last_object_rot[:] = self.object_rot

        # 统计
        self.extras["theta_step"]  = theta_step.mean()
        self.extras["Theta_total"] = self.theta_buf.mean()
        self.extras["dev_angle"]   = dev_angle.mean()
        self.extras["r_rot"]       = r_rot.mean()
        self.extras["r_push"]      = r_push.mean()
        self.extras["r_table"]     = r_table.mean()
        self.extras["r_fall"]      = r_fall.mean()
        self.extras["r_ftip"]      = r_ftip.mean()
        self.extras["r_energy"]    = r_energy.mean()
        self.extras["r_tb"]        = r_tb.mean()
        # print(f"[DEBUG] r_table: {r_table.mean()}")

    
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

        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        if self.obs_type == "openai":
            self.compute_fingertip_observations(True)
        elif self.obs_type == "full_no_vel":
            self.compute_full_observations(True)
        elif self.obs_type == "full":
            self.compute_full_observations()
        elif self.obs_type == "full_state":
            self.compute_full_state()
        else:
            print("Unknown observations type!")

        if self.asymmetric_obs:
            self.compute_full_state(True)

    def compute_fingertip_observations(self, no_vel=False):
        if no_vel:
            # Per https://arxiv.org/pdf/1808.00177.pdf Table 2
            #   Fingertip positions
            #   Object Position, but not orientation
            #   Relative target orientation

            # 3*self.num_fingertips = 15
            self.obs_buf[:, 0:15] = self.fingertip_pos.reshape(self.num_envs, 15)
            self.obs_buf[:, 15:18] = self.object_pose[:, 0:3]
            self.obs_buf[:, 18:22] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            self.obs_buf[:, 22:42] = self.actions
        else:
            # 13*self.num_fingertips = 65
            self.obs_buf[:, 0:65] = self.fingertip_state.reshape(self.num_envs, 65)
            self.obs_buf[:, 65:72] = self.object_pose
            self.obs_buf[:, 72:75] = self.object_linvel
            self.obs_buf[:, 75:78] = self.vel_obs_scale * self.object_angvel

            self.obs_buf[:, 78:85] = self.goal_pose
            self.obs_buf[:, 85:89] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            self.obs_buf[:, 89:109] = self.actions

    def compute_full_observations(self, no_vel=False):
        if no_vel:
            self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                                   self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)

            self.obs_buf[:, 24:31] = self.object_pose
            self.obs_buf[:, 31:38] = self.goal_pose
            self.obs_buf[:, 38:42] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            # 3*self.num_fingertips = 15
            self.obs_buf[:, 42:57] = self.fingertip_pos.reshape(self.num_envs, 15)

            self.obs_buf[:, 57:77] = self.actions
        else:
            self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                                   self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            self.obs_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel

            self.obs_buf[:, 48:55] = self.object_pose
            self.obs_buf[:, 55:58] = self.object_linvel
            self.obs_buf[:, 58:61] = self.vel_obs_scale * self.object_angvel

            self.obs_buf[:, 61:68] = self.goal_pose
            self.obs_buf[:, 68:72] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            # 13*self.num_fingertips = 65
            self.obs_buf[:, 72:137] = self.fingertip_state.reshape(self.num_envs, 65)

            self.obs_buf[:, 137:157] = self.actions

    def compute_full_state(self, asymm_obs=False):
        if asymm_obs:
            self.states_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                                      self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            self.states_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
            self.states_buf[:, 2*self.num_shadow_hand_dofs:3*self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

            obj_obs_start = 3*self.num_shadow_hand_dofs  # 72
            self.states_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
            self.states_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
            self.states_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

            goal_obs_start = obj_obs_start + 13  # 85
            self.states_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
            self.states_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            # fingertip observations, state(pose and vel) + force-torque sensors
            num_ft_states = 13 * self.num_fingertips  # 65
            num_ft_force_torques = 6 * self.num_fingertips  # 30

            fingertip_obs_start = goal_obs_start + 11  # 96
            self.states_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
            self.states_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states +
                            num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor

            # obs_end = 96 + 65 + 30 = 191
            # obs_total = obs_end + num_actions = 211
            obs_end = fingertip_obs_start + num_ft_states + num_ft_force_torques
            self.states_buf[:, obs_end:obs_end + self.num_actions] = self.actions
        else:
            self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                                   self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            self.obs_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
            self.obs_buf[:, 2*self.num_shadow_hand_dofs:3*self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

            obj_obs_start = 3*self.num_shadow_hand_dofs  # 72
            self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
            self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
            self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

            goal_obs_start = obj_obs_start + 13  # 85
            self.obs_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
            self.obs_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            # fingertip observations, state(pose and vel) + force-torque sensors
            num_ft_states = 13 * self.num_fingertips  # 65
            num_ft_force_torques = 6 * self.num_fingertips  # 30

            fingertip_obs_start = goal_obs_start + 11  # 96
            self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
            self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states +
                         num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor

            # obs_end = 96 + 65 + 30 = 191
            # obs_total = obs_end + num_actions = 211
            obs_end = fingertip_obs_start + num_ft_states + num_ft_force_torques
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
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

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

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

        # rotation ref / counters
        self.obj_xy_ref[env_ids] = self.root_state_tensor[self.object_indices[env_ids], 0:2]
        self.theta_buf[env_ids] = 0.0
        self.last_object_rot[env_ids] = self.root_state_tensor[self.object_indices[env_ids], 3:7]
        self.succ_hold_count[env_ids] = 0.0

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # if goals need reset in addition to other envs, call set API in reset_idx()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)

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

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)

            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape, device=self.device) * self.object_rb_masses * self.force_scale

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)

    def post_physics_step(self):
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
    obj_pos: torch.Tensor,
    obj_rot: torch.Tensor,
    last_obj_rot: torch.Tensor,
    k_world: torch.Tensor,
    v1_basis: torch.Tensor,
    v2_basis: torch.Tensor,
    tip_pos: torch.Tensor,
    tip_count: int,
    qdot: torch.Tensor,
    tau: torch.Tensor,
    has_tau: int,
    theta_buf_in: torch.Tensor,
    obj_xy_ref: torch.Tensor,
    c1_clip: float, p_bar: float, z_min: float,
    use_theta_goal: int, theta_goal: torch.Tensor, delta_theta_tol: torch.Tensor, delta_xy_tol: float,
    w_succ: float, w_rot: float, w_err: float, w_dist: float, w_push: float,
    w_table: float, w_fall: float, w_work: float, w_torque: float, w_pen: float, w_dev: float,
    fn_est_z: torch.Tensor, f_thr: float,
    succ_hold_in: torch.Tensor, hold_steps: int, table_height: float,
    # 扩展
    ftip_reward_scale: float,
    energy_scale: float,
    clip_energy_reward: int,
    energy_upper_bound: float,
    penalize_tb_contact: int,
    tb_strength: torch.Tensor,
    tb_cf_thr: float,
    tb_cf_scale: float,
):
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

    xy_disp = torch.norm(obj_pos[:, :2] - obj_xy_ref, p=2, dim=-1)
    r_push = -(xy_disp - p_bar).clamp_min(0.0)

    r_fall  = -(obj_pos[:, 2] < z_min).float()
    r_table = -torch.clamp(fn_est_z - f_thr, min=0.0)

    if has_tau == 1:
        r_work   = - (tau.abs() * qdot.abs()).sum(-1)
        r_torque = - (tau ** 2).sum(-1)
    else:
        r_work   = torch.zeros_like(theta_buf)
        r_torque = torch.zeros_like(theta_buf)

    if (tip_count > 0) and (ftip_reward_scale < 0.0):
        d_ftip = torch.norm(tip_pos - obj_pos[:, None, :], p=2, dim=-1).mean(dim=-1)
        r_ftip = ftip_reward_scale * d_ftip
    else:
        r_ftip = torch.zeros_like(theta_buf)

    if (has_tau == 1) and (energy_scale > 0.0):
        power = (qdot.abs() * tau.abs()).sum(dim=-1)
        if clip_energy_reward == 1:
            power = torch.clamp(power, max=energy_upper_bound)
        r_energy = - energy_scale * power
    else:
        r_energy = torch.zeros_like(theta_buf)

    if penalize_tb_contact == 1 and tb_cf_scale > 0.0:
        in_contact = torch.abs(tb_strength).sum(-1) > tb_cf_thr
        r_tb = - tb_cf_scale * in_contact.to(theta_buf.dtype)
    else:
        r_tb = torch.zeros_like(theta_buf)

    r_pen = torch.zeros_like(theta_buf)

    rew = (
        w_succ   * r_succ   +
        w_rot    * r_rot    +
        w_err    * r_err    +
        w_dist   * torch.zeros_like(theta_buf) +  # 保留接口
        w_push   * r_push   +
        w_table  * r_table  +
        w_fall   * r_fall   +
        w_work   * r_work   +
        w_torque * r_torque +
        w_pen    * r_pen    +
        w_dev    * r_dev    +
        r_ftip + r_energy + r_tb
    )

    resets_base = torch.max(resets_succ, (obj_pos[:,2] < z_min).float())
    return rew, resets_base, theta_buf, succ_hold, theta_step, dev_angle, r_rot, r_push, r_table, r_fall, r_ftip, r_energy, r_tb


@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
):
    # Distance from the hand to the object
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(progress_buf >= max_episode_length - 1, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot
