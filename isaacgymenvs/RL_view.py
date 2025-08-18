# -*- coding: utf-8 -*-
from isaacgym import gymapi
import os, math

# ===================== 仅改这里 =====================
ASSET_ROOT = "/home/chen/IsaacGymEnvs/assets/mjcf/open_ai_assets" # os.path.abspath(os.path.join(os.path.dirname(__file__), "../../open_ai_assets"))
SHADOW_HAND_ASSET_FILE = "hand/shadow_hand.xml"

OBJ_ASSET_ROOT = "/home/chen/IsaacGymEnvs/assets/urdf/objects"
OBJECT_TYPE = "block"  # 可选: block | egg | pen
OBJ_FILE_MAP = {
    "block":   "cube_multicolor.urdf",
    "egg":     "mjcf/open_ai_assets/hand/egg.xml",
    "pen":     "mjcf/open_ai_assets/hand/pen.xml",
}
OBJECT_ASSET_FILE = OBJ_FILE_MAP[OBJECT_TYPE]
# ====================================================

gym = gymapi.acquire_gym()

# ---- sim ----
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = False
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    raise RuntimeError("Failed to create sim")

# ---- ground ----
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# ---- env ----
env = gym.create_env(sim, gymapi.Vec3(-1, -1, -1), gymapi.Vec3(1, 1, 1), 1)

# ---- assets ----
# hand
hand_opt = gymapi.AssetOptions()
hand_opt.fix_base_link = True
hand_opt.disable_gravity = True
hand_opt.default_dof_drive_mode = gymapi.DOF_MODE_NONE
if sim_params.use_gpu_pipeline:
    hand_opt.use_physx_armature = True
hand_asset = gym.load_asset(sim, ASSET_ROOT, SHADOW_HAND_ASSET_FILE, hand_opt)

# object (动态)
obj_opt = gymapi.AssetOptions()
obj_opt.disable_gravity = False
obj_opt.vhacd_enabled = True
object_asset = gym.load_asset(sim, OBJ_ASSET_ROOT, OBJECT_ASSET_FILE, obj_opt)

# goal (静态可视)
goal_opt = gymapi.AssetOptions()
goal_opt.disable_gravity = True
goal_opt.vhacd_enabled = True
goal_asset = gym.load_asset(sim, OBJ_ASSET_ROOT, OBJECT_ASSET_FILE, goal_opt)

# table 用 box，尺寸与任务一致
table_opt = gymapi.AssetOptions()
table_opt.fix_base_link = True
table_opt.use_mesh_materials = True
table_sx, table_sy, table_sz = 0.12, 0.12, 0.5
table_asset = gym.create_box(sim, table_sx, table_sy, table_sz, table_opt)

# ---- placement 与 ShadowHandDown 保持一致 ----
table_center_z = table_sz * 0.5
table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.0, 0.0, table_center_z)
table_actor = gym.create_actor(env, table_asset, table_pose, "table", 0, 0)

object_pose = gymapi.Transform()
object_pose.p = gymapi.Vec3(0.0, 0.0, table_sz + 0.04)
object_pose.r = gymapi.Quat.from_euler_zyx(math.pi/2, 0.0, 0.0)
object_actor = gym.create_actor(env, object_asset, object_pose, "object", 0, 0)

hand_pose = gymapi.Transform()
hand_pose.p = gymapi.Vec3(-0.35, 0.0, table_sz + 0.125)
# ShadowHandDown: from_euler_zyx(z=pi, y=0, x=3*pi/2)
hand_pose.r = gymapi.Quat.from_euler_zyx(math.pi, 0.0, 1.5*math.pi)
hand_actor = gym.create_actor(env, hand_asset, hand_pose, "hand", 0, -1)

goal_pose = gymapi.Transform()
goal_pose.p = gymapi.Vec3(object_pose.p.x - 0.2, object_pose.p.y - 0.4, object_pose.p.z + 0.12)
goal_pose.r = object_pose.r
goal_actor = gym.create_actor(env, goal_asset, goal_pose, "goal_object", 1, 0)

# ---- 列表与映射：asset -> actor(handle) -> env(body_index) ----
hand_asset_rb_cnt = gym.get_asset_rigid_body_count(hand_asset)
obj_asset_rb_cnt  = gym.get_asset_rigid_body_count(object_asset)
table_asset_rb_cnt = gym.get_asset_rigid_body_count(table_asset)  # 一般为 1

print("\n[HAND asset rigid bodies]")
for i in range(hand_asset_rb_cnt):
    nm = gym.get_asset_rigid_body_name(hand_asset, i)
    print(f"  asset[{i:02d}] {nm}")

# env 级偏移：创建顺序 table -> hand -> object -> goal
table_rb_count = table_asset_rb_cnt
hand_env_offset = table_rb_count
obj_env_offset  = table_rb_count + hand_asset_rb_cnt

print(f"\n[Offsets] table_rb_count={table_rb_count}, hand_env_offset={hand_env_offset}, obj_env_offset={obj_env_offset}")

# 目标指尖名（与任务一致）
FINGERTIPS = ["robot0:ffdistal","robot0:mfdistal","robot0:rfdistal","robot0:lfdistal","robot0:thdistal"]

# name -> index（actor 域 / env 域）
tip_actor_idx = []
tip_env_idx   = []
for nm in FINGERTIPS:
    ia = gym.find_actor_rigid_body_index(env, hand_actor, nm, gymapi.DOMAIN_ACTOR)
    ie = gym.find_actor_rigid_body_index(env, hand_actor, nm, gymapi.DOMAIN_ENV)
    if ia < 0 or ie < 0:
        print(f"  ✗ {nm:<18} NOT FOUND")
        continue
    tip_actor_idx.append(ia)
    tip_env_idx.append(ie)
    print(f"  ✓ {nm:<18} actor={ia:02d} -> env={ie:02d}")

print("\n[Tips] actor_idx:", tip_actor_idx)
print("[Tips] env_idx  :", tip_env_idx)

# 给指尖上色（actor 域索引）
for ia in tip_actor_idx:
    gym.set_rigid_body_color(env, hand_actor, ia, gymapi.MESH_VISUAL, gymapi.Vec3(0.95, 0.2, 0.2))

# 也给 object/goal 上色，易区分
gym.set_rigid_body_color(env, object_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.8, 0.3))
gym.set_rigid_body_color(env, goal_actor,   0, gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.3, 0.95))

# ---- viewer ----
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise RuntimeError("Failed to create viewer")

cam_pos = gymapi.Vec3(1.2, 1.2, 1.2)
cam_target = gymapi.Vec3(0.0, 0.0, 0.6)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# print("\n[Tips] actor_handles:", tip_actor_handles)
# print("[Tips] env_indices  :", tip_env_indices)
print("\n提示：若红色十字没落在指尖，请核对 FINGERTIPS 名称或 hand_pose 旋转。")

# ---- loop ----
L = 0.02
while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)

    # 清线，避免累积
    gym.clear_lines(viewer)

    # 画目标与物体坐标轴
    def draw_axes(pose, env_ref, color):
        # 简单画十字
        lines = [
            pose.p.x-L, pose.p.y,   pose.p.z,  pose.p.x+L, pose.p.y,   pose.p.z,
            pose.p.x,   pose.p.y-L, pose.p.z,  pose.p.x,   pose.p.y+L, pose.p.z,
            pose.p.x,   pose.p.y,   pose.p.z-L,pose.p.x,   pose.p.y,   pose.p.z+L,
        ]
        gym.add_lines(viewer, env_ref, 3, lines, color)

    draw_axes(object_pose, env, [0.2, 0.8, 0.3])
    draw_axes(goal_pose,   env, [0.2, 0.3, 0.95])

    # 用 actor 级状态直接取指尖位置并画十字（免去 env 偏移易错）
    rb_states = gym.get_actor_rigid_body_states(env, hand_actor, gymapi.STATE_POS)
    for h in tip_actor_idx:
        p = rb_states['pose']['p'][h]
        lines = [
            p[0]-L, p[1],   p[2],  p[0]+L, p[1],   p[2],
            p[0],   p[1]-L, p[2],  p[0],   p[1]+L, p[2],
            p[0],   p[1],   p[2]-L,p[0],   p[1],   p[2]+L,
        ]
        gym.add_lines(viewer, env, 3, lines, [0.95, 0.1, 0.1])

    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

# from isaacgym import gymapi
# import math

# # ===================== 用户需要改的 3 个路径 =====================
# HAND_ASSET_ROOT = "/home/chen/IsaacGymEnvs/assets/urdf/allegro_hand_v4/allegro_hand_description"
# HAND_ASSET_FILE = "allegro_hand_right.urdf"

# TABLE_ASSET_ROOT = "/home/chen/IsaacGymEnvs/assets/urdf"   # 示例：你自己的table目录
# TABLE_ASSET_FILE = "table_small.urdf"

# OBJ_ASSET_ROOT = "/home/chen/IsaacGymEnvs/assets/urdf/objects"        # 示例：你自己的object目录
# OBJ_ASSET_FILE = "cube_multicolor.urdf"
# # ===============================================================

# gym = gymapi.acquire_gym()

# # 仿真参数
# sim_params = gymapi.SimParams()
# sim_params.up_axis = gymapi.UP_AXIS_Z
# sim_params.dt = 1.0/60.0
# sim_params.substeps = 2
# sim_params.use_gpu_pipeline = False

# sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
# if sim is None:
#     raise RuntimeError("Failed to create sim")

# # 地面
# plane_params = gymapi.PlaneParams()
# plane_params.normal = gymapi.Vec3(0, 0, 1)
# gym.add_ground(sim, plane_params)

# # 环境
# env = gym.create_env(sim, gymapi.Vec3(-1, 0, -1), gymapi.Vec3(1, 0, 1), 1)

# # 载入 Hand（固定底座，方便观察）
# hand_opt = gymapi.AssetOptions()
# hand_opt.fix_base_link = True
# hand_opt.use_mesh_materials = True
# hand_asset = gym.load_asset(sim, HAND_ASSET_ROOT, HAND_ASSET_FILE, hand_opt)

# # 载入 Table（建议固定底座，作为静态支撑面）
# table_opt = gymapi.AssetOptions()
# table_opt.fix_base_link = True
# table_opt.use_mesh_materials = True
# table_asset = gym.load_asset(sim, TABLE_ASSET_ROOT, TABLE_ASSET_FILE, table_opt)

# # 载入 Object（不固定，让它能落在桌子上）
# obj_opt = gymapi.AssetOptions()
# obj_opt.fix_base_link = False
# obj_opt.use_mesh_materials = True
# obj_asset = gym.load_asset(sim, OBJ_ASSET_ROOT, OBJ_ASSET_FILE, obj_opt)

# # ---- 放置 Table ----
# # 如果你的 table URDF 是以桌面中心为原点且高度大约 ~0.72m，下行可设 z=0.36；不确定高度就先放 0.35-0.4，观察后再微调
# table_pose = gymapi.Transform()
# table_pose.p = gymapi.Vec3(0.0, 0.0, 0.36)  # 桌面高度 ~0.72 的一半
# table_actor = gym.create_actor(env, table_asset, table_pose, "box", 0, 0)

# # ---- 放置 Object 在桌面上 ----
# # 先粗略假设小方块放在桌面中心，z 稍微高一点避免初始穿模；渲染后再微调
# obj_pose = gymapi.Transform()
# obj_pose.p = gymapi.Vec3(0.0, 0.0, 0.8)    # 桌面上方一点点，避免初始穿插；按你的桌面实际高度调整
# obj_actor = gym.create_actor(env, obj_asset, obj_pose, "object", 0, 0)

# # ---- 放置 Hand（朝 -Z）----
# hand_pose = gymapi.Transform()
# hand_pose.p = gymapi.Vec3(-0.08, 0.0, 0.75)  # 在桌面上方、稍微后撤一点，避免一上来就顶住物体

# # 【关键】把手心从“朝 +X”转成“朝 -Z”
# # 需求：绕 +Y 轴“逆时针”旋转 90°（右手定则），在 Isaac 的 from_euler_zyx( z, y, x ) 里就是 y=+pi/2
# hand_pose.r = gymapi.Quat.from_euler_zyx(0.0, math.pi/2, 0.0)

# # 如果你的手原始朝向并非严格 +X，或者结果反了，可以把角度换成 -math.pi/2 试一下：
# # hand_pose.r = gymapi.Quat.from_euler_zyx(0.0, -math.pi/2, 0.0)

# hand_actor = gym.create_actor(env, hand_asset, hand_pose, "allegro_hand_right", 0, 0)

# # ===== 列出 asset 级刚体名，并把名字映射到 actor 级 handle =====
# asset_rb_count = gym.get_asset_rigid_body_count(hand_asset)
# print("\n[Asset rigid bodies] count =", asset_rb_count)
# asset_rb_names = []
# for i in range(asset_rb_count):
#     nm = gym.get_asset_rigid_body_name(hand_asset, i)
#     asset_rb_names.append(nm)
#     print(f"  {i:02d}  {nm}")

# # 想要的 tip 名（或自动从 asset 名里筛 *_tip）
# # CANDIDATE_TIPS = ["link_3_tip", "link_7_tip", "link_11_tip", "link_15_tip"]
# CANDIDATE_TIPS = [n for n in asset_rb_names if n.endswith("_tip")]

# # 用名字在 actor 上找 handle
# actor_rb_count = gym.get_actor_rigid_body_count(env, hand_actor)
# tip_handles = []
# print("\n[Tip lookup -> actor handles]")
# for nm in CANDIDATE_TIPS:
#     h = gym.find_actor_rigid_body_handle(env, hand_actor, nm)
#     if h < 0:
#         print(f"  ✗ {nm:<16} -> NOT FOUND")
#     elif h >= actor_rb_count:
#         print(f"  ⚠ {nm:<16} -> handle {h} >= actor_rb_count {actor_rb_count} (异常)")
#     else:
#         tip_handles.append(h)
#         print(f"  ✓ {nm:<16} -> handle {h}")

# print(f"[Summary] tip_count = {len(tip_handles)}; handles = {tip_handles}")

# # ===== 直接用你指定的 4 个刚体索引（actor 级 handle）=====
# TIP_IDX = [6, 11, 16, 21]

# actor_rb_count = gym.get_actor_rigid_body_count(env, hand_actor)
# print(f"\n[Actor rigid bodies] count = {actor_rb_count}")
# print(f"[Use fixed tip indices] {TIP_IDX}")

# # 越界保护（如果 URDF 变了会直接提示）
# bad = [i for i in TIP_IDX if not (0 <= i < actor_rb_count)]
# if bad:
#     raise RuntimeError(f"这些索引越界了: {bad}（actor_rb_count={actor_rb_count}）。请确认 URDF 刚体顺序没变。")

# # 给 tip 上色（仅这四个）
# for h in TIP_IDX:
#     gym.set_rigid_body_color(env, hand_actor, h, gymapi.MESH_VISUAL, gymapi.Vec3(0.95, 0.2, 0.2))

# # Viewer
# viewer = gym.create_viewer(sim, gymapi.CameraProperties())
# if viewer is None:
#     raise RuntimeError("Failed to create viewer")

# # 相机对准桌面与手
# cam_pos = gymapi.Vec3(1.2, 1.2, 1.2)
# cam_target = gymapi.Vec3(0.0, 0.0, 0.6)
# gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# # 主循环（只显示）
# while not gym.query_viewer_has_closed(viewer):
#     gym.simulate(sim)
#     gym.fetch_results(sim, True)
#     gym.step_graphics(sim)

#     # 每帧：在 tip 质心位置画十字
#     rb_states = gym.get_actor_rigid_body_states(env, hand_actor, gymapi.STATE_POS)
#     L = 0.02
#     for h in TIP_IDX:
#         p = rb_states['pose']['p'][h]  # h 已做越界检查
#         lines = [
#             p[0]-L, p[1],   p[2],  p[0]+L, p[1],   p[2],
#             p[0],   p[1]-L, p[2],  p[0],   p[1]+L, p[2],
#             p[0],   p[1],   p[2]-L,p[0],   p[1],   p[2]+L,
#         ]
#         gym.add_lines(viewer, env, 3, lines, [0.95, 0.1, 0.1])

#     gym.draw_viewer(viewer, sim, True)
#     gym.sync_frame_time(sim)

# gym.destroy_viewer(viewer)
# gym.destroy_sim(sim)
