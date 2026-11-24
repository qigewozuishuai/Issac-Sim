
import numpy as np
import omni.usd
import omni.timeline

from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
from isaacsim.core.experimental.prims import Articulation, XformPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents


# ==========================================
# 清理之前运行留下的回调
# ==========================================
try:
    SimulationManager.deregister_all_callbacks()
    print("Previous callbacks cleared.")
except Exception:
    pass


class KukaTrajectoryTest:

    def __init__(self):
        self.step_counter = 0
        self.phase = 0
        self.is_finished = False
        self.wait_timer = 0

        # ==========================================
        # 核心路径
        # ==========================================
        self.robot = Articulation("/World/kr210_l150")

        self.tool0_path = "/World/gripper_pick"
        self.workpiece_path = "/World/part_pick"
        self.grasp_joint_path = "/World/part_pick/AttachJoint"

        stage = omni.usd.get_context().get_stage()

        self.part_prim = stage.GetPrimAtPath(self.workpiece_path)
        self.tool_prim = stage.GetPrimAtPath(self.tool0_path)

        self.part_xform = XformPrim(
            self.workpiece_path,
            reset_xform_op_properties=True
        )

        self.tool_xform = XformPrim(
            self.tool0_path,
            reset_xform_op_properties=True
        )

        # ==========================================
        # 检查 Prim
        # ==========================================
        if not self.part_prim or not self.part_prim.IsValid():
            raise RuntimeError(
                f"Workpiece Prim not found: {self.workpiece_path}"
            )

        if not self.tool_prim or not self.tool_prim.IsValid():
            raise RuntimeError(
                f"Tool Prim not found: {self.tool0_path}"
            )

        # ==========================================
        # 检查刚体 API
        # ==========================================
        if not self.part_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(self.part_prim)

        if not self.tool_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(
                "Tool Prim does not have RigidBodyAPI. "
                "Please set tool0_path to the actual rigid body Prim."
            )

        # ==========================================
        # 初始化时锁定工件
        # ==========================================
        part_rigid_api = UsdPhysics.RigidBodyAPI(self.part_prim)
        part_rigid_api.CreateKinematicEnabledAttr().Set(True)

        print("Workpiece locked: Kinematic=True.")

        # ==========================================
        # 轨迹点位配置
        # ==========================================
        self.pose_phase0_new = np.array([
            [-0.6279, -0.3327, -0.0321, -3.2029, -1.3556, 3.7020]
        ])

        self.pose_phase1_new = np.array([
            [-1.4730, -0.5172, 0.5041, -3.1841, -1.4020, 3.6014]
        ])

        self.pose_phase2_new = np.array([
            [-1.8491, -0.6370, 0.4625, -3.1972, -1.4851, 3.6788]
        ])

        self.pose_phase3_insert = np.array([
            [-2.0989, 0.2867, -0.4887, -3.2022, -1.4391, 3.7079]
        ])

        self.pose_2 = np.array([
            [-2.1315, 0.5783, -0.4359, -3.2018, -1.3588, 3.6997]
        ])

        self.pose_3 = np.array([
            [-2.1315, 0.6584, -0.3177, -3.2056, -1.1612, 3.7125]
        ])

        self.pose_4 = np.array([
            [-2.1315, 0.5783, -0.4359, -3.2018, -1.3588, 3.6997]
        ])

        self.pose_5 = np.array([
            [-2.1315, 0.5783, -0.4359, -3.2018, -1.3588, 3.6997]
        ])

        self.pose_phase6_new = np.array([
            [-0.6855, 0.4338, -0.3020, -3.2004, -1.3405, 3.6840]
        ])

        self.pose_phase7_new = np.array([
            [-0.6090, 0.8387, -0.5891, -3.2488, -1.2488, 3.7814]
        ])

        self.pose_phase8_new = np.array([
            [-0.6283, -0.3752, -0.0175, -3.2018, -1.3588, 3.6997]
        ])

        # ==========================================
        # 轨迹列表
        # ==========================================
        self.waypoints = [
            self.pose_phase0_new,    # Phase 0
            self.pose_phase1_new,    # Phase 1
            self.pose_phase2_new,    # Phase 2
            self.pose_phase3_insert, # Phase 3
            self.pose_2,             # Phase 4
            self.pose_3,             # Phase 5：抓取点
            self.pose_4,             # Phase 6
            self.pose_5,             # Phase 7
            self.pose_phase6_new,    # Phase 8
            self.pose_phase7_new,    # Phase 9：释放点
            self.pose_phase8_new     # Phase 10
        ]

    # ==========================================
    # 判断轨迹是否完成
    # ==========================================
    def is_done(self):
        return self.is_finished

    # ==========================================
    # 创建 FixedJoint
    # ==========================================
    def attach_workpiece(self, stage):

        if not self.tool_prim.IsValid():
            raise RuntimeError("Tool Prim is invalid.")

        if not self.part_prim.IsValid():
            raise RuntimeError("Workpiece Prim is invalid.")

        # 删除旧关节
        old_joint = stage.GetPrimAtPath(self.grasp_joint_path)

        if old_joint and old_joint.IsValid():
            stage.RemovePrim(self.grasp_joint_path)

        # 获取两个刚体的世界变换
        cache = UsdGeom.XformCache()

        tool_world = cache.GetLocalToWorldTransform(
            self.tool_prim
        )

        part_world = cache.GetLocalToWorldTransform(
            self.part_prim
        )

        # 以工件当前世界变换作为关节世界变换
        joint_world = part_world

        # 换算为两个刚体各自的局部连接变换
        tool_local_joint = (
            joint_world * tool_world.GetInverse()
        )

        part_local_joint = (
            joint_world * part_world.GetInverse()
        )

        tool_frame = Gf.Transform(tool_local_joint)
        part_frame = Gf.Transform(part_local_joint)

        tool_quat_d = tool_frame.GetRotation().GetQuat()
        part_quat_d = part_frame.GetRotation().GetQuat()

        tool_quat = Gf.Quatf(
            float(tool_quat_d.GetReal()),
            Gf.Vec3f(tool_quat_d.GetImaginary())
        )

        part_quat = Gf.Quatf(
            float(part_quat_d.GetReal()),
            Gf.Vec3f(part_quat_d.GetImaginary())
        )

        # 创建 FixedJoint
        grasp_joint = UsdPhysics.FixedJoint.Define(
            stage,
            self.grasp_joint_path
        )

        grasp_joint.CreateBody0Rel().SetTargets([
            self.tool0_path
        ])

        grasp_joint.CreateBody1Rel().SetTargets([
            self.workpiece_path
        ])

        # 抓手端局部连接坐标
        grasp_joint.CreateLocalPos0Attr().Set(
            Gf.Vec3f(tool_frame.GetTranslation())
        )

        grasp_joint.CreateLocalRot0Attr().Set(
            tool_quat
        )

        # 工件端局部连接坐标
        grasp_joint.CreateLocalPos1Attr().Set(
            Gf.Vec3f(part_frame.GetTranslation())
        )

        grasp_joint.CreateLocalRot1Attr().Set(
            part_quat
        )

        print("FixedJoint created.")
        print("Tool local joint position:", tool_frame.GetTranslation())
        print("Part local joint position:", part_frame.GetTranslation())

    # ==========================================
    # 主执行函数
    # ==========================================
    def forward(self):

        self.step_counter += 1
        stage = omni.usd.get_context().get_stage()

        # ==========================================
        # 关节运动控制
        # ==========================================
        def move_to(target_pose, base_speed=0.3):

            current_pos = self.robot.get_dof_positions()

            curr = (
                current_pos.numpy()[0]
                if hasattr(current_pos, "numpy")
                else current_pos[0]
            )

            target = target_pose[0]
            error = np.linalg.norm(target - curr)

            # 抓取点和释放点减速
            if self.phase == 5 or self.phase == 9:
                speed = base_speed * (error / 0.5)
                speed = max(speed, 0.01)
            else:
                speed = base_speed

            step = (target - curr) * speed

            self.robot.set_dof_position_targets([
                curr + step
            ])

            if error < 0.05:
                self.robot.set_dof_position_targets(target_pose)
                return True

            return False

        # ==========================================
        # 轨迹执行
        # ==========================================
        if self.phase < len(self.waypoints):

            target_pose = self.waypoints[self.phase]

            if move_to(target_pose):

                # ==========================================
                # Phase 5：抓取点
                # 到达后等待 2 秒，再附着工件
                # ==========================================
                if self.phase == 5:

                    if self.wait_timer == 0:
                        print(
                            "Phase 5: Reached pick pose. "
                            "Waiting 2 seconds..."
                        )

                    # 保持机器人在抓取点
                    self.robot.set_dof_position_targets(
                        self.pose_3
                    )

                    self.wait_timer += 1

                    # 120 帧约等于 2 秒（60 Hz）
                    if self.wait_timer < 120:
                        return

                    # 第 120 帧执行一次附着
                    if self.wait_timer == 120:

                        print(
                            "2 seconds elapsed. "
                            "Attaching workpiece..."
                        )

                        part_rigid_api = UsdPhysics.RigidBodyAPI(
                            self.part_prim
                        )

                        part_rigid_api.CreateKinematicEnabledAttr().Set(
                            False
                        )

                        # 清除工件原有速度
                        self.part_prim.GetAttribute(
                            "physics:velocity"
                        ).Set(Gf.Vec3f(0, 0, 0))

                        self.part_prim.GetAttribute(
                            "physics:angularVelocity"
                        ).Set(Gf.Vec3f(0, 0, 0))

                        self.attach_workpiece(stage)

                    # 附着后再等待 30 帧
                    if self.wait_timer < 150:
                        return

                # ==========================================
                # Phase 9：释放点
                # 释放后等待 2 秒
                # ==========================================
                elif self.phase == 9:

                    if self.wait_timer == 0:

                        print(
                            "Phase 9: Reached release pose. "
                            "Releasing workpiece..."
                        )

                        old_joint = stage.GetPrimAtPath(
                            self.grasp_joint_path
                        )

                        if old_joint and old_joint.IsValid():
                            stage.RemovePrim(
                                self.grasp_joint_path
                            )

                            print("AttachJoint removed.")

                        # 释放后保持工件当前世界位置
                        if self.part_prim.IsValid():

                            part_rigid_api = UsdPhysics.RigidBodyAPI(
                                self.part_prim
                            )

                            part_rigid_api.CreateKinematicEnabledAttr().Set(
                                True
                            )

                            print(
                                "Workpiece released and locked: "
                                "Kinematic=True."
                            )

                    self.wait_timer += 1

                    # 120 帧约等于 2 秒（60 Hz）
                    if self.wait_timer < 120:
                        return

                # ==========================================
                # 当前阶段完成，进入下一阶段
                # ==========================================
                print(
                    f"Phase {self.phase}: Reached waypoint."
                )

                self.phase += 1
                self.wait_timer = 0

        else:

            self.is_finished = True
            print("Trajectory Test Completed.")


# ==========================================
# 执行逻辑
# ==========================================
controller = KukaTrajectoryTest()


def physics_step_callback(dt, context):

    if not controller.is_done():

        controller.forward()

    else:

        SimulationManager.deregister_callback(
            global_callback_id
        )

        omni.timeline.get_timeline_interface().pause()

        print("=== Simulation paused ===")


# 注册物理步回调
global_callback_id = SimulationManager.register_callback(
    physics_step_callback,
    IsaacEvents.POST_PHYSICS_STEP
)

# 启动仿真
omni.timeline.get_timeline_interface().play()

print("Trajectory Test Simulation started...")

