import os
import sys

# --------------------------------------------------
# 1. 启动 Isaac Sim
# 必须在导入 omni / isaacsim 其他模块之前执行
# --------------------------------------------------
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False
})

try:
    print("Isaac Sim started successfully.")

    # --------------------------------------------------
    # 2. 设置项目路径
    # --------------------------------------------------
    project_dir = os.path.dirname(os.path.abspath(__file__))
    usd_path = os.path.join(project_dir, "pick_place_project.usd")
    py_path = os.path.join(project_dir, "pick_place_done.py")

    if not os.path.isfile(usd_path):
        raise FileNotFoundError(f"USD file not found: {usd_path}")
    if not os.path.isfile(py_path):
        raise FileNotFoundError(f"Python file not found: {py_path}")

    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    # --------------------------------------------------
    # 3. 打开 USD 场景
    # --------------------------------------------------
    import omni.usd
    usd_context = omni.usd.get_context()
    
    print(f"Opening USD stage: {usd_path}")
    if not usd_context.open_stage(usd_path):
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")
    print("USD stage opened successfully.")

    # --------------------------------------------------
    # 3.5 彻底清理坏掉的地面，并强制重建 (解决贴图丢失问题)
    # --------------------------------------------------
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.storage.native import get_assets_root_path

    stage = usd_context.get_stage()
    
    # 强行删掉之前那个贴图丢失的坏地面
    old_ground = stage.GetPrimAtPath("/World/ground")
    if old_ground and old_ground.IsValid():
        stage.RemovePrim("/World/ground")
        print("🗑️ Removed broken ground.")
        
    # 从官方资产库重新加载一个完美无缺的新地面
    assets_root = get_assets_root_path()
    stage_utils.add_reference_to_stage(
        usd_path=assets_root + "/Isaac/Environments/Grid/default_environment.usd",
        path="/World/ground"
    )
    print("✅ Recreated a new solid ground.")

    # --------------------------------------------------
    # 4. 等待场景初始化（物理引擎深度预热 60 帧）
    # --------------------------------------------------
    print("Warming up physics engine (200 frames)...")
    for _ in range(60):
        simulation_app.update()

    print("Scene initialization completed.")

    # --------------------------------------------------
    # 5. 导入机器人搬运控制脚本
    # --------------------------------------------------
    print("Loading pick_place_done.py...")
    import pick_place_done
    print("Robot trajectory controller loaded.")

    # --------------------------------------------------
    # 6. 持续运行 Isaac Sim
    # --------------------------------------------------
    print("Simulation is running.")
    while simulation_app.is_running():
        simulation_app.update()

except KeyboardInterrupt:
    print("Simulation interrupted by user.")

except Exception as e:
    print(f"Program error: {e}")
    raise

finally:
    simulation_app.close()
    print("Isaac Sim closed.")
