# G1 轮滑策略 Sim-to-Real 部署流水线

## 目录

1. [系统架构概览](#1-系统架构概览)
2. [策略概要](#2-策略概要)
3. [观测空间：仿真与真实映射](#3-观测空间仿真与真实映射)
4. [动作空间与执行器](#4-动作空间与执行器)
5. [观测归一化](#5-观测归一化)
6. [策略导出](#6-策略导出)
7. [硬件前置条件](#7-硬件前置条件)
8. [部署软件栈](#8-部署软件栈)
9. [推理循环设计](#9-推理循环设计)
10. [安全系统](#10-安全系统)
11. [系统辨识清单](#11-系统辨识清单)
12. [验证阶段](#12-验证阶段)
13. [已知风险与缓解措施](#13-已知风险与缓解措施)
14. [文件结构](#14-文件结构)

---

## 1. 系统架构概览

```
┌──────────────────────────────────────────────────────────────┐
│                    宿主计算机 (GPU)                           │
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │  手柄 /      │───▶│  观测构建器   │───▶│  ONNX/PyTorch  │  │
│  │  摇杆        │    │              │    │  策略 (50Hz)   │  │
│  └─────────────┘    └──────────────┘    └───────┬────────┘  │
│                            ▲                     │           │
│                            │                     ▼           │
│                     ┌──────┴───────┐    ┌────────────────┐  │
│                     │  状态接收器   │    │  动作后处理器   │  │
│                     │              │    │                │  │
│                     └──────┬───────┘    └───────┬────────┘  │
│                            │                     │           │
└────────────────────────────┼─────────────────────┼───────────┘
                             │  以太网              │
                             │  192.168.123.x      │
                       ┌─────┴─────────────────────┴─────┐
                       │        UNITREE G1 (EDU)         │
                       │                                  │
                       │  ┌────────────────────────────┐ │
                       │  │  Unitree SDK2 底层接口      │ │
                       │  │  • 电机状态读取              │ │
                       │  │  • PD 位置指令               │ │
                       │  │  • IMU 数据读取              │ │
                       │  └────────────────────────────┘ │
                       │  ┌────────────────────────────┐ │
                       │  │  物理机器人                  │ │
                       │  │  23 个驱动自由度 + 轮滑鞋    │ │
                       │  └────────────────────────────┘ │
                       └──────────────────────────────────┘
```

策略以 **50 Hz** 运行（与仿真控制频率一致：`timestep=0.005s × decimation=4 = 0.02s`）。
通信使用 Unitree SDK2 Python，基于 CycloneDDS 协议通过以太网进行。

---

## 2. 策略概要

| 属性 | 值 |
|---|---|
| 算法 | PPO（或 AMP-PPO 变体） |
| 网络结构 | MLP `512 → 256 → 128`（ELU 激活函数） |
| 观测归一化 | EmpiricalNormalization（滑动均值/方差） |
| Actor 输入维度 | **920**（每帧 92 维 × 10 帧历史） |
| Actor 输出维度 | **23**（关节位置偏移量） |
| 控制类型 | 通过 PD 控制器的关节位置目标 |
| 控制频率 | **50 Hz** |

---

## 3. 观测空间：仿真与真实映射

策略观测为每帧 **92 维** 向量，包含 **10 帧** 历史记录（最旧的在前），
展平为 **920 维** 输入。

### 每帧观测分解（92 维）

| 序号 | 项目 | 维度 | 仿真来源 | 真实来源 | 缩放系数 | 训练噪声 | 备注 |
|---|---|---|---|---|---|---|---|
| 1 | `command` | **3** | `[vx_cmd, vy_cmd, wz_cmd]` | 手柄 / 摇杆 | 1.0 | 无 | 机体坐标系速度指令 |
| 2 | `base_lin_vel` | **3** | `root_link_lin_vel_b` | **状态估计器**（见 §3.1） | 1.0 | U(-0.1, 0.1) | 机体坐标系线速度 |
| 3 | `base_ang_vel` | **3** | `root_link_ang_vel_b` | IMU 陀螺仪 | **0.25** | U(-0.2, 0.2) | 机体坐标系角速度 |
| 4 | `projected_gravity` | **3** | `projected_gravity` | IMU 姿态 → 旋转 [0,0,-1] | 1.0 | U(-0.05, 0.05) | 机体坐标系重力方向 |
| 5 | `joint_pos` | **23** | `joint_pos - standing_ref` | 电机编码器 − 站立参考值 | 1.0 | U(-0.01, 0.01) | 相对于站立姿态 |
| 6 | `joint_vel` | **23** | `joint_vel` | 电机编码器（差分或硬件速度） | **0.05** | U(-1.5, 1.5) | 缩放 ×0.05 |
| 7 | `actions` | **23** | `last_action` | 上一次动作缓存 | 1.0 | 无 | 上一次策略输出 |
| 8 | `wheel_contact` | **8** | 接触传感器滤波 | **估计值**（见 §3.2） | 1.0 | 无 | 每轮二值接触标志 |
| 9 | `skate_separation` | **3** | 标记点位置差 | **正运动学计算**（见 §3.3） | 1.0 | 无 | 世界坐标系 左→右溜冰鞋向量 |
| | **总计** | **92** | | | | | |

### 3.1 基座线速度（关键难点）

**仿真：** 可直接获取机体坐标系的真实线速度。

**真实：** G1 的板载 IMU 提供角速度和线性加速度，**而非**线速度。
需要状态估计器。按可靠性排序的方案：

1. **IMU + 腿部运动学估计器（推荐）**
   - 如果 EDU 固件提供了 Unitree SDK 内置的状态估计器，优先使用。
   - 或者实现互补滤波器或 EKF，融合：
     - IMU 加速度计（积分并修正漂移）
     - 腿部正运动学（轮子接触点提供约束）
   - 这是 ETH RSL（ANYmal）、Agility Robotics 和 SKATER 论文部署中使用的标准方法。

2. **外部动作捕捉（仅用于开发/验证）**
   - OptiTrack / Vicon → 通过位置有限差分获取机体坐标系速度。
   - 适用于初期验证，不适合最终部署。

3. **使用噪声/延迟速度训练（鲁棒性后备方案）**
   - 训练中已对该通道施加了 U(-0.1, 0.1) 噪声。
   - 如果真实估计器噪声较大，考虑在训练中增大噪声范围。

### 3.2 轮子接触标志

**仿真：** 接触传感器检测每个轮子的力 > 1N，配合 2 步滤波器（当前步与前一步取 OR）。

**真实方案：**
1. **电机电流阈值** — 轮子关节为被动关节（无电机），不直接适用。替代方案：
2. **溜冰鞋框架下方力/压力传感器** — 如果可用，对每轮垂直力设阈值。
3. **腿部运动学 + 重力** — 通过关节力矩的逆动力学估算地面反力。设阈值判定接触。
4. **保守默认值** — 将全部 8 个接触设为 `1.0`（所有轮子着地）。策略训练时惩罚了轮子离地，因此策略预期大多数轮子在大多数时间接触地面。这是合理的初始近似。

**建议：** 初次部署从方案 4 开始（所有接触 = 1.0）。策略对此具有鲁棒性，因为训练中要求轮子保持着地。后续通过力估计进行改进。

### 3.3 溜冰鞋间距向量

**仿真：** 通过 MuJoCo 中每只溜冰鞋的前/后标记点位置计算。

**真实：** 通过编码器读数的正运动学计算。运动链为：
```
骨盆 → 髋关节俯仰 → 髋关节侧摆 → 髋关节偏转 → 膝关节 → 踝关节俯仰 → 踝关节侧摆 → 溜冰鞋
```
使用 MJCF 模型的刚体变换和测量的关节位置计算
`left_skate_front_marker`、`left_skate_rear_marker`、`right_skate_front_marker`、
`right_skate_rear_marker` 的正运动学。间距向量为：
```
skate_separation = mean(右侧标记点) - mean(左侧标记点)
```
该向量在世界坐标系中表示。使用 IMU 姿态进行机体到世界坐标系的旋转。

---

## 4. 动作空间与执行器

### 4.1 动作语义

策略输出 **23 个连续值**，大致在 [-1, 1] 范围内。转换为关节位置目标的公式如下：

```
target_position[i] = default_pos[i] + action[i] × action_scale[i] × beta
```

其中：
- `default_pos`：站立参考姿态（来自 `STANDING_SKATE_CONTROLLED_JOINT_POS`）
- `action_scale`：每关节缩放因子，由 `0.25 × effort_limit / stiffness` 推导
- `beta`：课程学习参数，在最终训练阶段为 **1.0**（完全部署）

### 4.2 关节顺序（仿真与真实必须完全一致）

```
索引  关节名称                        执行器类型       力矩限制 (Nm)
─────  ────────────────────────────  ──────────────   ───────────
 0     left_hip_pitch_joint          7520_14          88.0
 1     left_hip_roll_joint           7520_22          139.0
 2     left_hip_yaw_joint            7520_14          88.0
 3     left_knee_joint               7520_22          139.0
 4     left_ankle_pitch_joint        ANKLE(2×5020)    50.0
 5     left_ankle_roll_joint         ANKLE(2×5020)    50.0
 6     right_hip_pitch_joint         7520_14          88.0
 7     right_hip_roll_joint          7520_22          139.0
 8     right_hip_yaw_joint           7520_14          88.0
 9     right_knee_joint              7520_22          139.0
10     right_ankle_pitch_joint       ANKLE(2×5020)    50.0
11     right_ankle_roll_joint        ANKLE(2×5020)    50.0
12     waist_yaw_joint               7520_14          88.0
13     waist_roll_joint              WAIST(2×5020)    50.0
14     waist_pitch_joint             WAIST(2×5020)    50.0
15     left_shoulder_pitch_joint     5020             25.0
16     left_shoulder_roll_joint      5020             25.0
17     left_shoulder_yaw_joint       5020             25.0
18     left_elbow_joint              5020             25.0
19     right_shoulder_pitch_joint    5020             25.0
20     right_shoulder_roll_joint     5020             25.0
21     right_shoulder_yaw_joint      5020             25.0
22     right_elbow_joint             5020             25.0
```

### 4.3 站立参考姿态（弧度）

```python
STANDING_SKATE_CONTROLLED_JOINT_POS = (
  -0.05, 0.0, 0.0, 0.45, -0.25, 0.0,       # 左腿
  -0.05, 0.0, 0.0, 0.45, -0.25, 0.0,       # 右腿
   0.0, 0.0, 0.05,                           # 腰部
  -0.10, 0.45, -0.20, 1.10,                 # 左臂
  -0.10, -0.45, 0.20, 1.10,                 # 右臂
)
```

### 4.4 真实机器人的 PD 增益

仿真使用位置模式执行器，其刚度/阻尼由电机物理参数推导。
真实机器人**必须使用完全相同的 PD 增益**：

```python
# 各执行器类型：
# natural_freq = 10 × 2π ≈ 62.83 rad/s
# damping_ratio = 2.0

# 5020 类（手臂）：   Kp ≈ ARMATURE_5020 × ω²    Kd ≈ 2 × 2.0 × ARMATURE_5020 × ω
# 7520_14 类（髋部）：Kp ≈ ARMATURE_7520_14 × ω²  ...
# 7520_22 类（膝部）：Kp ≈ ARMATURE_7520_22 × ω²  ...
# ANKLE 类：          Kp = 2 × STIFFNESS_5020      Kd = 2 × DAMPING_5020
# WAIST 类：          Kp = 2 × STIFFNESS_5020      Kd = 2 × DAMPING_5020
```

这些值作为 ONNX 元数据（`joint_stiffness`、`joint_damping`）导出，部署运行时必须加载。
通过 Unitree SDK 的底层电机接口发送位置目标时，使用这些 Kp/Kd 值。

### 4.5 关键：关节到 Unitree 电机 ID 的映射

Unitree G1 SDK 使用电机 ID（整机 0-28）。**必须**建立从 23 个策略关节名
到正确 Unitree 电机 ID 的经过验证的映射。此映射取决于具体的 G1 EDU 固件版本和配置。

**验证流程：**
1. 逐一对每个关节施加站立姿态的小偏移（+0.05 rad）。
2. 目视确认正确的物理关节沿预期方向运动。
3. 验证符号约定匹配（仿真中的正方向 = 真实的正方向）。

---

## 5. 观测归一化

策略使用 `EmpiricalNormalization`，通过运行统计量对观测进行归一化：

```
obs_normalized = (obs_raw - mean) / (std + eps)
```

其中 `eps = 0.01`，`mean`/`std` 在训练期间累积。

### 导出要求

归一化器状态保存在 PyTorch checkpoint 内部：
```
model_state_dict.actor_obs_normalizer._mean   # 形状：[1, 920]
model_state_dict.actor_obs_normalizer._var    # 形状：[1, 920]
model_state_dict.actor_obs_normalizer._std    # 形状：[1, 920]
model_state_dict.actor_obs_normalizer.count   # 标量
```

**ONNX 导出：** 现有的 `_OnnxPolicyExporter` 将归一化嵌入 ONNX 计算图，因此导出的
ONNX 模型接受原始观测并在内部处理归一化。

**PyTorch 部署：** 加载完整 checkpoint 并使用 `policy.act_inference(obs_dict)`，
该方法内部调用 `self.actor_obs_normalizer(obs)`。

**关键：** 部署期间归一化器必须处于 `eval` 模式（`.eval()`），以防止用真实世界数据
更新统计量，避免分布偏移。

---

## 6. 策略导出

### 6.1 PyTorch Checkpoint（推荐用于初期部署）

```python
import torch

checkpoint = torch.load("model_50000.pt", map_location="cpu")
# 键值：model_state_dict, optimizer_state_dict, iter, infos

# 加载到 ActorCritic
from rsl_rl.modules import ActorCritic
policy = ActorCritic(...)
policy.load_state_dict(checkpoint["model_state_dict"])
policy.eval()

# 推理
with torch.no_grad():
    action = policy.act_inference(obs_dict)
```

### 6.2 ONNX 导出（推荐用于生产环境）

代码库已有 ONNX 导出基础设施。扩展 skater 任务的 runner，调用
`export_roller_policy_as_onnx`（参考 roller 任务的 `RollerOnPolicyRunner`）。

```python
# 导出脚本（已创建）
from mjlab_roller.tasks.roller.rl.exporter import export_roller_policy_as_onnx, attach_onnx_metadata

normalizer = policy.actor_obs_normalizer if policy.actor_obs_normalization else None
export_roller_policy_as_onnx(policy, normalizer=normalizer, path="./export/", filename="skater_policy.onnx")
attach_onnx_metadata(env, run_path="checkpoint_name", path="./export/", filename="skater_policy.onnx")
```

ONNX 元数据包含：`joint_names`、`joint_stiffness`、`joint_damping`、`default_joint_pos`、
`action_scale`、`command_axes`、`joystick_mapping`、`action_beta_max`。

### 6.3 TorchScript（替代方案）

```python
traced = torch.jit.trace(policy.actor, torch.randn(1, 920))
traced.save("skater_policy.pt")
# 需要额外保存归一化器的 mean/std
```

---

## 7. 硬件前置条件

### 7.1 Unitree G1 EDU

- **固件：** v1.3.0 或更高版本（支持 SDK2 底层电机控制）
- **模式：** 调试/开发模式（遥控器 L2+B，然后 L2+R2）
- **网络：** 以太网连接，IP 192.168.123.x 子网
- **SDK：** `unitree_sdk2_python`，需要 CycloneDDS 支持 `unitree_hg` IDL

### 7.2 定制直排轮滑鞋

仿真模型使用定制直排轮滑附件：
- 每只脚 4 个被动轮子，安装在踝关节下方
- 轮子直径和间距参照 `g1.xml` 几何定义
- 物理溜冰鞋必须尽可能匹配仿真模型的几何尺寸

**需要匹配的关键尺寸：**
- 轮距（轴距长度）
- 轮子直径
- 溜冰鞋安装点相对于踝关节侧摆关节的位置
- 溜冰鞋框架高度（影响站立高度）

### 7.3 宿主计算机

- 支持 CUDA 的 GPU（用于 PyTorch 推理，但 50Hz 的 3 层 MLP 用 CPU 即可）
- Ubuntu 20.04+ 和 Python 3.10+
- 以太网接口配置为 192.168.123.x

### 7.4 手柄 / 摇杆

- 任何兼容 SDL2 的手柄（Xbox、PS4/5 等）
- 通过 pygame 映射到 `[vx, vy, wz]` 指令
- 训练配置使用：`左摇杆Y → vx`、`左摇杆X → vy`、`右摇杆X → wz`

### 7.5 可选：安全吊带 / 龙门架

强烈建议在初期测试中使用。机器人应悬挂在吊带中并留有余量，
使其可以自由滑行但在跌倒时被接住。

---

## 8. 部署软件栈

### 8.1 通信层

```python
# G1 的 CycloneDDS 配置
import os
os.environ["CYCLONEDDS_URI"] = (
    '<CycloneDDS><Domain><General>'
    '<NetworkInterfaceAddress>192.168.123.123</NetworkInterfaceAddress>'
    '</General></Domain></CycloneDDS>'
)

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg import LowCmd_, LowState_
```

### 8.2 状态读取器

以 SDK 原生速率（通常 500Hz 或 1000Hz）从 G1 读取，维护最新状态：

```python
class G1StateReader:
    """订阅 G1 LowState 并提供最新读数。"""

    def __init__(self):
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.latest_state = None
        self.lock = threading.Lock()

    def callback(self, msg):
        with self.lock:
            self.latest_state = msg

    def get_joint_pos(self, joint_ids: list[int]) -> np.ndarray:
        """获取指定电机 ID 的当前关节位置。"""
        with self.lock:
            return np.array([self.latest_state.motor_state[i].q for i in joint_ids])

    def get_joint_vel(self, joint_ids: list[int]) -> np.ndarray:
        with self.lock:
            return np.array([self.latest_state.motor_state[i].dq for i in joint_ids])

    def get_imu(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (四元数, 角速度)。"""
        with self.lock:
            quat = np.array(self.latest_state.imu_state.quaternion)  # [w,x,y,z]
            gyro = np.array(self.latest_state.imu_state.gyroscope)   # [wx,wy,wz]
            return quat, gyro
```

### 8.3 电机指令器

以 50Hz 发送位置目标：

```python
class G1MotorCommander:
    """向 G1 电机发布关节位置目标。"""

    def __init__(self, motor_ids: list[int], kp: np.ndarray, kd: np.ndarray):
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.motor_ids = motor_ids
        self.kp = kp
        self.kd = kd

    def send_targets(self, target_positions: np.ndarray):
        cmd = LowCmd_()
        for i, mid in enumerate(self.motor_ids):
            cmd.motor_cmd[mid].mode = 0x01  # 伺服模式
            cmd.motor_cmd[mid].q = float(target_positions[i])
            cmd.motor_cmd[mid].dq = 0.0
            cmd.motor_cmd[mid].tau = 0.0
            cmd.motor_cmd[mid].kp = float(self.kp[i])
            cmd.motor_cmd[mid].kd = float(self.kd[i])
        self.publisher.write(cmd)
```

---

## 9. 推理循环设计

### 9.1 主循环（50 Hz）

```
┌─────────────────────────────────────────────┐
│              50 Hz 控制循环                   │
│                                              │
│  1. 读取状态（IMU + 编码器）                  │
│  2. 估计基座速度                              │
│  3. 计算正运动学得到溜冰鞋间距                 │
│  4. 构建观测向量（92 维）                     │
│  5. 推入历史缓冲区（左移）                    │
│  6. 展平历史 → 920 维                        │
│  7. 运行策略推理（GPU < 1ms）                 │
│  8. 从动作计算关节目标                        │
│  9. 发送目标到电机指令器                      │
│ 10. 将动作存入缓冲区供下次观测使用             │
│                                              │
│  每步总预算：20ms                             │
│  典型推理时间：0.1-0.5ms（GPU 上的 MLP）      │
│  剩余：传感器 I/O + 正运动学 + 安全检查       │
└─────────────────────────────────────────────┘
```

### 9.2 观测构建（伪代码）

```python
def build_observation(state_reader, velocity_estimator, fk_solver,
                      command, last_action, wheel_contacts,
                      standing_ref, ang_vel_scale=0.25, joint_vel_scale=0.05):
    # 1. 指令 [3]
    cmd = np.array([command.vx, command.vy, command.wz])

    # 2. 机体坐标系线速度 [3]
    base_lin_vel_b = velocity_estimator.get_velocity()

    # 3. 机体坐标系角速度，缩放 [3]
    _, gyro = state_reader.get_imu()
    base_ang_vel_b = gyro * ang_vel_scale

    # 4. 投影重力 [3]
    quat, _ = state_reader.get_imu()
    gravity_world = np.array([0.0, 0.0, -1.0])
    proj_gravity = quat_apply_inverse(quat, gravity_world)

    # 5. 相对于站立参考的关节位置 [23]
    joint_pos = state_reader.get_joint_pos(POLICY_MOTOR_IDS)
    joint_pos_rel = joint_pos - standing_ref

    # 6. 缩放后的关节速度 [23]
    joint_vel = state_reader.get_joint_vel(POLICY_MOTOR_IDS)
    joint_vel_scaled = joint_vel * joint_vel_scale

    # 7. 上一次动作 [23]
    prev_actions = last_action

    # 8. 轮子接触标志 [8]
    contacts = wheel_contacts  # 初始为 np.ones(8)

    # 9. 溜冰鞋间距 [3]
    skate_sep = fk_solver.compute_skate_separation(joint_pos, quat)

    obs_frame = np.concatenate([
        cmd,              # 3
        base_lin_vel_b,   # 3
        base_ang_vel_b,   # 3
        proj_gravity,     # 3
        joint_pos_rel,    # 23
        joint_vel_scaled, # 23
        prev_actions,     # 23
        contacts,         # 8
        skate_sep,        # 3
    ])  # 总计：92

    return obs_frame
```

### 9.3 历史缓冲区管理

```python
class ObservationHistory:
    def __init__(self, frame_dim=92, history_length=10):
        self.buffer = np.zeros((history_length, frame_dim))
        self.history_length = history_length

    def push(self, obs_frame: np.ndarray):
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = obs_frame

    def get_flat(self) -> np.ndarray:
        return self.buffer.flatten()  # 形状：(920,)

    def reset(self, initial_frame: np.ndarray):
        self.buffer[:] = initial_frame[np.newaxis, :]
```

### 9.4 动作后处理

```python
def action_to_joint_targets(action: np.ndarray,
                            default_pos: np.ndarray,
                            action_scale: np.ndarray,
                            beta: float = 1.0) -> np.ndarray:
    """将策略输出转换为关节位置目标。"""
    return default_pos + action * action_scale * beta
```

### 9.5 完整推理循环

```python
def run_deployment(policy, state_reader, commander, joystick,
                   velocity_estimator, fk_solver, config):
    history = ObservationHistory(frame_dim=92, history_length=10)
    last_action = np.zeros(23)
    wheel_contacts = np.ones(8)
    rate = Rate(config.control_freq)  # 50 Hz

    # 用站立观测初始化历史
    initial_obs = build_observation(
        state_reader, velocity_estimator, fk_solver,
        Command(0, 0, 0), last_action, wheel_contacts,
        config.standing_ref
    )
    history.reset(initial_obs)

    while running:
        # 读取指令
        command = joystick.read()

        # 构建当前观测帧
        obs_frame = build_observation(
            state_reader, velocity_estimator, fk_solver,
            command, last_action, wheel_contacts,
            config.standing_ref
        )
        history.push(obs_frame)

        # 展平并运行策略
        obs_flat = torch.from_numpy(history.get_flat()).float().unsqueeze(0)
        with torch.no_grad():
            action = policy.act_inference({"policy": obs_flat})
        action_np = action.squeeze(0).cpu().numpy()

        # 转换为关节目标
        targets = action_to_joint_targets(
            action_np, config.default_pos,
            config.action_scale, beta=1.0
        )

        # 安全检查
        targets = apply_joint_limits(targets, config.joint_limits)
        targets = apply_rate_limit(targets, last_targets, config.max_delta)

        # 发送到机器人
        commander.send_targets(targets)

        # 存储供下次观测使用
        last_action = action_np

        rate.sleep()
```

---

## 10. 安全系统

### 10.1 关节限位钳制

将所有目标位置钳制到 MJCF 关节限位（训练中使用 10% 软限位裕度）：

```python
def apply_joint_limits(targets, limits, soft_factor=0.9):
    lower = limits[:, 0] * soft_factor
    upper = limits[:, 1] * soft_factor
    return np.clip(targets, lower, upper)
```

### 10.2 速率限制

防止关节目标突然大幅跳变：

```python
def apply_rate_limit(targets, prev_targets, max_delta_per_step):
    delta = targets - prev_targets
    delta = np.clip(delta, -max_delta_per_step, max_delta_per_step)
    return prev_targets + delta
```

### 10.3 姿态紧急停止

如果机器人倾斜超过训练终止角度（70°），切换到安全恢复模式
（松弛或保持最后已知安全位置）：

```python
def check_orientation_safe(projected_gravity, limit_rad=1.22):  # 70°
    cos_angle = -projected_gravity[2]  # 与 [0,0,-1] 的点积
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    return angle < limit_rad
```

### 10.4 急停

- 硬件急停按钮（Unitree 遥控器 L1+L2+A）
- 软件终止：Ctrl+C 设置 `running = False` 并指令站立姿态
- 站立恢复：以完整 PD 增益保持站立参考姿态 5 秒后关机

### 10.5 启动序列

1. 以阻尼/调试模式开机 G1
2. **系好安全吊带**（用于初期测试）
3. 在 3 秒内从当前位置缓慢移动到站立姿态（线性插值）
4. 等待 2 秒稳定
5. 以零速度指令开始策略循环，持续 5 秒
6. 启用手柄指令

---

## 11. 系统辨识清单

部署前需验证以下参数在仿真和真实之间匹配：

| 参数 | 仿真值 | 如何验证/标定 |
|---|---|---|
| 关节零位 | MJCF qpos0 | 向所有关节发送零位指令，用量角器/编码器测量 |
| 关节方向符号 | MJCF joint axis | 指令 +0.1 rad，验证物理方向匹配 |
| PD 增益（Kp, Kd） | 见 §4.4 | 从仿真值开始，如有振荡/迟滞则在真机上调参 |
| 关节限位 | MJCF range | 从 G1 固件读取，进行比较 |
| 连杆质量 | MJCF body mass | 如可拆卸则称量机器人肢体；域随机化覆盖 ±10% |
| IMU 姿态 | MJCF imu_in_pelvis 站点 | 验证四元数约定（wxyz vs xyzw） |
| 控制频率 | 50 Hz | 测量实际循环时序，确保抖动 < 2ms |
| 溜冰鞋尺寸 | MJCF 几何体 | 测量物理溜冰鞋，必要时更新 XML |
| 轮子摩擦力 | 域随机化范围 [0.1, 0.8] | 在目标地面上测试；应落在训练范围内 |

---

## 12. 验证阶段

### 阶段 1：桌面检查（无硬件）
- [ ] 导出 ONNX 模型，验证输入/输出维度
- [ ] 在仿真中无头运行推理，与训练回放比较动作
- [ ] 验证关节映射表与 G1 SDK 电机 ID 对应
- [ ] 验证归一化统计量正确加载

### 阶段 2：仿真回环（软件验证）
- [ ] 对 MuJoCo 仿真运行部署代码（用仿真 API 替代 SDK）
- [ ] 验证观测构建与训练观测完全一致
- [ ] 验证动作后处理产生相同的关节目标
- [ ] 运行 10,000 步，与训练回放比较策略行为

### 阶段 3：静态硬件测试
- [ ] 在调试模式下连接 G1
- [ ] 指令站立姿态，验证所有关节到达正确位置
- [ ] 读取 IMU 数据，验证四元数约定
- [ ] 逐一测试每个关节：+0.05 rad 偏移，验证正确关节和方向
- [ ] 手动移动关节时读取关节速度，验证符号约定

### 阶段 4：悬挂测试（安全吊带）
- [ ] 将机器人悬挂在吊带中，脚部刚好接触地面
- [ ] 以零速度指令运行策略
- [ ] 验证稳定站立（无振荡、无漂移）
- [ ] 施加小的 vx 指令（0.1 m/s），观察腿部运动
- [ ] 测试急停流程

### 阶段 5：地面测试（吊带 + 平坦地面）
- [ ] 放置在光滑平坦地面上，吊带留有余量
- [ ] 零速度指令：机器人应在溜冰鞋上稳定站立
- [ ] 逐步增大 vx：0.1 → 0.3 → 0.5 → 1.0 m/s
- [ ] 测试转弯：小的 wz 指令
- [ ] 测试 vx + wz 组合
- [ ] 监控并记录所有观测，用于后续分析

### 阶段 6：自由滑行（最终验证）
- [ ] 移除吊带（需有人在旁保护）
- [ ] 仅使用保守速度指令
- [ ] 全速度范围测试
- [ ] 长时间稳定性测试（> 60 秒持续滑行）

---

## 13. 已知风险与缓解措施

### 13.1 基座速度估计误差

**风险：** 策略训练时使用了真实速度。真实估计将存在噪声/偏差。

**缓解措施：**
- 训练中对 `base_lin_vel` 施加了 U(-0.1, 0.1) 噪声。如果真实估计器噪声更大，
  考虑增大到 U(-0.2, 0.2) 并重新训练。
- 使用卡尔曼滤波器或互补滤波器进行速度估计。
- 在阶段 4 中比较估计值与动捕测量速度进行验证。

### 13.2 仿真-真实执行器动力学差异

**风险：** 真实电机动力学可能与仿真位置模式执行器不同。

**缓解措施：**
- 训练中随机化了执行器 Kp（±10%）和 Kd（±10%）。
- 在真实硬件上调参 PD 增益。如果真实电机存在仿真未建模的显著间隙或摩擦，
  考虑在训练中增加额外的执行器建模。

### 13.3 轮子-地面摩擦力不匹配

**风险：** 真实滑行地面的摩擦力可能与训练范围不同。

**缓解措施：**
- 训练中随机化了静摩擦力 [0.1, 0.8] 和动摩擦力 [0.1, 0.4]。
- 在目标部署地面上测试。大多数室内光滑地面处于此范围内。
- 如需户外部署，需要扩展摩擦力范围并重新训练。

### 13.4 通信延迟

**风险：** DDS 通信延迟可能导致策略在过时状态上操作。

**缓解措施：**
- G1 SDK 通过以太网的延迟 < 1ms，远在 20ms 控制预算之内。
- 监控实际循环时序。如果抖动超过 5ms，检查操作系统调度（使用
  `SCHED_FIFO` 或实时内核）。
- 考虑训练时增加 1 步观测延迟以提升鲁棒性。

### 13.5 溜冰鞋几何不匹配

**风险：** 物理溜冰鞋尺寸与 MJCF 模型不匹配。

**缓解措施：**
- 在最终训练前精确测量物理溜冰鞋并更新 `g1.xml`。
- 关键参数：轮距、轮子半径、溜冰鞋高度、安装角度。
- 基于正运动学的溜冰鞋间距计算必须使用真实几何。

---

## 14. 文件结构

部署新增的文件：

```
scripts/deploy/
├── export_policy.py           # 导出 checkpoint → ONNX 及元数据
├── deploy_g1.py               # 主部署循环
├── g1_interface.py            # Unitree SDK 状态订阅和电机指令发布
├── observation_builder.py     # 从传感器构建 92 维观测
├── safety.py                  # 关节限位、速率限制、姿态检查
├── config.py                  # 关节映射、PD 增益、限位、参考姿态
└── verify_pipeline.py         # 验证维度匹配的自动化脚本
```

---

## 附录 A：观测维度验证

验证仿真和部署之间的观测维度匹配：

```python
# 在仿真中
env = ...  # 创建环境
obs = env.reset()
policy_obs = obs["policy"]
print(f"策略观测形状: {policy_obs.shape}")  # 预期：(num_envs, 920)
print(f"每帧维度: {policy_obs.shape[-1] // 10}")  # 预期：92
```

## 附录 B：快速参考 — 四元数约定

MJCF 模型和 Unitree SDK 均使用 **[w, x, y, z]** 四元数顺序。
验证你的 `quat_apply_inverse` 实现是否匹配。

```python
def quat_apply_inverse(quat_wxyz, vec):
    """用四元数 (w,x,y,z 格式) 的逆旋转 vec。"""
    w, x, y, z = quat_wxyz
    # 逆旋转：单位四元数的共轭
    # q_conj = [w, -x, -y, -z]
    # v_rotated = q_conj * v * q
    t = 2.0 * np.cross(np.array([-x, -y, -z]), vec)
    return vec + w * t + np.cross(np.array([-x, -y, -z]), t)
```

## 附录 C：与 Psi0 部署架构的对比

| 方面 | Psi0（VLA，运动操作） | 本项目（RL 运动控制） |
|---|---|---|
| 策略类型 | 扩散式 VLA（Qwen3-VL + 流匹配） | MLP actor-critic（PPO） |
| 推理延迟 | ~100-300ms（重型 GPU） | ~0.1-0.5ms（轻量级 MLP） |
| 架构 | 服务器-客户端（WebSocket/HTTP） | 单进程（直接 SDK） |
| 控制分层 | System-0 RL 追踪器 + VLA 上身 | 端到端关节位置控制 |
| 观测 | RGB 相机 + 32 维本体感受 | 920 维纯本体感受 |
| 动作空间 | 36 维（手 + 臂 + 躯干 + 运动） | 23 维（所有关节，位置目标） |
| 控制频率 | 30Hz VLA → 60Hz 追踪控制器 | 50Hz 直接控制 |

**关键区别：** Psi0 使用分层架构，其中基于 RL 的 "System-0" 追踪控制器处理底层运动，
而 VLA 控制上身和高级运动。我们的轮滑策略是单一的端到端控制器，直接输出全部 23 个关节
目标 —— 底层没有单独的运动控制器。

这意味着我们的部署更简单（不需要逆运动学求解器，不需要单独的追踪策略），但 RL 策略
必须直接处理平衡和运动的所有方面，因此需要更加仔细地弥合仿真到真实的差异。
