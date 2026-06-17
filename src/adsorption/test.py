import geotorch
import torch
import torch.nn as nn
from torch.optim import Adam

# ------------------------------
# 1. 定义底物（固定）和吸附质（可移动）
# ------------------------------

# 底物原子：位置 (N_sub, 3)，固定不变
substrate_pos = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
    requires_grad=False,
)

# 吸附质原子：以其质心为原点的坐标 (N_ads, 3)
# 例如一个三原子分子（水分子形状，但不严格）
adsorbate_initial = torch.tensor(
    [
        [0.0, 0.0, 0.5],
        [0.0, 0.5, -0.2],
        [0.0, -0.5, -0.2],
    ],
    dtype=torch.float32,
)
# 确保质心在原点
adsorbate_initial = adsorbate_initial - adsorbate_initial.mean(
    dim=0, keepdim=True
)

# 吸附位点（底物上的参考点，相对于底物全局坐标系）
binding_site = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

# ------------------------------
# 2. 定义可训练参数（旋转 + 平移）
# ------------------------------


# 旋转矩阵 R ∈ SO(3)
class RotationModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.R = nn.Parameter(torch.eye(3))
        geotorch.SO(self, "R")  # 约束为特殊正交群

    def forward(self):
        return self.R


rot_module = RotationModule()

# 平移向量 t：通过无界参数重参数化到有界区间 [t_min, t_max]
t_min, t_max = -2.0, 2.0  # 示例约束范围

# 无界参数（可训练）
t_unconstrained = nn.Parameter(torch.zeros(3))


def t_forward():
    """将无界参数通过 sigmoid 映射到 [t_min, t_max]"""
    # 将 t_unconstrained 映射到 (-∞, ∞) -> (0, 1) -> [t_min, t_max]
    sig = torch.sigmoid(t_unconstrained)
    return t_min + (t_max - t_min) * sig


# ------------------------------
# 3. 能量函数（Lennard-Jones 势）
# ------------------------------


def lj_potential(r, epsilon=1.0, sigma=1.0):
    """Lennard-Jones 势能，r: 距离矩阵 (N, M)"""
    r = torch.clamp(r, min=1e-8)  # 避免除零
    sr6 = (sigma / r) ** 6
    sr12 = sr6**2
    return 4 * epsilon * (sr12 - sr6)


def total_energy(ads_pos):
    """ads_pos: 当前吸附质原子的全局坐标 (N_ads, 3)
    返回总能量（标量）
    """
    # 计算所有吸附质原子与所有底物原子之间的距离矩阵
    dist = torch.cdist(ads_pos, substrate_pos)  # (N_ads, N_sub)
    # 求和所有原子对的 LJ 势能
    energy = lj_potential(dist).sum()
    return energy


# ------------------------------
# 4. 优化循环
# ------------------------------

# 收集所有可训练参数（旋转模块的参数 + 无界平移参数）
params = list(rot_module.parameters()) + [t_unconstrained]
optimizer = Adam(params, lr=0.01)

num_steps = 200
for step in range(num_steps):
    optimizer.zero_grad()

    # 获取当前满足 SO(3) 的旋转矩阵
    R = rot_module()

    # 获取当前有界平移向量
    t = t_forward()

    # 计算吸附质原子的全局坐标：
    #  先旋转，再平移到 binding_site + t
    # 注意：吸附质初始坐标以其质心为原点，旋转后质心仍在原点，加上偏移后质心位于 binding_site + t
    ads_global = adsorbate_initial @ R.T + (binding_site + t)

    # 计算总能量
    energy = total_energy(ads_global)

    # 反向传播，自动计算梯度
    energy.backward()

    # 更新参数（geotorch 会自动将 R 的梯度投影回 SO(3) 切空间）
    optimizer.step()

    # 打印信息
    if step % 20 == 0:
        print(
            f"Step {step:4d} | Energy = {energy.item():.6f} | t = {t.detach().numpy()}"
        )

# 最终结果
final_R = rot_module()
final_t = t_forward()
print("\n优化完成")
print(f"最终旋转矩阵:\n{final_R.detach().numpy()}")
print(f"最终平移向量（相对于吸附位点）: {final_t.detach().numpy()}")
print(f"最终吸附质质心位置: {(binding_site + final_t).detach().numpy()}")
