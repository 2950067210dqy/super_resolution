import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# 1. 数据加载 (假设您的CSV文件名为 'particle_tracks.csv')
# 请根据您的实际文件路径和列名（如果不同）进行调整
df = pd.read_csv('particle_tracks.csv')

# 2. 创建3D图形
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 3. 为不同的粒子ID分配颜色
# 获取唯一的粒子ID列表
unique_particle_ids = df['Particle ID'].unique()
unique_particle_ids=unique_particle_ids[0:100]
# 选择颜色映射
cmap = cm.get_cmap('viridis')  # 或者 'plasma', 'tab20' 等
num_particles = len(unique_particle_ids)

# 4. 绘制每个粒子的轨迹
for idx, pid in enumerate(unique_particle_ids):
    # 筛选出当前粒子的数据
    particle_data = df[df['Particle ID'] == pid]

    # 按照时间排序，确保轨迹正确连接
    particle_data = particle_data.sort_values('Particle Time')

    # 为当前粒子分配颜色
    color = cmap(idx / max(num_particles - 1, 1))  # 防止除以零
    # color ='blue'
    # 获取坐标
    x = particle_data['Particle X Position']
    y = particle_data['Particle Y Position']
    z = particle_data['Particle Z Position']

    # 绘制3D轨迹线
    ax.plot(x, y, z,
            color=color,
            linewidth=1.5,
            alpha=0.7,
            label=f'Particle {pid}')

    # # 可选：标记起点和终点
    ax.scatter(x.iloc[0], y.iloc[0], z.iloc[0],
               color=color, s=50, marker='o', depthshade=True)
    ax.scatter(x.iloc[-1], y.iloc[-1], z.iloc[-1],
               color=color, s=50, marker='s', depthshade=True)

# 5. 设置图表属性
ax.set_xlabel('X Position', fontsize=12, labelpad=10)
ax.set_ylabel('Y Position', fontsize=12, labelpad=10)
ax.set_zlabel('Z Position', fontsize=12, labelpad=10)
ax.set_title('3D Particle Trajectories', fontsize=16, pad=20)

# # 6. 添加图例（如果粒子数量过多，可省略或限制显示条目）
# if num_particles <= 100:  # 只在粒子数量适中时显示图例
#     ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=9)
# else:
#     # 粒子太多时，显示数量信息
#     ax.text2D(0.05, 0.95, f'Total Particles: {num_particles}',
#               transform=ax.transAxes, fontsize=10,
#               bbox=dict(boxstyle="round,pad=0.3", facecolor='wheat', alpha=0.5))

# 7. 优化视角
ax.view_init(elev=20, azim=135)  # 调整俯仰角和方位角

# 8. 设置坐标轴比例相等（可选，使三轴比例一致）
# ax.set_aspect('auto')

# 9. 添加网格
ax.grid(True, alpha=0.3)

# 10. 显示图形
plt.tight_layout()
plt.show()

# 可选：保存图形
# fig.savefig('particle_trajectories_3d.png', dpi=300, bbox_inches='tight')

# 打印数据概览
print(f"数据集包含 {len(df)} 行数据")
print(f"包含 {num_particles} 个独立的粒子")
print(f"时间范围: {df['Particle Time'].min():.3f} 到 {df['Particle Time'].max():.3f}")
print(f"位置范围:")
print(f"  X: [{df['Particle X Position'].min():.3f}, {df['Particle X Position'].max():.3f}]")
print(f"  Y: [{df['Particle Y Position'].min():.3f}, {df['Particle Y Position'].max():.3f}]")
print(f"  Z: [{df['Particle Z Position'].min():.3f}, {df['Particle Z Position'].max():.3f}]")