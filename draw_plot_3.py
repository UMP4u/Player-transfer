import pandas as pd
import numpy as np
import math
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# 读取数据并构建加权无向图
csv_path = 'D:\\Desktop\\workspace\\workspace\\football\\new_data\\transfers_clean.csv'
df = pd.read_csv(csv_path)
adj = defaultdict(lambda: defaultdict(int))
for f, t in zip(df['from_club_id'], df['to_club_id']):
    adj[f][t] += 1
    adj[t][f] += 1

# Louvain算法第一阶段：划分社区
def louvain_first_phase(graph):
    m = sum(sum(w for w in nbrs.values()) for nbrs in graph.values()) / 2
    community = {node: node for node in graph}
    tot = {node: sum(graph[node].values()) for node in graph}
    in_w = {node: 0.0 for node in graph}
    improved = True
    while improved:
        improved = False
        for v in graph:
            com_v = community[v]
            k_v = tot[v]
            neigh_comm_wt = defaultdict(float)
            for u, w in graph[v].items():
                neigh_comm_wt[community[u]] += w
            tot[com_v] -= k_v
            in_w[com_v] -= 2 * neigh_comm_wt.get(com_v, 0.0)
            community[v] = None
            best_com = com_v
            best_gain = 0.0
            for com, wt in neigh_comm_wt.items():
                gain = wt - k_v * tot[com] / (2 * m)
                if gain > best_gain:
                    best_gain = gain
                    best_com = com
            community[v] = best_com
            tot[best_com] += k_v
            in_w[best_com] += 2 * neigh_comm_wt.get(best_com, 0.0)
            if best_com != com_v:
                improved = True
    return community

communities = louvain_first_phase(adj)
comm_sizes = Counter(communities.values())

# 为每个社区随机分配中心位置
np.random.seed(42)
comm_ids = list(set(communities.values()))
comm_centers = {c: np.random.rand(2) for c in comm_ids}

# 计算每个节点的跨社区邻居比例
neighbor_comm_sets = {}
bridge_ratio = {}
for node, nbrs in adj.items():
    comm_set = {communities[u] for u in nbrs}
    neighbor_comm_sets[node] = comm_set
    degree = len(nbrs)
    if degree == 0:
        bridge_ratio[node] = 0
    else:
        cross = sum(1 for u in nbrs if communities[u] != communities[node])
        bridge_ratio[node] = cross / degree

# 根据社区中心和抖动计算每个节点的位置
positions = {}
base_sigma = 0.03
size_constant = 50    # 调节社区大小对抖动的影响，值越小团块越大
bridge_threshold = 0.5
for node in adj:
    c = communities[node]
    center = comm_centers[c]
    size = comm_sizes[c]
    # 根据社区规模调整抖动半径
    jitter_scale = base_sigma * (math.sqrt(size) / size_constant)
    jitter_scale = min(jitter_scale, 0.15)  # 限制最大抖动
    if bridge_ratio[node] >= bridge_threshold:
        # 跨社区节点：按邻居社区中心平均值定位
        neigh = neighbor_comm_sets[node]
        centres = [comm_centers[x] for x in neigh]
        if c not in neigh:
            centres.append(center)
        base = np.mean(centres, axis=0)
        jitter = np.random.normal(scale=jitter_scale * 0.3, size=2)
        positions[node] = base + jitter
    else:
        jitter = np.random.normal(scale=jitter_scale, size=2)
        positions[node] = center + jitter

# 归一化到 [0,1] 区间便于绘图
pos_arr = np.array(list(positions.values()))
min_vals, max_vals = pos_arr.min(axis=0), pos_arr.max(axis=0)
for node in positions:
    positions[node] = (positions[node] - min_vals) / (max_vals - min_vals + 1e-9)

# 为了避免画出所有边导致拥挤，随机抽取部分边来显示
max_edges_per_node = 2   # 每个节点最多抽取2条边绘制
edge_pairs = set()
for v, nbrs in adj.items():
    neigh = list(nbrs.keys())
    if len(neigh) > max_edges_per_node:
        sampled = np.random.choice(neigh, size=max_edges_per_node, replace=False)
    else:
        sampled = neigh
    for u in sampled:
        a, b = (v, u) if v < u else (u, v)
        edge_pairs.add((a, b))

# 绘图
cmap = plt.cm.get_cmap('tab20', len(comm_ids))
comm_index = {c: i for i, c in enumerate(comm_ids)}
bridging_nodes = [n for n, r in bridge_ratio.items() if r >= bridge_threshold]
all_nodes = list(adj.keys())

plt.figure(figsize=(12, 8), dpi=150)
# 绘制抽样的边
for u, v in edge_pairs:
    x1, y1 = positions[u]
    x2, y2 = positions[v]
    plt.plot([x1, x2], [y1, y2], color=(0.6, 0.6, 0.6, 0.02), linewidth=0.15)

# 绘制普通节点
non_bridge = [n for n in all_nodes if n not in bridging_nodes]
xy_non = np.array([positions[n] for n in non_bridge])
clr_non = [cmap(comm_index[communities[n]]) for n in non_bridge]
plt.scatter(xy_non[:,0], xy_non[:,1], s=2, c=clr_non, edgecolors='none')

# 绘制桥梁节点：使用黑色轮廓突出
xy_bridge = np.array([positions[n] for n in bridging_nodes])
clr_bridge = [cmap(comm_index[communities[n]]) for n in bridging_nodes]
plt.scatter(xy_bridge[:,0], xy_bridge[:,1], s=6, c=clr_bridge,
            edgecolors='black', linewidths=0.2)

plt.title('俱乐部转会网络——随机社区布局', fontsize=12)
plt.axis('off')
plt.tight_layout()
plt.savefig('transfer_network_random_layout.png', dpi=300)
print('可视化已保存为 transfer_network_random_layout.png')
