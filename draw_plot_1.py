import pandas as pd
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import random
import math

random.seed(42)
np.random.seed(42)

# 读取数据
csv_path = 'D:\\Desktop\\workspace\\workspace\\football\\new_data\\transfers_clean.csv'
df = pd.read_csv(csv_path)

# 计算俱乐部转会活跃度（出入总次数）
degree_counter = Counter()
for f, t in zip(df['from_club_id'], df['to_club_id']):
    degree_counter[f] += 1
    degree_counter[t] += 1

# 选取转会次数最高的前 N 个俱乐部
N = 150000
top_nodes = [club for club, _ in degree_counter.most_common(N)]
top_set = set(top_nodes)

# 构建加权无向图：字典的字典表示
adj = defaultdict(lambda: defaultdict(int))
for f, t in zip(df['from_club_id'], df['to_club_id']):
    if f in top_set and t in top_set:
        adj[f][t] += 1
        adj[t][f] += 1

# ------ Louvain 算法第一阶段函数 ------

def total_edges(graph):
    """图中加权边数的一半（无向图）"""
    total = sum(w for u in graph for w in graph[u].values())
    return total / 2

def get_node_weight(graph, node):
    return sum(graph[node].values())

def initialize_communities(graph):
    # 每个节点初始作为独立社区
    return {node: [node] for node in graph}

def get_current_community(communities, node):
    for comm, nodes in communities.items():
        if node in nodes:
            return comm
    return None

def get_community_weight_tot(graph, communities, comm):
    return sum(get_node_weight(graph, n) for n in communities[comm])

def get_community_weight_in(graph, communities, comm):
    weight = 0
    nodes = communities[comm]
    node_set = set(nodes)
    for n in nodes:
        for nbr, w in graph[n].items():
            if nbr in node_set:
                weight += w
    return weight

def get_self_modularity(graph, node, m):
    edge_to_self = graph[node].get(node, 0)
    node_tot = get_node_weight(graph, node)
    return (2*edge_to_self/(2*m)) - (node_tot/(2*m))**2

def modularity_community(graph, communities, comm, m):
    tot = get_community_weight_tot(graph, communities, comm)
    in_w = get_community_weight_in(graph, communities, comm)
    return in_w/(2*m) - (tot/(2*m))**2

def delta_node_to_other_community(graph, communities, node, comm, m):
    if node in communities[comm]:
        return 0
    node_in = sum(w for nbr, w in graph[node].items() if nbr in communities[comm])
    node_tot = get_node_weight(graph, node)
    q_after = (
        (get_community_weight_in(graph, communities, comm) + node_in)/(2*m)
        - ((get_community_weight_tot(graph, communities, comm) + node_tot)/(2*m))**2
    )
    q_before = modularity_community(graph, communities, comm, m) + get_self_modularity(graph, node, m)
    return q_after - q_before

def delta_node_to_remove_community(graph, communities, node, comm, m):
    if node not in communities[comm]:
        return 0
    node_in = sum(w for nbr, w in graph[node].items() if nbr in communities[comm])
    node_tot = get_node_weight(graph, node)
    q_after = (
        (get_community_weight_in(graph, communities, comm) - node_in)/(2*m)
        - ((get_community_weight_tot(graph, communities, comm) - node_tot)/(2*m))**2
        + get_self_modularity(graph, node, m)
    )
    q_before = modularity_community(graph, communities, comm, m)
    return q_after - q_before

def delta_total(graph, communities, node, from_comm, to_comm, m):
    return (
        delta_node_to_other_community(graph, communities, node, to_comm, m)
        + delta_node_to_remove_community(graph, communities, node, from_comm, m)
    )

def get_best_community(graph, communities, node, m):
    current = get_current_community(communities, node)
    best = current
    best_delta = 0
    for comm in communities.keys():
        delta = delta_total(graph, communities, node, current, comm, m)
        if delta > best_delta + 1e-9:
            best_delta = delta
            best = comm
    return best

def louvain_first_phase(graph):
    """简单实现 Louvain 算法的第一阶段"""
    m = total_edges(graph)
    communities = initialize_communities(graph)
    change_count = 10
    while change_count > 0:
        change_count = 0
        for node in list(graph.keys()):
            current_comm = get_current_community(communities, node)
            best_comm = get_best_community(graph, communities, node, m)
            if best_comm != current_comm:
                communities[current_comm].remove(node)
                communities[best_comm].append(node)
                change_count += 1
    return communities

# 运行社区划分
communities = louvain_first_phase(adj)
community_list = [nodes for nodes in communities.values() if nodes]
node_to_comm = {node: i for i, nodes in enumerate(community_list) for node in nodes}

# ------ 自定义 Fruchterman–Reingold 布局 ------
def fruchterman_reingold_layout(graph, iterations=50, area=1.0):
    nodes = list(graph.keys())
    N = len(nodes)
    if N == 0:
        return {}
    k = math.sqrt(area / N)
    positions = {node: np.array([random.random(), random.random()]) for node in nodes}
    t = 0.1  # 初始温度
    for _ in range(iterations):
        disp = {node: np.array([0.0, 0.0]) for node in nodes}
        # 斥力
        for i in range(N):
            v = nodes[i]
            for j in range(i+1, N):
                u = nodes[j]
                delta = positions[v] - positions[u]
                dist = np.linalg.norm(delta) + 1e-9
                force = (k * k) / dist
                disp[v] += (delta / dist) * force
                disp[u] -= (delta / dist) * force
        # 引力
        for v in nodes:
            for u, w in graph[v].items():
                if v == u:
                    continue
                delta = positions[v] - positions[u]
                dist = np.linalg.norm(delta) + 1e-9
                force = (dist * dist) / k * w
                disp[v] -= (delta / dist) * force
                disp[u] += (delta / dist) * force
        # 更新位置
        for v in nodes:
            d = disp[v]
            disp_len = np.linalg.norm(d)
            if disp_len > 0:
                step = min(disp_len, t)
                positions[v] += (d / disp_len) * step
        t *= 0.95  # 降温
    # 归一化坐标到 [0,1]
    pos_arr = np.array(list(positions.values()))
    min_vals = pos_arr.min(axis=0)
    max_vals = pos_arr.max(axis=0)
    scale = max_vals - min_vals
    for v in nodes:
        positions[v] = (positions[v] - min_vals) / (scale + 1e-9)
    return positions

positions = fruchterman_reingold_layout(adj, iterations=50, area=1.0)

# 计算桥梁节点比例
bridging_ratio = {}
for node in top_nodes:
    neighbors = adj[node]
    if not neighbors:
        bridging_ratio[node] = 0
        continue
    diff_comm_edges = sum(
        1 for nbr in neighbors if node_to_comm[nbr] != node_to_comm[node]
    )
    bridging_ratio[node] = diff_comm_edges / len(neighbors)

# 绘图
plt.figure(figsize=(12, 8))
num_comms = len(community_list)
cmap = plt.cm.get_cmap('tab20', num_comms)

# 绘制边（只显示权重≥5的边）
threshold = 5
for u in adj:
    for v, w in adj[u].items():
        if u < v and w >= threshold:
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            alpha = min(0.2 + w / 30, 1.0)
            plt.plot([x1, x2], [y1, y2], color=(0.7, 0.7, 0.7, alpha), linewidth=0.5)

# 绘制节点
max_degree = max(degree_counter[n] for n in top_nodes)
for node in top_nodes:
    comm = node_to_comm[node]
    color = cmap(comm)
    x, y = positions[node]
    size = (degree_counter[node] / max_degree) * 500 + 50
    if bridging_ratio[node] >= 0.5:
        # 跨社区联系比例高，绘制黑边
        plt.scatter(x, y, s=size, c=[color], edgecolors='black', linewidths=1.0, zorder=3)
    else:
        plt.scatter(x, y, s=size, c=[color], edgecolors='none', zorder=3)

# 图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='社区节点', markerfacecolor='grey', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='桥梁俱乐部 (跨社区活跃)', markerfacecolor='grey', markeredgecolor='black', markersize=10)
]
plt.legend(handles=legend_elements, loc='upper right')

plt.title(f'Top {N} 俱乐部转会网络 (Louvain 社区 + Fruchterman–Reingold 布局)')
plt.axis('off')
plt.tight_layout()

# 保存图像
plt.savefig('transfer_network.png', dpi=300)
plt.show()
