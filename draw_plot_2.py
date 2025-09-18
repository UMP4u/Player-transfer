import argparse
import math
import random
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_adjacency(df):
    """构建加权无向图的邻接表，键是俱乐部ID，值是相邻俱乐部及转会次数。"""
    adj = defaultdict(lambda: defaultdict(int))
    for f, t in zip(df['from_club_id'], df['to_club_id']):
        adj[f][t] += 1
        adj[t][f] += 1
    return adj


def louvain_first_phase(graph):
    """Louvain算法的第一阶段：在不合并社区的情况下迭代提升模块度。

    每个节点初始在独立社区中，遍历所有节点，尝试将节点移动到邻居所在社区，
    只有当移动能提升模块度才执行:contentReference[oaicite:2]{index=2}。当一轮迭代中没有任何节点
    移动时算法结束。返回各节点的社区ID。
    """
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
            # 统计节点v与每个邻居社区之间的加权连接数
            neigh_comm_wt = defaultdict(float)
            for u, w in graph[v].items():
                neigh_comm_wt[community[u]] += w

            # 从当前社区中移除v
            tot[com_v] -= k_v
            in_w[com_v] -= 2 * neigh_comm_wt.get(com_v, 0.0)
            community[v] = None

            # 选择能最大提升模块度的邻居社区
            best_com = com_v
            best_gain = 0.0
            for com, wt in neigh_comm_wt.items():
                gain = wt - k_v * tot[com] / (2 * m)
                if gain > best_gain:
                    best_gain = gain
                    best_com = com

            # 将v加入最佳社区
            community[v] = best_com
            tot[best_com] += k_v
            in_w[best_com] += 2 * neigh_comm_wt.get(best_com, 0.0)
            if best_com != com_v:
                improved = True
    return community


def compute_layout(graph, communities, base_sigma=0.05, bridge_threshold=0.5):
    """
    根据社区将每个节点定位到圆周上的团块，并对跨社区节点重新摆放位置，
    添加随机扰动使同社区节点分散开来，最后归一化坐标到[0,1]区间。
    """
    comm_ids = list(set(communities.values()))
    num_comm = len(comm_ids)

    # 每个社区中心均匀分布在单位圆上
    community_centres = {}
    for idx, c in enumerate(comm_ids):
        angle = 2 * math.pi * idx / num_comm
        community_centres[c] = np.array([math.cos(angle), math.sin(angle)])

    # 计算每个节点的跨社区邻居比例
    bridge_ratio = {}
    neighbour_comm_sets = {}
    for node, neighbours in graph.items():
        comm_set = {communities[u] for u in neighbours}
        neighbour_comm_sets[node] = comm_set
        degree = len(neighbours)
        if degree == 0:
            bridge_ratio[node] = 0.0
        else:
            cross = sum(1 for u in neighbours if communities[u] != communities[node])
            bridge_ratio[node] = cross / degree

    # 计算节点坐标：普通节点在本社区中心附近，桥梁节点居两个或多个社区中心之间
    positions = {}
    for node in graph:
        comm = communities[node]
        centre = community_centres[comm]
        if bridge_ratio[node] >= bridge_threshold:
            # 取邻居所有社区中心的平均值
            neigh_comms = neighbour_comm_sets[node]
            centres = [community_centres[c] for c in neigh_comms]
            if comm not in neigh_comms:
                centres.append(centre)
            base = np.mean(centres, axis=0)
        else:
            base = centre
        # 加入正态分布扰动
        jitter = np.random.normal(loc=0.0, scale=base_sigma, size=2)
        positions[node] = base + jitter

    # 归一化到[0,1]范围
    pos_arr = np.array(list(positions.values()))
    min_vals = pos_arr.min(axis=0)
    max_vals = pos_arr.max(axis=0)
    range_vals = max_vals - min_vals + 1e-9
    for node in positions:
        positions[node] = (positions[node] - min_vals) / range_vals
    return positions


def sample_edges(graph, max_edges_per_node=3):
    """
    为了避免绘制所有约24万条边导致图形过于杂乱，仅从每个节点的邻居中随机选取最多
    ``max_edges_per_node`` 条边用于绘图。
    """
    sampled_edges = set()
    if max_edges_per_node <= 0:
        return sampled_edges

    for v, neighbours in graph.items():
        nbr_list = list(neighbours.keys())
        if len(nbr_list) > max_edges_per_node:
            chosen = np.random.choice(nbr_list, size=max_edges_per_node, replace=False)
        else:
            chosen = nbr_list
        for u in chosen:
            a, b = (v, u) if v < u else (u, v)
            sampled_edges.add((a, b))
    return sampled_edges


def plot_network(graph, communities, positions, edge_pairs, out_path,
                 node_size=2, bridge_threshold=0.5):
    """
    按社区颜色绘制节点，桥梁节点用黑色轮廓标记，绘制采样的边。
    """
    comm_ids = list(set(communities.values()))
    comm_index = {c: i for i, c in enumerate(comm_ids)}
    cmap = plt.cm.get_cmap('tab20', len(comm_ids))

    # 判断桥梁节点
    bridge_nodes = []
    for node, neighbours in graph.items():
        degree = len(neighbours)
        if degree == 0:
            continue
        cross = sum(1 for u in neighbours if communities[u] != communities[node])
        ratio = cross / degree
        if ratio >= bridge_threshold:
            bridge_nodes.append(node)

    all_nodes = list(graph.keys())
    xy = np.array([positions[n] for n in all_nodes])
    colours = [cmap(comm_index[communities[n]]) for n in all_nodes]
    is_bridge = np.array([n in bridge_nodes for n in all_nodes])

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    # 绘制采样边
    for u, v in edge_pairs:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        ax.plot([x1, x2], [y1, y2],
                color=(0.6, 0.6, 0.6, 0.03), linewidth=0.15)
    # 绘制普通节点
    non_bridge_mask = ~is_bridge
    ax.scatter(xy[non_bridge_mask, 0], xy[non_bridge_mask, 1],
               s=node_size, c=[colours[i] for i, flag in enumerate(is_bridge) if not flag],
               edgecolors='none', marker='o')
    # 绘制桥梁节点：稍大且有黑色边框
    bridge_mask = is_bridge
    ax.scatter(xy[bridge_mask, 0], xy[bridge_mask, 1],
               s=node_size * 3, c=[colours[i] for i, flag in enumerate(is_bridge) if flag],
               edgecolors='black', linewidths=0.2, marker='o')

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='普通俱乐部',
               markerfacecolor='grey', markersize=6),
        Line2D([0], [0], marker='o', color='w', label='跨社区活跃俱乐部',
               markerfacecolor='grey', markeredgecolor='black', markersize=6)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_title('俱乐部转会网络社区布局 (近似力导)\n', fontsize=12)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    # 示例调用：根据需要修改路径
    csv_path = 'D:\\Desktop\\workspace\\workspace\\football\\new_data\\transfers_clean.csv'
    output_path = 'all_nodes_network.png'

    # 读取数据并构建图
    df = pd.read_csv(csv_path)
    graph = build_adjacency(df)

    # 运行 Louvain 社区发现
    print('Detecting communities...')
    communities = louvain_first_phase(graph)
    print(f'共识别出 {len(set(communities.values()))} 个社区')

    # 布局计算，计算节点坐标
    positions = compute_layout(graph, communities,
                               base_sigma=0.05, bridge_threshold=0.5)

    # 抽样部分边用于绘图
    edge_pairs = sample_edges(graph, max_edges_per_node=3)

    # 绘图并保存
    plot_network(graph, communities, positions, edge_pairs, output_path,
                 node_size=2, bridge_threshold=0.5)

    print(f'网络图已保存为 {output_path}')
