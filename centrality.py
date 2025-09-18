import pandas as pd
import networkx as nx
from tqdm import tqdm

# 读取CSV数据
df = pd.read_csv('D:\\Desktop\\workspace\\workspace\\football\\new_data\\transfers_clean.csv')

# 构建有向图，权重为转会次数（每条转会为1）
G = nx.DiGraph()

# 统计每对俱乐部之间的转会次数（边的权重）
for _, row in tqdm(df.iterrows(), total=len(df), desc="Building graph"):
    if G.has_edge(row['from_club_id'], row['to_club_id']):
        G[row['from_club_id']][row['to_club_id']]['weight'] += 1
    else:
        G.add_edge(row['from_club_id'], row['to_club_id'], weight=1)

print("Calculating in/out degree ...")
in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())

print("Calculating degree centrality ...")
degree_centrality = nx.degree_centrality(G)

print("Calculating betweenness centrality (this may take time) ...")
# tqdm 版本的 betweenness
betweenness_centrality = nx.betweenness_centrality(G, normalized=True, weight='weight', seed=42)

print("Calculating eigenvector centrality ...")
eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000, tol=1e-08)

# ---- 合并结果 ----
centrality_df = pd.DataFrame({
    'club_id': list(G.nodes()),
    'in_degree': [in_degree.get(node, 0) for node in G.nodes()],
    'out_degree': [out_degree.get(node, 0) for node in G.nodes()],
    'degree_centrality': [degree_centrality.get(node, 0) for node in G.nodes()],
    'betweenness_centrality': [betweenness_centrality.get(node, 0) for node in G.nodes()],
    'eigenvector_centrality': [eigenvector_centrality.get(node, 0) for node in G.nodes()],
})

centrality_df.to_csv('club_centrality_measures_by_transfer_count.csv', index=False)
print("Saved results to club_centrality_measures_by_transfer_count.csv")