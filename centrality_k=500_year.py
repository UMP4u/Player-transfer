import pandas as pd
import networkx as nx
from tqdm import tqdm

# 读取数据
df = pd.read_csv('D:\\Desktop\\workspace\\workspace\\football\\new_data\\transfers_clean.csv')

# 从 transfer_date 提取年份（日期格式是 dd/mm/yyyy）
df['year'] = pd.to_datetime(df['transfer_date'], dayfirst=True, errors='coerce').dt.year

# 删除年份为空的行
df = df.dropna(subset=['year'])
df['year'] = df['year'].astype(int)

all_results = []

# 按年份循环
for year, group in df.groupby('year'):
    print(f"\n===== Calculating for year {year} =====")
    
    # 构建有向图，权重为转会次数
    G = nx.DiGraph()
    for _, row in tqdm(group.iterrows(), total=len(group), desc=f"Building graph {year}"):
        if G.has_edge(row['from_club_id'], row['to_club_id']):
            G[row['from_club_id']][row['to_club_id']]['weight'] += 1
        else:
            G.add_edge(row['from_club_id'], row['to_club_id'], weight=1)

    # ---- 计算中心性 ----
    print("Calculating in/out degree ...")
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())

    print("Calculating degree centrality ...")
    degree_centrality = nx.degree_centrality(G)

    print("Calculating betweenness centrality (approx, k=500) ...")
    # 大图计算较慢，使用近似版本
    betweenness_centrality = nx.betweenness_centrality(G, k=min(500, len(G)), normalized=True, weight='weight', seed=42)

    print("Calculating eigenvector centrality ...")
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000, tol=1e-08)
    except nx.PowerIterationFailedConvergence:
        print(f"  ⚠ Eigenvector centrality did not converge for {year}, filling with 0.")
        eigenvector_centrality = {node: 0.0 for node in G.nodes()}

    # ---- 合并结果 ----
    yearly_df = pd.DataFrame({
        'year': year,
        'club_id': list(G.nodes()),
        'in_degree': [in_degree.get(node, 0) for node in G.nodes()],
        'out_degree': [out_degree.get(node, 0) for node in G.nodes()],
        'degree_centrality': [degree_centrality.get(node, 0) for node in G.nodes()],
        'betweenness_centrality': [betweenness_centrality.get(node, 0) for node in G.nodes()],
        'eigenvector_centrality': [eigenvector_centrality.get(node, 0) for node in G.nodes()],
    })

    all_results.append(yearly_df)

# 合并所有年份结果
final_df = pd.concat(all_results, ignore_index=True)

# 保存到 CSV
final_df.to_csv('club_centrality_by_year.csv', index=False)
print("\n✅ Saved results to club_centrality_by_year.csv")
