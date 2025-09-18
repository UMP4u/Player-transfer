# -*- coding: utf-8 -*-
import pandas as pd
import networkx as nx
from tqdm import tqdm

INPUT_CSV = "D:\\Desktop\\workspace\\workspace\\football\\new_data\\transfers_clean.csv"  # 替换为你的CSV
OUTPUT_CSV = "club_centrality_by_year_month.csv"

df = pd.read_csv(INPUT_CSV)
df["date"] = pd.to_datetime(df["transfer_date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["date"]).copy()
df["year"] = df["date"].dt.year.astype(int)
df["month"] = df["date"].dt.month.astype(int)

# 先聚合为加权边（权重=当月该方向转会次数）
edges_all = (
    df.groupby(["year", "month", "from_club_id", "to_club_id"], dropna=False)
      .size().reset_index(name="weight")
)

periods = edges_all[["year", "month"]].drop_duplicates().sort_values(["year","month"]).itertuples(index=False, name=None)
results = []

def robust_eigenvector_centrality(G: nx.DiGraph, weight="weight"):
    """
    更稳健的EC计算：
    1) 直接算（加大迭代上限）
    2) 加极小自环再算（破除周期/二分）
    3) 取最大弱连通分量 -> 无向图再算，其他点补0
    4) 仍失败则 Katz 作为回退（与EC亲缘近）
    """
    try:
        return nx.eigenvector_centrality(G, weight=weight, max_iter=20000, tol=1e-9)
    except nx.PowerIterationFailedConvergence:
        pass

    # 2) 加极小自环
    eps = 1e-12
    H = G.copy()
    for n in H.nodes():
        if not H.has_edge(n, n):
            H.add_edge(n, n, **{weight: eps})
        else:
            H[n][n][weight] = H[n][n].get(weight, 0.0) + eps
    try:
        return nx.eigenvector_centrality(H, weight=weight, max_iter=20000, tol=1e-9)
    except nx.PowerIterationFailedConvergence:
        pass

    # 3) 最大弱连通分量 + 无向
    if H.number_of_nodes() == 0:
        return {}
    wccs = list(nx.weakly_connected_components(H))
    if not wccs:
        return {}
    giant = H.subgraph(max(wccs, key=len)).copy()
    Ug = giant.to_undirected()
    try:
        ec_giant = nx.eigenvector_centrality(Ug, weight=weight, max_iter=20000, tol=1e-9)
        # 其他不在giant的点补0
        out = {n: 0.0 for n in G.nodes()}
        out.update(ec_giant)
        return out
    except nx.PowerIterationFailedConvergence:
        pass

    # 4) 仍失败 → Katz 作为回退（一定收敛，alpha 取保守值）
    try:
        # 用无向图的 Katz，保证非强连通也收敛
        katz = nx.katz_centrality(Ug, alpha=0.05, beta=1.0, weight=weight, max_iter=20000, tol=1e-9)
        out = {n: 0.0 for n in G.nodes()}
        out.update(katz)
        return out
    except Exception:
        # 最后的兜底
        return {n: 0.0 for n in G.nodes()}

print("开始按【年-月】计算中心性 ...")
for (yy, mm) in tqdm(list(periods), desc="Periods"):
    sub = edges_all[(edges_all["year"] == yy) & (edges_all["month"] == mm)]
    if sub.empty:
        continue

    # 构图：权重=次数
    G = nx.DiGraph()
    G.add_weighted_edges_from(
        sub[["from_club_id", "to_club_id", "weight"]].itertuples(index=False, name=None),
        weight="weight"
    )
    if G.number_of_nodes() == 0:
        continue

    # 度 & 度中心性（度=入+出；如果你要分别的中心性，可另外算 in/out degree centrality）
    in_deg  = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    deg_cent = nx.degree_centrality(G)

    # 介数：distance=1/weight（强度越大，距离越近），大图用近似k
    H = G.copy()
    for u, v, data in H.edges(data=True):
        w = float(data.get("weight", 1.0))
        data["distance"] = 1.0 / w if w > 0 else float("inf")
    n = H.number_of_nodes()
    k = min(500, n) if n >= 300 else None
    btw = nx.betweenness_centrality(H, normalized=True, weight="distance", k=k, seed=42)

    # 更稳健的EC
    eig = robust_eigenvector_centrality(G, weight="weight")

    # 汇总
    nodes = list(G.nodes())
    period_df = pd.DataFrame({
        "year": yy,
        "month": mm,
        "club_id": nodes,
        "in_degree": [in_deg.get(n, 0) for n in nodes],
        "out_degree": [out_deg.get(n, 0) for n in nodes],
        "degree_centrality": [deg_cent.get(n, 0.0) for n in nodes],
        "betweenness_centrality": [btw.get(n, 0.0) for n in nodes],
        "eigenvector_centrality": [eig.get(n, 0.0) for n in nodes],
    })
    results.append(period_df)

final_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame(
    columns=["year","month","club_id","in_degree","out_degree","degree_centrality","betweenness_centrality","eigenvector_centrality"]
)
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ 已保存：{OUTPUT_CSV}  （periods={len(results)}）")
