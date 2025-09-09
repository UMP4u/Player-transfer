# club_transfer_graph.py
# -*- coding: utf-8 -*-
import argparse, math, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain   # Louvain 社区检测


# --------------------------- 参数 ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out_dir", default="out_graph")
    p.add_argument("--year_min", type=int, default=None)
    p.add_argument("--year_max", type=int, default=None)
    p.add_argument("--remove_without_club", action="store_true")
    p.add_argument("--weight_mode", choices=["events","unique"], default="events")
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--edge_quantile", type=float, default=0.60)
    p.add_argument("--label_top_n", type=int, default=15)
    p.add_argument("--min_edge_weight", type=int, default=1, help="聚合后保留的最小边权（1=不过滤）")
    p.add_argument("--pretty", action="store_true", help="输出论文风格 PNG")
    p.add_argument("--pretty_max_nodes", type=int, default=20000)
    p.add_argument("--pretty_edge_alpha", type=float, default=0.05)
    p.add_argument("--pretty_label_top_n", type=int, default=60)
    return p.parse_args()

# --------------------------- 读文件/识别列 ---------------------------
def read_csv(csv_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, low_memory=False, encoding="latin1")

def normalize_col(col: str) -> str:
    col = unicodedata.normalize("NFKC", str(col)).strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in col)

def pick_col(cols_norm_map, candidates):
    for raw, norm in cols_norm_map.items():
        for cand in candidates:
            if norm == cand or norm.endswith("_"+cand) or cand in norm:
                return raw
    return None

def detect_columns(df: pd.DataFrame):
    cols_norm = {c: normalize_col(c) for c in df.columns}
    from_candidates = ["from_club_name","from_club","club_from","fromclub","from_club_id"]
    to_candidates   = ["to_club_name","to_club","club_to","toclub","to_club_id"]
    id_candidates   = ["player_id","tm_player_id","sofifa_id","person_id","player"]
    name_candidates = ["player_name","name","player_full_name"]
    date_candidates = ["year","transfer_date","date","transfer_season","season"]

    col_from = pick_col(cols_norm, from_candidates)
    col_to   = pick_col(cols_norm, to_candidates)
    col_pid  = pick_col(cols_norm, id_candidates) or pick_col(cols_norm, name_candidates)
    col_date = pick_col(cols_norm, date_candidates)

    if "player_id" in df.columns: col_pid = "player_id"
    if "from_club_name" in df.columns: col_from = "from_club_name"
    if "to_club_name" in df.columns:   col_to   = "to_club_name"
    return dict(from_col=col_from, to_col=col_to, player_key_col=col_pid, date_col=col_date)

# --------------------------- 清洗/聚合 ---------------------------
def filter_by_time(df: pd.DataFrame, col_date: str, y0: int|None, y1: int|None) -> pd.DataFrame:
    if y0 is None and y1 is None: return df
    df2 = df.copy()
    if "year" in df2.columns:
        df2["__year__"] = pd.to_numeric(df2["year"], errors="coerce")
    elif col_date and col_date in df2.columns:
        dt = pd.to_datetime(df2[col_date], errors="coerce")
        df2["__year__"] = dt.dt.year
    else:
        df2["__year__"] = np.nan
    if y0 is not None: df2 = df2[df2["__year__"] >= y0]
    if y1 is not None: df2 = df2[df2["__year__"] <= y1]
    return df2.drop(columns=["__year__"])

def build_work_df(df: pd.DataFrame, cols: dict, remove_without_club: bool) -> pd.DataFrame:
    work = df[[cols["from_col"], cols["to_col"], cols["player_key_col"]]].copy()
    work.columns = ["from_club","to_club","player_key"]
    work = work.dropna()
    work = work[(work["from_club"].astype(str).str.len()>0) & (work["to_club"].astype(str).str.len()>0)]
    work = work[work["from_club"] != work["to_club"]]
    if remove_without_club:
        work = work[(work["from_club"]!="Without Club") & (work["to_club"]!="Without Club")]
    return work

def aggregate_edges(work: pd.DataFrame, weight_mode: str) -> pd.DataFrame:
    agg_weight = ("player_key", "size") if weight_mode=="events" else ("player_key", pd.Series.nunique)
    edge_counts = (
        work.groupby(["from_club","to_club"])
            .agg(weight=agg_weight, unique_players=("player_key", pd.Series.nunique))
            .reset_index()
            .sort_values("weight", ascending=False)
    )
    return edge_counts

def build_graph(edge_counts: pd.DataFrame) -> tuple[nx.DiGraph, dict, dict, dict]:
    G = nx.DiGraph()
    for _, r in edge_counts.iterrows():
        G.add_edge(r["from_club"], r["to_club"], weight=int(r["weight"]), unique_players=int(r["unique_players"]))
    out_strength = dict(G.out_degree(weight="weight"))
    in_strength  = dict(G.in_degree(weight="weight"))
    total_strength = {n: out_strength.get(n,0) + in_strength.get(n,0) for n in G.nodes()}
    return G, out_strength, in_strength, total_strength

def export_artifacts(edge_counts: pd.DataFrame, G: nx.DiGraph, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    edges_csv = out_dir / "global_transfers_edges.csv"
    gexf_path = out_dir / "club_transfers_graph.gexf"
    edge_counts.to_csv(edges_csv, index=False)
    nx.write_gexf(G, gexf_path)
    return edges_csv, gexf_path

# --------------------------- 论文风格可视化（spring_layout） ---------------------------
def visualize_pretty(
    G: nx.DiGraph,
    out_png_path: str,
    max_nodes: int = 20000,
    draw_edges: bool = True,
    edge_alpha: float = 0.06,
    node_min: int = 2,
    node_max: int = 16,
    label_top_n: int = 60,
    seed: int = 42,
):
    import random
    random.seed(seed); np.random.seed(seed)
    if G.number_of_nodes() == 0:
        print("Empty graph, skip pretty drawing.")
        return

    # 最大弱连通分量
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    H = G.subgraph(largest_cc).copy()

    if H.number_of_nodes() > max_nodes:
        strength = dict(H.degree(weight="weight"))
        keep = [n for n,_ in sorted(strength.items(), key=lambda x: x[1], reverse=True)[:max_nodes]]
        H = H.subgraph(keep).copy()

    # Louvain 社区
    undirected = H.to_undirected()
    part = community_louvain.best_partition(undirected, random_state=seed, weight="weight")

    # spring 布局
    k = 0.12 / np.sqrt(max(1, H.number_of_nodes()))
    pos = nx.spring_layout(H, seed=seed, k=k, iterations=500, weight="weight")

    # 节点大小
    strength = dict(H.degree(weight="weight"))
    s_vals = np.array([strength.get(n, 0) for n in H.nodes()], dtype=float)
    s_min, s_max = float(np.min(s_vals)), float(np.max(s_vals))
    if s_max > s_min:
        sizes = node_min + (node_max - node_min) * (s_vals - s_min) / (s_max - s_min)
    else:
        sizes = np.full(len(s_vals), (node_min + node_max)/2.0)

    # 社区着色
    import matplotlib.cm as cm
    cmap = cm.get_cmap("tab20")
    comms = sorted(set(part.values()))
    color_map = {c: cmap(i % cmap.N) for i, c in enumerate(comms)}
    node_colors = [color_map[part[n]] for n in H.nodes()]

    # 绘制
    plt.figure(figsize=(14, 12))
    if draw_edges and H.number_of_edges() > 0:
        w_all = [d.get("weight", 1) for _,_,d in H.edges(data=True)]
        w_min, w_max = min(w_all), max(w_all)
        denom = (w_max - w_min) if (w_max - w_min) > 0 else 1.0
        ew = [0.2 + 1.2 * (d.get("weight", 1) - w_min) / denom for _,_,d in H.edges(data=True)]
        nx.draw_networkx_edges(H, pos, arrows=False, width=ew, alpha=edge_alpha)
    nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color=node_colors, linewidths=0, alpha=0.95)

    # labels = {n: n for n in sorted(H.nodes(), key=lambda n: strength.get(n, 0), reverse=True)[:label_top_n]}
    # nx.draw_networkx_labels(H, pos, labels=labels, font_size=7, font_weight="bold")

    plt.axis("off"); plt.tight_layout()
    plt.savefig(out_png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Pretty] Saved: {out_png_path}")

# --------------------------- 主流程 ---------------------------
def main():
    args = parse_args()
    csv_path = Path(args.csv); out_dir = Path(args.out_dir)
    assert csv_path.exists(), f"找不到文件：{csv_path}"

    df = read_csv(csv_path)
    cols = detect_columns(df)
    df = filter_by_time(df, cols["date_col"], args.year_min, args.year_max)
    work = build_work_df(df, cols, args.remove_without_club)

    edge_counts = aggregate_edges(work, args.weight_mode)
    if args.min_edge_weight > 1:
        edge_counts = edge_counts[edge_counts["weight"] >= args.min_edge_weight]

    G, out_s, in_s, total_s = build_graph(edge_counts)
    edges_csv, gexf_path = export_artifacts(edge_counts, G, out_dir)

    if args.pretty:
        visualize_pretty(
            G, str(out_dir/"club_transfers_pretty.png"),
            max_nodes=args.pretty_max_nodes,
            edge_alpha=args.pretty_edge_alpha,
            label_top_n=args.pretty_label_top_n,
            node_min=2, node_max=16, seed=42
        )

    print(f"Edges CSV: {edges_csv}")
    print(f"GEXF:      {gexf_path}")
    if args.pretty:
        print(f"Pretty PNG:{out_dir/'club_transfers_pretty.png'}")

if __name__ == "__main__":
    main()
