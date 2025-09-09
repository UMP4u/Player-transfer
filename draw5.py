#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Big-5 transfer network — compact, no-legend, safe-bounds, TOP-N internal flows.
兼容：CSV 中联赛可为 GB1/ES1/… 或 EPL/LaLiga/… 的任意写法。
"""

# —— 可选依赖：没有 adjustText 也能跑 —— #
try:
    from adjustText import adjust_text
except Exception:
    def adjust_text(*args, **kwargs):
        return None

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import networkx as nx
from matplotlib.patches import FancyArrowPatch

import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams['font.family'] = 'DejaVu Sans'


def read_csv_flex(path):
    # 常见编码按优先级轮询；需要时你可调整次序
    tried = []
    for enc in ["utf-8", "utf-8-sig", "gbk", "cp936", "latin1"]:
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError as e:
            tried.append(f"{enc}({e.reason})")
    # 实在不行，用最宽容的 latin1 强行读（不会报错，但会保留原字节）
    return pd.read_csv(path, low_memory=False, encoding="latin1")




# —— 联赛别名归一化（无论 CSV 写 GB1 / ES1 / La Liga / LIGUE 1 都归到短名） —— #
LEAGUE_ALIASES = {
    # 旧代码 → 短名
    "GB1": "EPL", "ES1": "LaLiga", "IT1": "SerieA", "L1": "Bundesliga", "FR1": "Ligue1",
    # 常见写法 → 短名（大小写/空格不敏感）
    "EPL": "EPL",
    "LA LIGA": "LaLiga", "LALIGA": "LaLiga",
    "SERIE A": "SerieA", "SERIEA": "SerieA",
    "BUNDESLIGA": "Bundesliga",
    "LIGUE 1": "Ligue1", "LIGUE1": "Ligue1",
}
def norm_league(x):
    if x is None: return None
    s = str(x).strip().replace("\u00A0", " ")   # 去掉不间断空格
    key = " ".join(s.upper().split())           # 统一大写与多空格
    return LEAGUE_ALIASES.get(key, s)           # 不认识的就原样返回

# 用“短名”作为配色键
LEAGUE_COLORS = {
    "EPL": "#f28e2b",
    "LaLiga": "#ffdf5d",
    "SerieA": "#4e79a7",
    "Bundesliga": "#59a14f",
    "Ligue1": "#e15759",
}

def normalize_season(s):
    if not isinstance(s, str): return s
    t = s.strip().replace(" ", "").replace("-", "/")
    a = t.split("/")
    if len(a)==2 and a[0].isdigit() and a[1].isdigit():
        y0 = int(a[0]) if len(a[0])==4 else 2000+int(a[0])
        y1 = int(a[1]); return f"{y0}/{y1:02d}"
    return s

def sqrt_scale(values: pd.Series, out_min: float, out_max: float) -> pd.Series:
    v = values.astype(float).clip(lower=0)
    s = np.sqrt(v); lo, hi = float(s.min()), float(s.max() or 1.0)
    if hi == lo: return pd.Series([(out_min+out_max)/2]*len(v), index=v.index)
    return out_min + (s-lo)*(out_max-out_min)/(hi-lo)

def draw_edge(ax, p1, p2, width, color, curvature=0.12, alpha=0.9, arrow_size=12, shrink=16, z=2):
    patch = FancyArrowPatch(
        p1, p2, connectionstyle=f"arc3,rad={curvature}", arrowstyle='-|>',
        mutation_scale=arrow_size, linewidth=width, color=color, alpha=alpha,
        shrinkA=shrink, shrinkB=shrink, zorder=z, joinstyle="round", capstyle="round"
    ); ax.add_patch(patch)

def fit_positions(pos: dict, margin: float = 0.10) -> dict:
    xs = np.array([p[0] for p in pos.values()], float)
    ys = np.array([p[1] for p in pos.values()], float)
    xmin,xmax, ymin,ymax = xs.min(), xs.max(), ys.min(), ys.max()
    sx = 1.0 / max(xmax-xmin, 1e-9); sy = 1.0 / max(ymax-ymin, 1e-9)
    s = (1-2*margin)*min(sx, sy); cx, cy = (xmax+xmin)/2, (ymax+ymin)/2
    return {k: ((x-cx)*s+0.5, (y-cy)*s+0.5) for k,(x,y) in pos.items()}

def build_nodes_edges(df: pd.DataFrame, season: str|None, big_player_thr: float, edge_min_total: float):
    # 赛季筛选 + 正金额
    if season:
        key = normalize_season(season)
        df = df[df["transfer_season_norm"] == key]
    df = df[df["transfer_fee_num"].notna() & (df["transfer_fee_num"] > 0)]

    # —— 这里 groupby 时就用已经归一化好的联赛列 —— #
    sellers = df.groupby(
        ["from_club_id","club_name_from","domestic_competition_id_from"], as_index=False
    )["transfer_fee_num"].sum().rename(columns={
        "from_club_id":"club_id","club_name_from":"club_name",
        "domestic_competition_id_from":"league","transfer_fee_num":"sum_as_seller"
    })
    buyers = df.groupby(
        ["to_club_id","club_name_to","domestic_competition_id_to"], as_index=False
    )["transfer_fee_num"].sum().rename(columns={
        "to_club_id":"club_id","club_name_to":"club_name",
        "domestic_competition_id_to":"league","transfer_fee_num":"sum_as_buyer"
    })
    nodes = pd.merge(sellers, buyers, on=["club_id","club_name","league"], how="outer").fillna(0.0)
    nodes["total_volume"] = nodes["sum_as_seller"] + nodes["sum_as_buyer"]

    grouped = df.groupby(
        ["from_club_id","to_club_id","club_name_from","club_name_to",
         "domestic_competition_id_from","domestic_competition_id_to"],
        as_index=False
    ).agg(edge_weight=("transfer_fee_num","sum"),
          transfers_count=("player_name","count"))

    big = df[df["transfer_fee_num"] >= big_player_thr].copy()
    big["player_name"] = big["player_name"].astype(str).str.strip()
    big_names = (big.sort_values("transfer_fee_num", ascending=False)
                   .groupby(["from_club_id","to_club_id"])["player_name"]
                   .apply(lambda s: ", ".join(dict.fromkeys(s.tolist())))
                   .reset_index().rename(columns={"player_name":"big_player_names"}))
    edges = grouped.merge(big_names, on=["from_club_id","to_club_id"], how="left")
    edges["big_player_names"] = edges["big_player_names"].fillna("")

    if edge_min_total and edge_min_total>0:
        edges = edges[edges["edge_weight"] >= edge_min_total]

    kept = set(edges["from_club_id"]).union(set(edges["to_club_id"]))
    nodes = nodes[nodes["club_id"].isin(kept)].copy()
    return nodes, edges

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--season", default=None)
    ap.add_argument("--edge_min_total", type=float, default=2e7)
    ap.add_argument("--big_player_thr", type=float, default=3e7)
    ap.add_argument("--label_top_k", type=int, default=20)
    ap.add_argument("--nodes_top_k", type=int, default=20)
    ap.add_argument("--keep_isolates_top", action="store_true")
    ap.add_argument("--figsize", nargs=2, type=float, default=[6.5, 6.5])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--img", default="big5_network_topN_internal.png")
    args = ap.parse_args()

    # df = pd.read_csv(args.csv, low_memory=False)
    df = read_csv_flex(args.csv)

    # —— 第一步就把联赛列统一成短名（关键！）—— #
    for col in ["domestic_competition_id_from","domestic_competition_id_to"]:
        if col in df.columns:
            df[col] = df[col].map(norm_league)

    need = ["from_club_id","to_club_id","club_name_from","club_name_to",
            "domestic_competition_id_from","domestic_competition_id_to",
            "transfer_fee_num","transfer_season_norm","player_name"]
    miss = [c for c in need if c not in df.columns]
    if miss: raise KeyError(f"CSV 缺少列：{miss}")

    # 自检：现在应当只剩短名
    # print(sorted(set(df["domestic_competition_id_from"].dropna().unique())
    #            | set(df["domestic_competition_id_to"].dropna().unique())))

    nodes, edges = build_nodes_edges(df, args.season, args.big_player_thr, args.edge_min_total)
    if edges.empty or nodes.empty:
        print("⚠️ 当前阈值/赛季下没有可视化的连边。请降低 --edge_min_total 或更换赛季。")
        return

    # Top-N 内部子图
    top_ids = (nodes.sort_values("total_volume", ascending=False)
                    .head(args.nodes_top_k)["club_id"].astype(str).tolist())
    edges = edges[
        edges["from_club_id"].astype(str).isin(top_ids)
        & edges["to_club_id"].astype(str).isin(top_ids)
    ].copy()
    nodes = nodes[nodes["club_id"].astype(str).isin(top_ids)].copy()
    if not args.keep_isolates_top:
        kept = set(edges["from_club_id"].astype(str)).union(set(edges["to_club_id"].astype(str)))
        nodes = nodes[nodes["club_id"].astype(str).isin(kept)].copy()
    if edges.empty or nodes.empty:
        print("⚠️ Top-N 过滤后没有内部连边。可尝试：--edge_min_total 0 或减小 --nodes_top_k。")
        return

    # 构图
    G = nx.DiGraph()
    for _, r in nodes.iterrows():
        G.add_node(str(r["club_id"]),
                   club_name=r["club_name"],
                   league=norm_league(r["league"]),  # <—— 保证写入短名
                   total_volume=float(r["total_volume"]))
    for _, r in edges.iterrows():
        G.add_edge(str(r["from_club_id"]), str(r["to_club_id"]),
                   weight=float(r["edge_weight"]),
                   big_names=r["big_player_names"],
                   source_name=r["club_name_from"],
                   target_name=r["club_name_to"])

    pos = nx.kamada_kawai_layout(G)
    pos = fit_positions(pos, margin=args.margin)

    node_sizes = sqrt_scale(pd.Series({n: G.nodes[n]["total_volume"] for n in G.nodes}),
                            out_min=120, out_max=1200)
    node_colors = [LEAGUE_COLORS.get(G.nodes[n]["league"], "#bdbdbd") for n in G.nodes]
    edge_weights = pd.Series({(u, v): G[u][v]["weight"] for u, v in G.edges})
    edge_widths = sqrt_scale(edge_weights, out_min=0.3, out_max=2.5)

    fig, ax = plt.subplots(figsize=tuple(args.figsize), dpi=args.dpi)
    ax.axis("off"); ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0)

    for (u, v), w in edge_widths.items():
        curv = 0.18 if (v, u) in G.edges else 0.10
        draw_edge(ax, pos[u], pos[v], width=max(0.25, w*0.4),
                  color="#c7c7c7", curvature=curv, alpha=0.35,
                  arrow_size=10, shrink=14, z=1)
    for (u, v), w in edge_widths.sort_values().items():
        curv = 0.18 if (v, u) in G.edges else 0.10
        draw_edge(ax, pos[u], pos[v], width=float(w),
                  color="#111111", curvature=curv, alpha=0.9,
                  arrow_size=12, shrink=16, z=2)

    nx.draw_networkx_nodes(
        G, pos,
        node_size=[node_sizes[n] for n in G.nodes],
        node_color=node_colors,
        edgecolors="#2f2f2f", linewidths=0.8, alpha=0.98, ax=ax
    )

    # 仅标前 K 大（用短名），你也可以接着用 adjust_text 做避让
    label_top_k = min(args.label_top_k, len(G.nodes))
    top_nodes = (pd.Series({n: G.nodes[n]["total_volume"] for n in G.nodes})
                    .sort_values(ascending=False).head(label_top_k).index)
    size_scale = {n: max(5.0, min(8.0, float(node_sizes[n]) / 300.0)) for n in top_nodes}
    texts = []
    for n in top_nodes:
        x, y = pos[n]
        t = ax.text(x, y,
                    f'{G.nodes[n]["club_name"]}_{G.nodes[n]["league"]}',  # <—— 标签也用短名
                    fontsize=size_scale[n], ha="center", va="center",
                    zorder=5, color="#111111")
        t.set_path_effects([pe.withStroke(linewidth=1.2, foreground="white")])
        texts.append(t)
    # adjust_text(texts)  # 如需自动避让，安装 adjustText 后再打开

    # 大额转会球员名
    for (u, v) in G.edges:
        text = G[u][v]["big_names"]
        if not text: continue
        x1, y1 = pos[u]; x2, y2 = pos[v]
        xm, ym = (x1+x2)/2.0, (y1+y2)/2.0
        dx, dy = (y2-y1), -(x2-x1); nrm = max(np.hypot(dx, dy), 1e-6)
        xm += 0.02 * dx / nrm; ym += 0.02 * dy / nrm
        t = ax.text(xm, ym, text, fontsize=5.5, color="#111111",
                    ha="center", va="center", zorder=4)
        t.set_path_effects([pe.withStroke(linewidth=1.0, foreground="white")])

    out = Path(args.img)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.2)
    print(f"✅ Saved: {out.resolve()}")

if __name__ == "__main__":
    main()
