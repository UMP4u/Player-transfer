#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make a Fig.1-style transfer network (pure Python, no Gephi).

Input
  transfers_big5_both.csv  # 你前一步脚本生成的文件（双方都在五大）

What it does (matches the paper's Fig.1):
  - 只保留“当季总交易额 >= NODE_MIN_TOTAL(默认€70m)”的俱乐部作为节点
  - 只保留两端都在上述集合内的边；边宽 ∝ 俱乐部对累计金额
  - 仅在单笔金额 >= BIG_PLAYER_THR(默认€30m) 的边上显示球员名字
  - 节点颜色=联赛（英/西/意/德/法），大小=总交易额（√缩放）
  - Kamada–Kawai 布局 + 浅灰背景边 + 黑色主边 + 清晰箭头
  - 自动隐藏孤立节点；只给“前 K 个关键俱乐部”贴标签（按总额+度数综合）

Usage example
  python fig1_style_network.py \
    --csv ./out_big5/transfers_big5_both.csv \
    --season 2014/15 \
    --img fig_big5_fig1.svg \
    --node_min_total 70000000 \
    --big_player_thr 30000000 \
    --label_top_k 20
"""

import argparse
from pathlib import Path
import re, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import networkx as nx
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

# 五大联赛配色（可改）
LEAGUE_COLORS = {
    "GB1": "#f28e2b",  # EPL
    "ES1": "#ffdf5d",  # LaLiga
    "IT1": "#4e79a7",  # Serie A
    "L1" : "#59a14f",  # Bundesliga
    "FR1": "#e15759",  # Ligue 1
}

# 论文里用到的两个阈值（默认与原图一致）
DEFAULT_NODE_MIN = 70_000_000     # 节点：当季总交易额阈值
DEFAULT_BIG_THR   = 30_000_000     # 单笔：标注人名阈值

STOPWORDS = [
    r"Football Club", r"Club de F[úu]tbol", r"S\.A\.D\.", r"Association Sportive ",
    r"Turn- und Sportverein", r"Verein f[üu]r Bewegungsspiele", r"Societ[aà] Sportiva",
    r"Associazione Calcio", r" CF ", r" FC ", r" S\.p\.A\."
]

def normalize_season(s):
    if not isinstance(s, str): return s
    t = s.strip().replace(" ", "").replace("-", "/")
    a = t.split("/")
    if len(a)==2 and a[0].isdigit() and a[1].isdigit():
        y0 = int(a[0]) if len(a[0])==4 else 2000+int(a[0])
        y1 = int(a[1])
        return f"{y0}/{y1:02d}"
    return s

def short_name(name: str, league: str, max_chars=18, wrap=True):
    s = name
    for sw in STOPWORDS:
        s = re.sub(sw, "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s{2,}", " ", s)
    s2 = f"{s}_{league}"
    if wrap:
        s2 = "\n".join(textwrap.wrap(s2, width=max_chars//2 or 1))
    else:
        if len(s2) > max_chars:
            s2 = s2[:max_chars-1] + "…"
    return s2

def sqrt_scale(values: pd.Series, out_min: float, out_max: float) -> pd.Series:
    v = values.astype(float).clip(lower=0)
    s = np.sqrt(v)
    lo, hi = float(s.min()), float(s.max() or 1.0)
    if hi == lo:
        return pd.Series([(out_min+out_max)/2]*len(v), index=v.index)
    return out_min + (s-lo)*(out_max-out_min)/(hi-lo)

def draw_edge(ax, p1, p2, width, color, curvature=0.18, alpha=0.9, arrow_size=11):
    """弧形箭头边（卖→买）"""
    patch = FancyArrowPatch(
        p1, p2,
        connectionstyle=f"arc3,rad={curvature}",
        arrowstyle='-|>',
        mutation_scale=arrow_size,
        linewidth=width,
        color=color,
        alpha=alpha,
        shrinkA=10, shrinkB=10,
    )
    ax.add_patch(patch)

def build_nodes_edges(df: pd.DataFrame, season: str|None,
                      big_player_thr: float):
    if season:
        key = normalize_season(season)
        df = df[df["transfer_season_norm"] == key]
    df = df[df["transfer_fee_num"].notna() & (df["transfer_fee_num"] > 0)]

    # 节点：入+出金额
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

    # 边：按俱乐部对聚合
    grouped = df.groupby(
        ["from_club_id","to_club_id","club_name_from","club_name_to",
         "domestic_competition_id_from","domestic_competition_id_to"],
        as_index=False
    ).agg(edge_weight=("transfer_fee_num","sum"),
          transfers_count=("player_name","count"))

    # ≥阈值的人名
    big = df[df["transfer_fee_num"] >= big_player_thr].copy()
    big["player_name"] = big["player_name"].astype(str).str.strip()
    big_names = (big.sort_values("transfer_fee_num", ascending=False)
                   .groupby(["from_club_id","to_club_id"])["player_name"]
                   .apply(lambda s: ", ".join(dict.fromkeys(s.tolist())))
                   .reset_index()
                   .rename(columns={"player_name":"big_player_names"}))
    edges = grouped.merge(big_names, on=["from_club_id","to_club_id"], how="left")
    edges["big_player_names"] = edges["big_player_names"].fillna("")
    return nodes, edges

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="transfers_big5_both.csv")
    ap.add_argument("--season", default=None, help="如 2014/15；不填则全时段（不建议）")
    ap.add_argument("--node_min_total", type=float, default=DEFAULT_NODE_MIN,
                    help="节点总交易额阈值（默认 7000万欧）")
    ap.add_argument("--big_player_thr", type=float, default=DEFAULT_BIG_THR,
                    help="仅在单笔 ≥ 此阈值的边上显示球员名（默认 3000万欧）")
    ap.add_argument("--label_top_k", type=int, default=20,
                    help="只给前K个关键节点贴标签（默认 20）")
    ap.add_argument("--img", default="fig_big5_fig1.svg",
                    help="输出文件名（.svg/.png）")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    need = ["from_club_id","to_club_id","club_name_from","club_name_to",
            "domestic_competition_id_from","domestic_competition_id_to",
            "transfer_fee_num","transfer_season_norm","player_name"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"CSV 缺少列：{miss}")

    nodes, edges = build_nodes_edges(df, args.season, args.big_player_thr)

    # —— Fig.1 关键：只保留“当季总额 ≥ 指定阈值”的俱乐部 —— #
    nodes = nodes[nodes["total_volume"] >= args.node_min_total].copy()
    keep = set(nodes["club_id"].astype(str))
    edges = edges[
        edges["from_club_id"].astype(str).isin(keep) &
        edges["to_club_id"].astype(str).isin(keep)
    ].copy()

    if nodes.empty or edges.empty:
        print("⚠️ 当前赛季达不到阈值，尝试降低 --node_min_total 或更换赛季。")
        return

    # —— 建图 —— #
    G = nx.DiGraph()
    for _, r in nodes.iterrows():
        G.add_node(str(r["club_id"]),
                   club_name=r["club_name"],
                   league=r["league"],
                   total_volume=float(r["total_volume"]))
    for _, r in edges.iterrows():
        G.add_edge(str(r["from_club_id"]), str(r["to_club_id"]),
                   weight=float(r["edge_weight"]),
                   big_names=r["big_player_names"])

    # —— 布局（Kamada–Kawai）并整体收紧 —— #
    pos = nx.kamada_kawai_layout(G)
    for k in pos: pos[k] = pos[k] * 0.90

    # —— 映射尺寸/颜色 —— #
    vol = pd.Series({n: G.nodes[n]["total_volume"] for n in G.nodes})
    node_sizes = sqrt_scale(vol, out_min=230, out_max=2600)
    node_colors = [LEAGUE_COLORS.get(G.nodes[n]["league"], "#bdbdbd") for n in G.nodes]
    edge_weights = pd.Series({(u, v): G[u][v]["weight"] for u, v in G.edges})
    edge_widths = sqrt_scale(edge_weights, out_min=0.8, out_max=9.5)

    # —— 绘图（论文风） —— #
    import matplotlib
    matplotlib.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(9, 9), dpi=300)
    ax.axis("off")

    # 背景边：浅灰，弱
    for (u, v), w in edge_widths.items():
        curv = 0.18 if (v, u) in G.edges else 0.0
        draw_edge(ax, pos[u], pos[v], width=max(0.6, w*0.30),
                  color="#bdbdbd", curvature=curv, alpha=0.22, arrow_size=7)

    # 主边：黑，粗
    for (u, v), w in edge_widths.sort_values().items():
        curv = 0.18 if (v, u) in G.edges else 0.0
        draw_edge(ax, pos[u], pos[v], width=float(w*1.10),
                  color="#111111", curvature=curv, alpha=0.95, arrow_size=12)

    # 节点
    nx.draw_networkx_nodes(
        G, pos,
        node_size=[node_sizes[n] for n in G.nodes],
        node_color=node_colors,
        edgecolors="#2f2f2f", linewidths=0.9, alpha=0.98, ax=ax
    )

    # 只给“前K个关键俱乐部”贴标签（总额+度数综合）
    score = (vol.rank(pct=True)*0.7 + pd.Series(dict(G.degree())).rank(pct=True)*0.3)
    top_nodes = score.sort_values(ascending=False).head(args.label_top_k).index
    label_dict = {n: short_name(G.nodes[n]['club_name'], G.nodes[n]['league'], max_chars=18, wrap=True)
                  for n in top_nodes}
    txts = nx.draw_networkx_labels(G, pos, labels=label_dict, font_size=6.2, ax=ax)
    for t in txts.values():
        t.set_path_effects([pe.withStroke(linewidth=1.2, foreground="white")])

    # 边上仅标“≥ big_player_thr”的人名；只给最粗的15%边标，避免拥挤
    cut = edge_widths.quantile(0.85)
    for (u, v), w in edge_widths.items():
        text = G[u][v]["big_names"]
        if not text or w < cut: 
            continue
        x1, y1 = pos[u]; x2, y2 = pos[v]
        xm, ym = (x1+x2)/2.0, (y1+y2)/2.0
        dx, dy = (y2-y1), -(x2-x1)
        nrm = max(np.hypot(dx, dy), 1e-6)
        xm += 0.02*dx/nrm; ym += 0.02*dy/nrm
        t = ax.text(xm, ym, text, fontsize=6.0, color="#111111", ha="center", va="center")
        t.set_path_effects([pe.withStroke(linewidth=1.1, foreground="white")])

    # 联赛图例
    legend_items = [("GB1","Premier League"),("ES1","La Liga"),
                    ("IT1","Serie A"),("L1","Bundesliga"),("FR1","Ligue 1")]
    handles = [Line2D([0],[0], marker='o', color='w',
                      markerfacecolor=LEAGUE_COLORS[k], markeredgecolor="#2f2f2f",
                      markersize=6.5, label=lab) for k, lab in legend_items]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=7)

    # 保存
    out = Path(args.img)
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"✅ Saved: {out.resolve()}")

if __name__ == "__main__":
    main()



