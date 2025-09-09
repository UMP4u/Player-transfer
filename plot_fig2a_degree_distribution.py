#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ------------------------------ 数据准备 ------------------------------
def load_transfers(csv_path, year_min=None, year_max=None,
                   col_from="from_club_name", col_to="to_club_name", col_date="transfer_date"):
    usecols = [col_from, col_to, col_date]
    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)

    # 解析日期并可选按年份过滤
    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
    df = df.dropna(subset=[col_from, col_to, col_date])

    if year_min is not None:
        df = df[df[col_date].dt.year >= int(year_min)]
    if year_max is not None:
        df = df[df[col_date].dt.year <= int(year_max)]

    # 删除自环
    df = df[df[col_from] != df[col_to]]

    # 删除非俱乐部标签
    invalid = {"Retired", "Without Club","Unknown"}
    df = df[~df[col_from].isin(invalid)]
    df = df[~df[col_to].isin(invalid)]

    # 去掉首尾空白
    df[col_from] = df[col_from].astype(str).str.strip()
    df[col_to]   = df[col_to].astype(str).str.strip()
    return df

def build_digraph_unique_edges(df, col_from="from_club_name", col_to="to_club_name"):
    unique_edges = df[[col_from, col_to]].drop_duplicates().values.tolist()
    G = nx.DiGraph()
    G.add_nodes_from(pd.unique(df[[col_from, col_to]].values.ravel()))
    G.add_edges_from(unique_edges)
    return G

def degree_pmf(G, mode="out"):
    degs = dict(G.out_degree()) if mode == "out" else dict(G.in_degree())
    k_vals = np.array(list(degs.values()), dtype=int)
    vc = pd.Series(k_vals).value_counts().sort_index()
    pmf = vc / len(k_vals)
    # 丢掉 p(k)=0 的（不会有），并确保 k>=1（论文横轴从 10^0 开始）
    pmf = pmf[pmf.index >= 1]
    return pmf

# ------------------------------ 拟合工具 ------------------------------
def fit_powerlaw_ls(k, p):
    # 拟合 log p = a + b * log k, 返回 (A, gamma) with p ≈ A * k^{-(gamma+1)}
    x = np.log(k)
    y = np.log(p)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    A0 = np.exp(a)
    gamma = -b - 1.0
    y_hat = a + b * x
    rss = np.sum((y - y_hat) ** 2)
    return A0, gamma, rss

def fit_exponential_ls(k, p):
    # 拟合 log p = c - beta * k, 返回 (C, beta) with p ≈ C * exp(-beta k)
    x = k.astype(float)
    y = np.log(p)
    A = np.vstack([np.ones_like(x), -x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    c, beta = coef
    C0 = np.exp(c)
    y_hat = c - beta * x
    rss = np.sum((y - y_hat) ** 2)
    return C0, beta, rss

def search_k0_and_fit(pmf, k0_min=3, k0_max=None, min_points_each_side=5):
    """
    在离散 (k, p) 上搜索分界点 k0（左幂律、右指数），用对数域最小二乘的总残差最小准则。
    """
    ks = pmf.index.values.astype(int)
    ps = pmf.values.astype(float)
    if k0_max is None:
        k0_max = ks.max() - 1
    best = None

    for k0 in range(max(k0_min, ks.min()+1), min(k0_max, ks.max()-1)+1):
        left_mask  = ks <= k0
        right_mask = ks >= k0  # 注意：两侧都包含 k0，可视作拼接点

        if left_mask.sum() < min_points_each_side or right_mask.sum() < min_points_each_side:
            continue

        A0, gamma, rss_L = fit_powerlaw_ls(ks[left_mask], ps[left_mask])
        C0, beta,  rss_R = fit_exponential_ls(ks[right_mask], ps[right_mask])
        rss_total = rss_L + rss_R

        if (best is None) or (rss_total < best["rss"]):
            best = dict(k0=k0, A=A0, gamma=gamma, C=C0, beta=beta,
                        rss=rss_total, rss_L=rss_L, rss_R=rss_R)

    return best

# ------------------------------ 画图 ------------------------------
def plot_scatter_with_fit(pmf_out, pmf_in, fit_out, fit_in,
                          out_path="fig2a_scatter_fit.png", title=None):
    plt.figure(figsize=(4.8, 3.6), dpi=300)

    # 空心散点
    plt.loglog(pmf_out.index, pmf_out.values, marker='o', mfc='none',
               mec='blue', mew=0.9, linestyle='None', label=r"$k^{out}$")
    plt.loglog(pmf_in.index,  pmf_in.values,  marker='s', mfc='none',
               mec='red',  mew=0.9, linestyle='None', label=r"$k^{in}$")

    # 拟合曲线：用平滑的 k 采样画两段
    def draw_piecewise(fit, color, lw=1.1):
        if fit is None:
            return
        k0 = fit["k0"]
        # 左段幂律
        kL = np.linspace(1, k0, int(max(50, k0)), endpoint=True)
        pL = fit["A"] * (kL ** (-(fit["gamma"] + 1.0)))
        plt.loglog(kL, pL, color=color, lw=lw)
        # 右段指数
        kR = np.linspace(k0, max(pmf_out.index.max(), pmf_in.index.max()), 200, endpoint=True)
        pR = fit["C"] * np.exp(-fit["beta"] * kR)
        plt.loglog(kR, pR, color=color, lw=lw, linestyle='--')

    draw_piecewise(fit_out, color='green')
    draw_piecewise(fit_in,  color='magenta')

    plt.xlabel(r"$k$")
    plt.ylabel(r"$p$")
    if title:
        plt.title(title)
    # 简洁的网格与图例
    plt.grid(True, which="both", ls=":", alpha=0.35)
    plt.legend(frameon=False)
    
    # 手动设置显示范围，避免留白
    plt.xlim(1, max(pmf_out.index.max(), pmf_in.index.max()) * 1.05)
    plt.ylim(1e-5, 1)   # 纵轴范围固定为 [10^-5, 1]


    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"[OK] Saved figure: {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Scatter + bimodal fit for Fig.2(a)-style degree distributions")
    ap.add_argument("--csv", required=True, help="Path to transfers.csv (Kaggle player-scores)")
    ap.add_argument("--year_min", type=int, default=None)
    ap.add_argument("--year_max", type=int, default=None)
    ap.add_argument("--col_from", default="from_club_name")
    ap.add_argument("--col_to",   default="to_club_name")
    ap.add_argument("--col_date", default="transfer_date")
    ap.add_argument("--out", default="fig2a_scatter_fit.png")
    args = ap.parse_args()

    df = load_transfers(args.csv, args.year_min, args.year_max,
                        args.col_from, args.col_to, args.col_date)
    print(f"[INFO] Rows after filtering: {len(df):,}")

    G = build_digraph_unique_edges(df, args.col_from, args.col_to)
    print(f"[INFO] Nodes: {G.number_of_nodes():,}  Edges (unique pairs): {G.number_of_edges():,}")

    pmf_out = degree_pmf(G, mode="out")
    pmf_in  = degree_pmf(G, mode="in")

    # 搜索分界点并拟合
    fit_out = search_k0_and_fit(pmf_out, k0_min=3, min_points_each_side=5)
    fit_in  = search_k0_and_fit(pmf_in,  k0_min=3, min_points_each_side=5)

    # 打印结果
    def print_fit(name, fit):
        if fit is None:
            print(f"[WARN] {name}: 拟合失败（可尝试调大数据量或放宽 k0 搜索范围）")
            return
        print(f"[FIT] {name}: k0={fit['k0']} | gamma≈{fit['gamma']:.3f} | beta≈{fit['beta']:.3f} "
              f"| RSS={fit['rss']:.3g} (L={fit['rss_L']:.3g}, R={fit['rss_R']:.3g})")

    print_fit("k_out", fit_out)
    print_fit("k_in",  fit_in)

    # 标题（可选）
    title = None
    if args.year_min or args.year_max:
        ymin = args.year_min if args.year_min is not None else int(df["transfer_date"].dt.year.min())
        ymax = args.year_max if args.year_max is not None else int(df["transfer_date"].dt.year.max())
        title = f"Degree distributions ({ymin}–{ymax})"

    plot_scatter_with_fit(pmf_out, pmf_in, fit_out, fit_in, out_path=args.out, title=title)

if __name__ == "__main__":
    main()
