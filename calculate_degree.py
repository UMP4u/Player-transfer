#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_club_degree.py

从 Kaggle: davidcariboo/player-scores 的 transfers.csv（或同结构数据）计算：
- 每一年，每个俱乐部的 in_degree / out_degree / degree
- 过滤掉俱乐部名包含 ["Retired", "Without Club", "Unknown"] 的记录
- 为 (year, club_id) 选一个代表名称（当年出现次数最多；并列取字母序最小）

用法示例：
    python compute_club_degree.py \
        --csv fc94b8f0-56f7-4c13-838d-adbbb4fc1bed.csv \
        --out_degree club_degree_by_year.csv \
        --out_clean transfers_clean.csv
"""

import argparse
from collections import Counter
from typing import List, Optional

import numpy as np
import pandas as pd


def pick_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    """从若干候选列名中挑选存在于 df.columns 的那个。"""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"缺少必要列：任一 {candidates}")
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute per-year club degree from transfers CSV")
    p.add_argument("--csv", required=True, help="输入 CSV（转会明细）路径")
    p.add_argument("--out_degree", default="club_degree_by_year.csv", help="输出的度统计 CSV")
    p.add_argument("--out_clean", default="transfers_clean.csv", help="输出的清洗后明细 CSV")
    p.add_argument("--blacklist",
                   default="Retired,Without Club,Unknown",
                   help="需要过滤掉的俱乐部名关键字（逗号分隔，大小写不敏感，子串匹配）")
    p.add_argument("--year_min", type=int, default=None, help="仅保留 >= year_min 的记录")
    p.add_argument("--year_max", type=int, default=None, help="仅保留 <= year_max 的记录")
    return p


def main():
    args = build_parser().parse_args()

    # 读取
    df = pd.read_csv(args.csv, low_memory=False)

    # 列名适配（按你给的信息：from_club_name / to_club_name / transfer_date）
    col_transfer_date = pick_col(df, ["transfer_date", "date", "Transfer_date"])
    col_from_name     = pick_col(df, ["from_club_name", "from_club", "from", "from_team_name"])
    col_to_name       = pick_col(df, ["to_club_name", "to_club", "to", "to_team_name"])
    col_from_id       = pick_col(df, ["from_club_id", "from_id", "from_team_id"])
    col_to_id         = pick_col(df, ["to_club_id", "to_id", "to_team_id"])

    # 提取年份
    df["year"] = pd.to_datetime(df[col_transfer_date], errors="coerce").dt.year.astype("Int64")

    # 年份范围过滤（如果指定）
    if args.year_min is not None:
        df = df[df["year"].notna() & (df["year"] >= args.year_min)]
    if args.year_max is not None:
        df = df[df["year"].notna() & (df["year"] <= args.year_max)]

    # 黑名单过滤（名字包含关键字就剔除；大小写不敏感，包含即可）
    black_tokens = [t.strip().casefold() for t in args.blacklist.split(",") if t.strip()]

    def contains_black(name) -> bool:
        if pd.isna(name):
            return False
        s = str(name).casefold()
        return any(tok in s for tok in black_tokens)

    bad_mask = df[col_from_name].apply(contains_black) | df[col_to_name].apply(contains_black)
    df_clean = df.loc[~bad_mask].copy()

    # -------------------------------
    # 统计 in/out 度（按 ID 作为主键）
    # -------------------------------
    # in-degree: 以 to 侧 club_id 累计
    to_df = df_clean[[col_to_id, col_to_name, "year"]].dropna(subset=[col_to_id, "year"]).copy()
    to_df.rename(columns={col_to_id: "club_id", col_to_name: "club_name"}, inplace=True)
    to_df["in_degree"] = 1

    # out-degree: 以 from 侧 club_id 累计
    from_df = df_clean[[col_from_id, col_from_name, "year"]].dropna(subset=[col_from_id, "year"]).copy()
    from_df.rename(columns={col_from_id: "club_id", col_from_name: "club_name"}, inplace=True)
    from_df["out_degree"] = 1

    # 聚合
    in_deg = to_df.groupby(["year", "club_id"], dropna=False)["in_degree"].sum().reset_index()
    out_deg = from_df.groupby(["year", "club_id"], dropna=False)["out_degree"].sum().reset_index()

    deg = pd.merge(in_deg, out_deg, on=["year", "club_id"], how="outer")
    deg["in_degree"] = deg["in_degree"].fillna(0).astype(int)
    deg["out_degree"] = deg["out_degree"].fillna(0).astype(int)
    deg["degree"] = deg["in_degree"] + deg["out_degree"]

    # -------------------------------
    # 代表名称：该年该 club_id 出现次数最多的名称（并列取字母序最小）
    # -------------------------------
    name_candidates = pd.concat(
        [to_df[["year", "club_id", "club_name"]],
         from_df[["year", "club_id", "club_name"]]],
        ignore_index=True
    ).dropna(subset=["club_name"])

    def most_common_name(group: pd.DataFrame) -> str:
        cnt = Counter(map(str, group["club_name"]))
        if not cnt:
            return np.nan
        max_freq = max(cnt.values())
        candidates = sorted([name for name, f in cnt.items() if f == max_freq])
        return candidates[0]  # 字母序最小

    rep_names = name_candidates.groupby(["year", "club_id"]).apply(most_common_name).reset_index(name="club_name")
    deg = pd.merge(deg, rep_names, on=["year", "club_id"], how="left")

    # 排序与列顺序
    deg = deg.sort_values(["year", "degree"], ascending=[True, False])
    deg = deg[["year", "club_id", "club_name", "in_degree", "out_degree", "degree"]]

    # 导出
    deg.to_csv(args.out_degree, index=False, encoding="utf-8-sig")
    df_clean.to_csv(args.out_clean, index=False, encoding="utf-8-sig")

    # 控制台提示
    print(f"✅ Done. Wrote:\n  - degrees: {args.out_degree}\n  - cleaned: {args.out_clean}")
    # 打印前几行预览
    with pd.option_context("display.max_rows", 20, "display.max_columns", None, "display.width", 120):
        print(deg.head(20))


if __name__ == "__main__":
    main()
