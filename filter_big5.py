#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
 基本用法：(1)筛出涉及五大的全部转会 & 五大内部互转
python filter_big5.py \
  --data_dir /path/to/player-scores \
  --out_dir ./out_big5
# 2) 只看某个赛季（例如 2014/15）
python filter_big5.py \
  --data_dir /path/to/player-scores \
  --out_dir ./out_big5_2014 \
  --season 2014/15

# 3) 叠加最小转会费阈值（例如 ≥ 100 万欧）
python filter_big5.py \
  --data_dir /path/to/player-scores \
  --out_dir ./out_big5_min1m \
  --min_fee 1000000
'''

"""
Filter Big-5 league clubs and transfers from `davidcariboo/player-scores` dataset.

Inputs (in one folder):
- competitions.csv
- clubs.csv
- transfers.csv

Outputs (in --out_dir):
- big5_clubs.csv                 # 五大联赛俱乐部清单（含联赛代码与名称）
- transfers_with_leagues.csv     # 所有转会 + 双方联赛标注（方便你自定义过滤）
- transfers_big5_either.csv      # 至少一方在五大联赛的转会（默认）
- transfers_big5_both.csv        # 双方都在五大联赛的转会（五大内部互转）
"""

import argparse
import os
import sys
import pandas as pd
from pathlib import Path
from typing import Tuple, Set

# —— 常量：五大联赛 competition_id（Transfermarkt/Kaggle 数据常用编码）——
# 英超：GB1；西甲：ES1；意甲：IT1；德甲：L1；法甲：FR1
BIG5_IDS: Set[str] = {"GB1", "ES1", "IT1", "L1", "FR1"}

def normalize_season(s: str) -> str:
    """把 26/27、2014/15、2014-2015 等统一成 YYYY/YY 形式。无则返回原值。"""
    if not isinstance(s, str):
        return s
    t = s.strip().replace(" ", "").replace("-", "/")
    parts = t.split("/")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        y0 = int(parts[0]) if len(parts[0]) == 4 else (2000 + int(parts[0]))
        y1 = int(parts[1])
        return f"{y0}/{y1:02d}"
    return s

def derive_season_from_date(date_str: str) -> str:
    """若缺少 transfer_season，可根据 transfer_date（YYYY-M-D）推断赛季：
    规则：每年 7 月到次年 6 月为一个赛季。"""
    if not isinstance(date_str, str) or not date_str:
        return None
    try:
        y, m, *_ = [int(x) for x in date_str.replace(".", "-").replace("/", "-").split("-")]
        if m >= 7:
            y0 = y
        else:
            y0 = y - 1
        return f"{y0}/{(y0+1)%100:02d}"
    except Exception:
        return None

def load_tables(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comps = pd.read_csv(data_dir / "competitions.csv", low_memory=False)
    clubs = pd.read_csv(data_dir / "clubs.csv", low_memory=False)
    transfers = pd.read_csv(data_dir / "transfers.csv", low_memory=False)
    return comps, clubs, transfers

def build_big5_club_table(clubs: pd.DataFrame, comps: pd.DataFrame) -> pd.DataFrame:
    # 只要俱乐部的国内联赛属于 BIG5 即视为五大俱乐部
    cols_needed = ["club_id", "name", "domestic_competition_id"]
    for c in cols_needed:
        if c not in clubs.columns:
            raise KeyError(f"clubs.csv 缺少列：{c}")
    c = clubs[cols_needed].rename(columns={"name": "club_name"})
    c["is_big5"] = c["domestic_competition_id"].isin(BIG5_IDS)
    big5 = c[c["is_big5"]].copy()

    # 附上赛事中文名/英文名，便于核对
    comp_lookup = comps[["competition_id", "name"]].drop_duplicates()
    comp_lookup = comp_lookup.rename(columns={"name": "competition_name"})
    big5 = big5.merge(comp_lookup, left_on="domestic_competition_id",
                      right_on="competition_id", how="left")
    big5 = big5.drop(columns=["competition_id"])
    return big5

def annotate_transfer_leagues(transfers: pd.DataFrame, big5: pd.DataFrame) -> pd.DataFrame:
    # 校验列
    need_cols = ["from_club_id", "to_club_id", "transfer_fee", "transfer_date", "transfer_season"]
    for c in need_cols:
        if c not in transfers.columns:
            raise KeyError(f"transfers.csv 缺少列：{c}")

    # 赛季标准化（便于你后续按赛季过滤）
    if "transfer_season" in transfers.columns:
        transfers["transfer_season_norm"] = transfers["transfer_season"].apply(normalize_season)
    else:
        transfers["transfer_season_norm"] = None

    # 若没有 season 或有缺失，用日期推导一个（可选）
    mask_missing = transfers["transfer_season_norm"].isna()
    if "transfer_date" in transfers.columns:
        transfers.loc[mask_missing, "transfer_season_norm"] = transfers.loc[mask_missing, "transfer_date"].apply(derive_season_from_date)

    # 用 big5 表给 from/to 两侧打联赛标记（如果不是五大，字段为空）
    small = big5[["club_id", "club_name", "domestic_competition_id"]].copy()

    left = transfers.merge(
        small.add_suffix("_from"),
        left_on="from_club_id", right_on="club_id_from", how="left"
    ).drop(columns=["club_id_from"])

    both = left.merge(
        small.add_suffix("_to"),
        left_on="to_club_id", right_on="club_id_to", how="left"
    ).drop(columns=["club_id_to"])

    # 语义字段：是否在五大
    both["from_is_big5"] = both["domestic_competition_id_from"].isin(BIG5_IDS)
    both["to_is_big5"]   = both["domestic_competition_id_to"].isin(BIG5_IDS)

    return both

def main():
    ap = argparse.ArgumentParser(description="Filter Big-5 league clubs & transfers")
    ap.add_argument("--data_dir", required=True, help="folder containing competitions.csv / clubs.csv / transfers.csv")
    ap.add_argument("--out_dir",  required=True, help="output folder")
    ap.add_argument("--season",   default=None, help="optional season filter, e.g. 2014/15 or 14/15")
    ap.add_argument("--min_fee",  type=float, default=None, help="optional minimum transfer fee (EUR) to keep")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comps, clubs, transfers = load_tables(data_dir)
    big5 = build_big5_club_table(clubs, comps)
    big5.to_csv(out_dir / "big5_clubs.csv", index=False)

    tf = annotate_transfer_leagues(transfers, big5)

    # 金额清洗（转为数值）
    def to_num(x):
        try:
            return float(x)
        except Exception:
            return None
    tf["transfer_fee_num"] = tf["transfer_fee"].apply(to_num)

    # 赛季过滤（可选）
    if args.season:
        season_key = normalize_season(args.season)
        tf = tf[tf["transfer_season_norm"] == season_key]

    # 最小费用过滤（可选）
    if args.min_fee is not None:
        tf = tf[(tf["transfer_fee_num"].notna()) & (tf["transfer_fee_num"] >= args.min_fee)]

    # 输出 1：附带联赛注释的完整表（方便你自己再筛）
    tf.to_csv(out_dir / "transfers_with_leagues.csv", index=False)

    # 输出 2：至少一方在五大的子集
    either = tf[ tf["from_is_big5"] | tf["to_is_big5"] ].copy()
    either.to_csv(out_dir / "transfers_big5_either.csv", index=False)

    # 输出 3：双方都在五大的子集（五大内部互转）
    both  = tf[ tf["from_is_big5"] & tf["to_is_big5"] ].copy()
    both.to_csv(out_dir / "transfers_big5_both.csv", index=False)

    # 简要汇总
    print("✅ Done.")
    print(f"- Big-5 clubs: {len(big5)} rows -> {out_dir/'big5_clubs.csv'}")
    print(f"- Transfers (annotated): {len(tf)} rows -> {out_dir/'transfers_with_leagues.csv'}")
    print(f"- Big-5 (either side): {len(either)} rows -> {out_dir/'transfers_big5_either.csv'}")
    print(f"- Big-5 (both sides) : {len(both)} rows -> {out_dir/'transfers_big5_both.csv'}")

if __name__ == "__main__":
    pd.set_option("display.max_columns", 200)
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
