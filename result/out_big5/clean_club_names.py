#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean club names in transfers CSV.

功能：
- 去掉俱乐部名字里的 "Football Club"（不区分大小写）
- 压缩多余空格
- 可以扩展：去掉 "FC", "CF", "Club de Fútbol", "Società Sportiva" 等前后缀
- 输出一个新的 CSV 文件供绘图使用
"""

import re
import argparse
import pandas as pd

def clean_name(name: str) -> str:
    """清理俱乐部名字"""
    if not isinstance(name, str):
        return name

    # 去掉 "Football Club"
    name = re.sub(r"\bfootball club\b", "", name, flags=re.I)

    # 可选扩展：去掉常见缩写
    name = re.sub(r"\bF\.?C\.?\b", "", name, flags=re.I)        # FC
    name = re.sub(r"\bC\.?F\.?\b", "", name, flags=re.I)        # CF
    name = re.sub(r"Club de Fútbol", "", name, flags=re.I)      # 西语
    name = re.sub(r"Società Sportiva", "", name, flags=re.I)    # 意语

    # 压缩多余空格
    name = re.sub(r"\s{2,}", " ", name).strip()
    return name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="原始 transfers_big5_both.csv 文件路径")
    ap.add_argument("--out", default="transfers_big5_both_clean.csv",
                    help="输出干净版 CSV 文件名（默认：transfers_big5_both_clean.csv）")
    args = ap.parse_args()

    # 读数据
    df = pd.read_csv(args.csv, low_memory=False)

    # 清理俱乐部名字列
    for col in ["club_name_from", "club_name_to"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_name)

    # 保存新文件
    df.to_csv(args.out, index=False)
    print(f"✅ Cleaned CSV saved to {args.out}")

if __name__ == "__main__":
    main()
