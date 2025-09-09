import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# ======== 修改区：你的CSV路径 ========
CSV_PATH = "D:\\Desktop\\workspace\\workspace\\football\\out_big5\\transfers_with_leagues.csv"   # 改成你的文件路径
OUT_PNG  = "transfers_nonbig5_to_big5_vs_total.png"

# 年份区间限制
YEAR_MIN = 2000
YEAR_MAX = 2024

# （可选）过滤特殊俱乐部
DROP_SPECIAL_CLUBS = True
FROM_CLUB_COLS = ["from_club_name", "from_club", "from_team"]
TO_CLUB_COLS   = ["to_club_name", "to_club", "to_team"]
SPECIAL_CLUB_KEYWORDS = {"retired", "without club", "unknown"}

# ======== 列名自适应配置 ========
POSSIBLE_YEAR_COLS = ["year"]
POSSIBLE_DATE_COLS = ["transfer_date", "date"]
POSSIBLE_SEASON_COLS = ["season", "Season"]

POSSIBLE_FROM_BIG5_COLS = [
    "from_big5", "from_is_big5", "from_league_is_big5",
]
POSSIBLE_TO_BIG5_COLS = [
    "to_big5", "to_is_big5", "to_league_is_big5",
]

def coerce_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    s_str = s.astype(str).str.strip().str.lower()
    true_set  = {"1", "true", "t", "yes", "y"}
    false_set = {"0", "false", "f", "no", "n"}
    out = pd.Series(index=s.index, dtype="boolean")
    out[s_str.isin(true_set)] = True
    out[s_str.isin(false_set)] = False
    return out.astype("boolean")

def pick_first_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def season_to_year(s):
    if pd.isna(s): return np.nan
    s = str(s)
    m = re.search(r"(\d{4})\s*[/\-]", s)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d{4})", s)
    if m2:
        return int(m2.group(1))
    return np.nan

def date_to_year(s):
    if pd.isna(s): return np.nan
    try:
        return pd.to_datetime(s, errors="coerce").year
    except Exception:
        return np.nan

def main():
    df = pd.read_csv(CSV_PATH)

    # ==== 年份列推断 ====
    year_col = pick_first_col(df, POSSIBLE_YEAR_COLS)
    if year_col is not None:
        years = pd.to_numeric(df[year_col], errors="coerce")
    else:
        date_col = pick_first_col(df, POSSIBLE_DATE_COLS)
        season_col = pick_first_col(df, POSSIBLE_SEASON_COLS)
        years = None
        if date_col is not None:
            years = df[date_col].map(date_to_year)
        if (years is None) or (years.isna().all()):
            if season_col is not None:
                years = df[season_col].map(season_to_year)
    df = df.assign(year=years)
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    # ====== 只保留在 YEAR_MIN ~ YEAR_MAX 范围的数据 ======
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)]

    # ==== Big5标记列推断 ====
    from_big5_col = pick_first_col(df, POSSIBLE_FROM_BIG5_COLS)
    to_big5_col   = pick_first_col(df, POSSIBLE_TO_BIG5_COLS)
    from_big5 = coerce_bool(df[from_big5_col])
    to_big5   = coerce_bool(df[to_big5_col])

    # ==== 统计 ====
    total_per_year = df.groupby("year").size().rename("total_transfers")
    mask_nonbig5_to_big5 = (from_big5 == False) & (to_big5 == True)
    nonbig5_to_big5_per_year = (
        df[mask_nonbig5_to_big5]
        .groupby("year")
        .size()
        .rename("nonbig5_to_big5")
    )

    stat = pd.concat([total_per_year, nonbig5_to_big5_per_year], axis=1).fillna(0).astype(int)
    stat = stat.sort_index()

    # ==== 绘图 ====
    
    
        # 设置中文字体（黑体）
    plt.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号

    plt.figure(figsize=(10, 5), dpi=140)
    plt.plot(stat.index, stat["nonbig5_to_big5"], marker="o", label="非五大联赛 → 五大联赛")
    plt.plot(stat.index, stat["total_transfers"], marker="o", label="全部转会")
    plt.xlabel("年份")
    plt.ylabel("转会次数")
    # plt.title(f"转会统计 ({YEAR_MIN}-{YEAR_MAX})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, bbox_inches="tight")
    print(f"已保存图像：{OUT_PNG}")

if __name__ == "__main__":
    main()
