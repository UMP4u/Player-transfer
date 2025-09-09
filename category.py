import pandas as pd

# 读取数据
file_path = "D:\\Desktop\\workspace\\workspace\\football\\club_degree_by_year.csv"
df = pd.read_csv(file_path)

# 过滤年份 2000–2024
df_filtered = df[(df["year"] >= 2000) & (df["year"] <= 2024)].copy()

# 按 club_id 聚合，计算总 in_degree 和 out_degree
club_summary = (
    df_filtered.groupby("club_id")
    .agg(
        club_name=("club_name", lambda x: x.value_counts().index[0]),  # 当年出现最多的名称
        total_in=("in_degree", "sum"),
        total_out=("out_degree", "sum"),
    )
    .reset_index()
)

# 添加俱乐部类型（买方 / 卖方 / 均衡）
club_summary["type"] = club_summary.apply(
    lambda row: "buyer" if row["total_in"] > row["total_out"]
    else ("seller" if row["total_out"] > row["total_in"] else "balanced"),
    axis=1
)

# 计算总转会次数
club_summary["total_transfers"] = club_summary["total_in"] + club_summary["total_out"]

# 按总转会次数排序
club_summary_sorted = club_summary.sort_values("total_transfers", ascending=False)

# 保存结果到 CSV
out_path = "club_in_out_summary_2000_2024.csv"
club_summary_sorted.to_csv(out_path, index=False)

print(f"统计完成，结果已保存到 {out_path}")
print(club_summary_sorted.head(10))
