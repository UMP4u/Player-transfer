"""
dedupe_union.py
----------------
输入两张表 A 与 B（列集合相同），输出：
1) C：B 中相对于 A 的“独有行”
2) D：A 与 C 的并集（纵向合并 A 与 C）

核心思路：
- 统一把要比较的列转成 pandas 的“string”类型，并把缺失值统一替换为占位符 "<NA>"，
  这样可以保证相等性比较稳定（空值与类型差异都不会影响）。
- 对“整行”做哈希（hash_pandas_object），把 A 的行哈希放入集合中，
  然后检查 B 的每一行哈希是否存在于集合，即可快速得到“B 的独有行”。

使用示例：
    python dedupe_union.py --a path/to/A.csv --b path/to/B.csv --outdir out

输出文件：
- C_unique_in_B.csv
- D_A_plus_C.csv
"""
import argparse
import os
import sys
from pathlib import Path
import pandas as pd

def read_table(path: Path) -> pd.DataFrame: #Path类型，来自pathlib，代表文件路径    ->返回值的类型标注(可不写)
    """
    根据扩展名自动选择读表方式，统一读成 dtype="object"（后续再统一转 string）。
    这样可以最大程度兼容“同一列混合类型”的真实数据场景。

    支持：
    - .csv / .txt：默认逗号分隔
    - .tsv：制表符分隔
    - .xlsx / .xls：Excel
    """
    ext = path.suffix.lower() #path.suffix返回扩展名    suffix：后缀
    if ext in (".csv", ".txt", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        return pd.read_csv(path, sep=sep, dtype="object", keep_default_na=True) #先把所有数据转换成object，保证数据读取不出错。     让空值识别为缺失值NaN
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype="object")
    else:
        raise ValueError(f"Unsupported file extension for {path}. Use CSV/TSV/XLSX/XLS.") #raise抛出异常，中断程序


def normalize_for_hash(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    为“行级哈希比较”做标准化（非常关键）：
    1) 只保留要比较的列（顺序已在外部统一）
    2) 全部转为 pandas 的扩展字符串类型 'string'
    3) 用统一占位符 "<NA>" 填充缺失值
       ——这样 A 与 B 若在同一列都缺失，会被视为“相等”，避免 NaN != NaN 的问题

    注意：若数据里可能真的出现字符串 "<NA>"，可以把占位符换成更冷门的标记。
    """
    norm = df[cols].copy() #复制需要比较的列
    for c in cols: #遍历每一列，把这一列转为
        norm[c] = norm[c].astype("string")  
    norm = norm.fillna("<NA>")
    return norm


def save_table(df: pd.DataFrame, path_csv: Path, also_excel: bool = False):
    """
    保存结果：
    - 默认保存为 CSV
    - 若 also_excel=True，再额外保存一份 .xlsx
    """
    df.to_csv(path_csv, index=False, encoding="utf-8-sig") #不保存 DataFrame 的行索引
    if also_excel:
        xlsx_path = path_csv.with_suffix(".xlsx")
        df.to_excel(xlsx_path, index=False)


def main():
    # ---------- 1) 解析命令行参数 ----------
    parser = argparse.ArgumentParser( # parse解析
        description="从 B 中剔除与 A 完全相同的行得到 C，并将 A 与 C 合并为 D。支持 CSV/TSV/Excel。"
    ) #描述在 python .py -h 时显示    required=True表示必须提供该参数
    parser.add_argument("--a", required=True, help="A 文件路径（CSV/TSV/XLSX/XLS）")
    parser.add_argument("--b", required=True, help="B 文件路径（CSV/TSV/XLSX/XLS）")
    parser.add_argument("--outdir", default=".", help="输出目录（默认当前目录）")
    parser.add_argument("--excel", action="store_true", help="同时输出 .xlsx 版本") # 如果命令行里出现了这个选项，就把它的值设为 True
    args = parser.parse_args() #读取命令行里用户输入的参数 

    # ---------- 2) 路径准备与输出目录创建 ----------
    a_path = Path(args.a)
    b_path = Path(args.b)
    outdir = Path(args.outdir) 
    outdir.mkdir(parents=True, exist_ok=True) # 创建文件夹 如果父目录不存在，也一并创建  如果目录已经存在，就不报错，直接跳过

    # ---------- 3) 读取输入表 ----------
    A = read_table(a_path)
    B = read_table(b_path)

    if A.empty and B.empty:
        print("Both A and B are empty. Nothing to do.", file=sys.stderr)
        sys.exit(0) # 正常退出（0 通常表示成功）

    # ---------- 4) 列校验与列顺序对齐 ----------
    a_cols = list(A.columns) 
    b_cols = list(B.columns) 
    if set(a_cols) != set(b_cols): #列名顺序可以不同
        # 打印出彼此缺了哪些列
        missing_in_B = [c for c in a_cols if c not in b_cols]
        missing_in_A = [c for c in b_cols if c not in a_cols]
        raise ValueError(
            "A and B must have the same set of columns.\n"
            f"Missing in B: {missing_in_B}\nMissing in A: {missing_in_A}"
        )

    # 将 B 的列顺序对齐为 A 的列顺序（确保逐行比较时“同名列对同名列”）
    cols = a_cols
    B = B[cols]

    # ---------- 5) 归一化并计算“整行哈希” ----------
    
    A_norm = normalize_for_hash(A, cols) # 归一化（转 string + 统一缺失）
    B_norm = normalize_for_hash(B, cols)

    ha = pd.util.hash_pandas_object(A_norm, index=False) #对每行计算哈希值（index=False 表示只基于数据，不考虑行索引）
    hb = pd.util.hash_pandas_object(B_norm, index=False)

    # 把 A 表每一行的哈希值 放进一个 集合 (set) 
    a_hash_set = set(ha.values.tolist()) #.values返回一个numpy数组，再转成列表

    mask_unique_B = ~hb.isin(a_hash_set) #~取反  返回布尔序列

    C = B.loc[mask_unique_B].copy() #.loc按标签索引

    # ---------- 6) 生成 D（A ∪ C） ----------
    D = pd.concat([A, C], ignore_index=True) #纵向拼接  忽略原来的行索引，重新生成新的行索引

    # ---------- 7) 保存结果 ----------
    C_csv = outdir / "C_unique_in_B.csv"  # /路径拼接运算符
    D_csv = outdir / "D_A_plus_C.csv"

    save_table(C, C_csv, also_excel=args.excel)
    save_table(D, D_csv, also_excel=args.excel)

    # ---------- 8) 打印汇总信息 ----------
    print(f"[OK] Rows in A: {len(A)}")
    print(f"[OK] Rows in B: {len(B)}")
    print(f"[OK] Rows unique to B (C): {len(C)} -> saved to: {C_csv}")
    if args.excel:
        print(f"      Also wrote: {C_csv.with_suffix('.xlsx')}")
    print(f"[OK] Rows in D (A + C): {len(D)} -> saved to: {D_csv}")
    if args.excel:
        print(f"      Also wrote: {D_csv.with_suffix('.xlsx')}")


if __name__ == "__main__": 
    main()
#Python 在运行时，会给当前模块定义一个特殊变量 __name__
# 如果这个文件是 被直接运行 的，那么 __name__ 的值就是 "__main__"；  执行main()函数
# 如果这个文件是 被当作模块 import 的，那么 __name__ 的值就是这个模块的名字
# （比如 utils.py 被 import 时，__name__ == "utils"）