#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测 data/dft.csv 中是否存在以下任一情况：
- S1 < T1
- ST Gap < 0（若无 ST Gap 列，则用 S1 - T1 计算）
满足任一条件则输出整行数据。
"""
import sys
import argparse
import pandas as pd


def find_col(df, candidates):
    """在 DataFrame 中寻找第一个存在的列名（大小写不敏感），返回真实列名或 None"""
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        key = name.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def main():
    parser = argparse.ArgumentParser(description="Check anomalies in dft.csv: S1<T1 or ST Gap<0")
    parser.add_argument("--csv", default="data/dft.csv", help="CSV 路径，默认 data/dft.csv")
    parser.add_argument("--out", default=True, help="可选：将异常行保存到该路径（如 data/dft_anomalies.csv）")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    s1_col = find_col(df, ["S1", "s1"]) 
    t1_col = find_col(df, ["T1", "t1"]) 
    if s1_col is None or t1_col is None:
        print("找不到 S1/T1 列，请确认列名是否为 'S1'/'T1'（大小写不敏感）")
        sys.exit(1)

    gap_col = find_col(df, ["ST Gap", "ST_Gap", "st_gap", "STgap", "stgap"]) 

    cond_s1_lt_t1 = df[s1_col] < df[t1_col]

    if gap_col is not None:
        cond_gap_lt_0 = df[gap_col] < 0
        gap_series = df[gap_col]
    else:
        gap_series = df[s1_col] - df[t1_col]
        cond_gap_lt_0 = gap_series < 0

    mask = cond_s1_lt_t1 | cond_gap_lt_0
    anomalies = df[mask].copy()

    print(f"总行数: {len(df)}, 异常行数: {len(anomalies)}")

    if anomalies.empty:
        print("未发现异常行。")
    else:
        # 输出整行数据
        # 为了便于查看，同时附上计算/选择的 ST_Gap 列（若原本没有则是计算值）
        anomalies = anomalies.assign(ST_Gap_Checked=gap_series[mask].values)
        # 直接以 CSV 形式打印，避免列过宽时换行混乱
        csv_text = anomalies.to_csv(index=False)
        print(csv_text)
        if args.out:
            anomalies.to_csv(args.out, index=False)
            print(f"已保存到: {args.out}")


if __name__ == "__main__":
    main()
