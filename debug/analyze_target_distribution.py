#!/usr/bin/env python3
import pandas as pd
import numpy as np
import yaml
import sys

with open("config/car_liab/v1/config.yaml") as f:
    cfg = yaml.safe_load(f)

pc = open("current.pc").read().strip()
ob = cfg["machines"][f"PC{pc}"]["paths"]["output_path"]
target = cfg["experiment"]["target"]
exposure = cfg["experiment"]["exposure"]

print("="*70)
print("TARGET DISTRIBUTION ANALYSIS")
print("="*70)

train = pd.read_parquet(f"{ob}/data/04b_train.parquet")
glm = pd.read_parquet(f"{ob}/data/04c_train_glm.parquet")
glm = glm[glm["glm_pred"].notna()]
idx = train.index.intersection(glm.index)
train = train.loc[idx]
glm_pred = glm.loc[idx, "glm_pred"].values

pp = train[target].values
exp = train[exposure].values
y = pp / glm_pred

print(f"\nRecords: {len(pp):,}")
print(f"\n1. PP_ACTUAL:")
print(f"  Mean: {pp.mean():.4f}, Median: {np.median(pp):.4f}")
print(f"  Min: {pp.min():.4f}, Max: {pp.max():.4f}")
print(f"  Zeros: {(pp==0).sum():,} ({100*(pp==0).sum()/len(pp):.2f}%)")

print(f"\n2. GLM PREDICTIONS:")
print(f"  Mean: {glm_pred.mean():.4f}, Median: {np.median(glm_pred):.4f}")
print(f"  Min: {glm_pred.min():.4f}, Max: {glm_pred.max():.4f}")

print(f"\n3. RATIO (PP/GLM):")
print(f"  Mean: {y.mean():.4f}, Median: {np.median(y):.4f}, Std: {np.std(y):.4f}")
print(f"  Min: {y.min():.4f}, Max: {y.max():.4f}")
print(f"  NaN: {np.isnan(y).sum()}")

print(f"\n  Percentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"    {p:5.1f}%: {np.percentile(y, p):10.4f}")

print(f"\n4. DECILES:")
df = pd.DataFrame({"ratio": y, "pp": pp, "glm": glm_pred})
df["dec"] = pd.qcut(df["ratio"], q=10, labels=False, duplicates="drop") + 1

print(f"\n  Dec | Count    | Ratio Avg | Ratio Min | Ratio Max")
for d in sorted(df["dec"].unique()):
    sub = df[df["dec"] == d]
    print(f"  {int(d):3d} | {len(sub):8,} | {sub["ratio"].mean():9.4f} | {sub["ratio"].min():9.4f} | {sub["ratio"].max():9.2f}")

print(f"\n  Dec | PP Avg    | PP Max        | GLM Avg")
for d in sorted(df["dec"].unique()):
    sub = df[df["dec"] == d]
    print(f"  {int(d):3d} | {sub["pp"].mean():9.2f} | {sub["pp"].max():13.2f} | {sub["glm"].mean():7.2f}")

print(f"\n5. EXTREMES:")
print(f"  Ratio > 10: {(y>10).sum():,} ({100*(y>10).sum()/len(y):.2f}%)")
print(f"  Ratio < 0.1: {(y<0.1).sum():,} ({100*(y<0.1).sum()/len(y):.2f}%)")

print("\n" + "="*70)
