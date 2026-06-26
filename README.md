---
title: SemiSight — Semiconductor Process Intelligence
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: true
short_description: Yield Prediction · Wafer Defects · SHAP Root Cause
---

# 🔬 SemiSight — Semiconductor Process Intelligence System

**Production ML platform for semiconductor manufacturing — built by [Bara Luch](https://github.com/BARA-LUCH)**

## What This Does

- **Yield Prediction** — XGBoost on 590 SECOM process parameters (imbalanced, ~6.5% failure rate)
- **Wafer Defect Classification** — 9-class pattern recognition (Center, Donut, Scratch, Edge-Ring, etc.)
- **Root Cause Analysis** — SHAP identifies which process parameters drive yield failures
- **Process Intelligence** — 8 SQL queries analyzing chamber, lot, time, and step-level patterns
- **Business Impact** — Revenue saved calculator for fab operations

## Dataset

- **SECOM** (UCI ML Repository) — 1,567 semiconductor manufacturing runs, 590 process parameters
- **WM-811K style** — Wafer map defect patterns (synthetic, matching real distribution)

## Results

- XGBoost handles class imbalance via `scale_pos_weight`
- SHAP waterfall charts show per-run failure explanation
- Chamber × Process Step heatmap identifies equipment issues

## Stack

`XGBoost` `Isolation Forest` `SHAP` `SQLite` `Plotly` `Streamlit` `Scikit-learn` `Pandas`

## Author

**Bara Luch** · ML Engineer & Data Scientist · Expected Apr 2026

[GitHub](https://github.com/BARA-LUCH) · [LinkedIn](https://linkedin.com/in/bara-luch) · [Portfolio](https://bara-luch.github.io)
