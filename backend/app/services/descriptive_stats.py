"""Server-side descriptive statistics computation using pandas/scipy."""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats as scipy_stats


def compute_numeric_stats(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute mean, SD, median, min, max, skewness, kurtosis per numeric column."""
    results = []
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        results.append({
            "column": str(col),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "median": round(float(series.median()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "skewness": round(float(series.skew()), 4),
            "kurtosis": round(float(series.kurtosis()), 4),
        })
    return results


def compute_categorical_stats(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute frequencies, percentages, and mode per non-numeric column."""
    results = []
    non_numeric = df.select_dtypes(exclude="number").columns
    for col in non_numeric:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        freq = series.value_counts().to_dict()
        total = len(series)
        pct = {k: round(v / total * 100, 2) for k, v in freq.items()}
        results.append({
            "column": str(col),
            "frequencies": {str(k): int(v) for k, v in freq.items()},
            "percentages": {str(k): v for k, v in pct.items()},
            "mode": str(series.mode().iloc[0]) if len(series.mode()) > 0 else "",
        })
    return results


def compute_correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> dict[str, dict[str, float]]:
    """Compute Pearson or Spearman correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return {}
    corr = numeric_df.corr(method=method)
    return {
        str(col): {str(idx): round(float(val), 4) for idx, val in row.items()}
        for col, row in corr.items()
    }


def compute_normality_tests(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Run Shapiro-Wilk (n<=5000) or D'Agostino (n>5000) normality test per numeric column."""
    results = []
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 3:
            continue
        # Shapiro-Wilk for small samples, D'Agostino's K^2 for large
        if len(series) <= 5000:
            stat, p_val = scipy_stats.shapiro(series)
        else:
            stat, p_val = scipy_stats.normaltest(series)
        results.append({
            "column": str(col),
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 4),
            "is_normal": bool(p_val > 0.05),
        })
    return results


def compute_all_descriptive_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Combine all descriptive statistics into a single dict."""
    return {
        "numeric_stats": compute_numeric_stats(df),
        "categorical_stats": compute_categorical_stats(df),
        "correlation_matrix": compute_correlation_matrix(df),
        "normality_tests": compute_normality_tests(df),
    }
