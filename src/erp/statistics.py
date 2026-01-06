from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

import pingouin as pg
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests


class ERPStatistics:
    """Statistics runner for subject-level ERP measurements."""

    REQUIRED_COLUMNS = ["subject_id", "component", "condition"]

    def __init__(self, data: Union[str, Path, pd.DataFrame]):
        if isinstance(data, (str, Path)):
            p = Path(data)
            if not p.exists():
                raise FileNotFoundError(f"Data file not found: {p}")
            self.data = pd.read_csv(p, dtype={"subject_id": str})
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
            if "subject_id" in self.data.columns:
                self.data["subject_id"] = self.data["subject_id"].astype(str)
        else:
            raise TypeError("data must be a file path or pandas DataFrame")

        missing = [c for c in self.REQUIRED_COLUMNS if c not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def filter_data(
        self,
        *,
        component: Optional[str] = None,
        condition: Optional[str] = None,
        min_snr: Optional[float] = None,
        dropna: bool = True,
    ) -> pd.DataFrame:
        df = self.data.copy()
        if component is not None:
            df = df[df["component"] == component]
        if condition is not None:
            df = df[df["condition"] == condition]
        if min_snr is not None and "snr" in df.columns:
            df = df[df["snr"] >= float(min_snr)]
        if dropna:
            df = df.dropna()
        return df

    def descriptives(self, *, component: str, dv: str, groupby: str = "condition", **filter_kwargs) -> pd.DataFrame:
        df = self.filter_data(component=component, **filter_kwargs)
        if dv not in df.columns:
            raise ValueError(f"Missing dependent variable column: {dv}")
        g = df.groupby(groupby, dropna=False)[dv]
        out = g.agg(["count", "mean", "std", "min", "max"]).reset_index()
        out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
        return out

    def run_anova(
        self,
        *,
        data: pd.DataFrame,
        dv: str,
        within: str = "condition",
        subject: str = "subject_id",
    ) -> pd.DataFrame:
        required = [subject, within, dv]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns for ANOVA: {missing}")
        return pg.rm_anova(data=data, dv=dv, within=within, subject=subject, detailed=True)

    def run_pairwise(
        self,
        *,
        data: pd.DataFrame,
        dv: str,
        within: str = "condition",
        subject: str = "subject_id",
        correction: str = "fdr_bh",
        effsize: str = "cohen",
    ) -> pd.DataFrame:
        required = [subject, within, dv]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns for pairwise tests: {missing}")
        return pg.pairwise_tests(
            data=data,
            dv=dv,
            within=within,
            subject=subject,
            padjust=correction,
            effsize=effsize,
            parametric=True,
            return_desc=False,
        )

    def run_lmm(
        self,
        *,
        data: pd.DataFrame,
        dv: str,
        fixed: str,
        random: str = "subject_id",
        method: str = "lbfgs",
    ) -> Dict[str, Any]:
        if dv not in data.columns:
            raise ValueError(f"Missing dependent variable column: {dv}")
        if random not in data.columns:
            raise ValueError(f"Missing random-effects grouping column: {random}")

        # Build formula: dv ~ fixed
        formula = f"{dv} ~ {fixed}"
        model = mixedlm(formula, data=data, groups=data[random])

        # MixedLM can fail with singular matrices depending on the dataset/contrast.
        # We keep LMM-first, but degrade gracefully (retry a few optimizers; then emit an error payload).
        import numpy.linalg as la

        attempted = []
        last_err: Optional[Exception] = None
        for m in [str(method), "lbfgs", "powell", "nm"]:
            if m in attempted:
                continue
            attempted.append(m)
            try:
                result = model.fit(method=m, reml=False, disp=False)
                last_err = None
                break
            except (la.LinAlgError, ValueError) as e:
                last_err = e
                continue
        else:
            # No break -> failed all attempts
            return {
                "aic": float("nan"),
                "bic": float("nan"),
                "converged": False,
                "error": str(last_err) if last_err else "Unknown LMM fitting error",
                "attempted_methods": attempted,
                "summary": [],
            }

        # Build a stable JSON-friendly summary
        summary_rows = []
        try:
            params = result.params
            bse = result.bse
            pvalues = result.pvalues
            zvalues = getattr(result, "tvalues", None)  # MixedLM uses z-like
            for name in params.index:
                summary_rows.append(
                    {
                        "name": str(name) if str(name).lower() != "intercept" else "Intercept",
                        "Coef.": float(params[name]),
                        "Std.Err.": float(bse[name]) if name in bse.index else float("nan"),
                        "z": float(zvalues[name]) if (zvalues is not None and name in zvalues.index) else float("nan"),
                        "P>|z|": float(pvalues[name]) if name in pvalues.index else float("nan"),
                    }
                )
        except Exception:
            summary_rows = []

        return {
            "aic": float(getattr(result, "aic", float("nan"))),
            "bic": float(getattr(result, "bic", float("nan"))),
            "converged": bool(getattr(result, "converged", True)),
            "summary": summary_rows,
        }

    def run_lmm_pairwise(
        self,
        *,
        data: pd.DataFrame,
        dv: str,
        fixed: str,
        random: str = "subject_id",
        correction: str = "hs",
        method: str = "lbfgs",
    ) -> pd.DataFrame:
        """Pairwise comparisons via LMM by fitting a two-level model for each condition pair."""
        if dv not in data.columns:
            raise ValueError(f"Missing dependent variable column: {dv}")
        if "condition" not in data.columns:
            raise ValueError("Missing 'condition' column for lmm_pairwise")

        levels = sorted({str(x) for x in data["condition"].astype(str).unique()})
        rows = []
        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                a, b = levels[i], levels[j]
                sub = data[data["condition"].astype(str).isin([a, b])].copy()
                # Keep a as reference
                sub["condition"] = pd.Categorical(sub["condition"].astype(str), categories=[a, b], ordered=True)
                try:
                    res = self.run_lmm(data=sub, dv=dv, fixed=fixed, random=random, method=method)
                    # Find the condition coefficient row (anything starting with condition[T.)
                    p_unc = float("nan")
                    beta = float("nan")
                    for row in res.get("summary", []):
                        nm = str(row.get("name", ""))
                        if nm.lower().startswith("condition"):
                            p_unc = float(row.get("P>|z|", float("nan")))
                            beta = float(row.get("Coef.", float("nan")))
                            break
                except Exception:
                    p_unc = float("nan")
                    beta = float("nan")
                rows.append({"A": a, "B": b, "beta": beta, "p-unc": p_unc})

        out = pd.DataFrame(rows)
        if len(out) == 0:
            out["p-corr"] = []
            return out

        # Multipletest correction
        pvals = out["p-unc"].astype(float).fillna(1.0).values
        method_map = {
            "hs": "holm-sidak",
            "bonferroni": "bonferroni",
            "sidak": "sidak",
            "fdr_bh": "fdr_bh",
            "none": None,
        }
        m = method_map.get(str(correction).lower(), "holm-sidak")
        if m is None:
            out["p-corr"] = out["p-unc"]
        else:
            _, p_corr, _, _ = multipletests(pvals, method=m)
            out["p-corr"] = p_corr
        return out


def get_library_versions() -> Dict[str, str]:
    import sys
    import mne
    import pingouin
    import statsmodels

    return {
        "python": "{}.{}.{}".format(*sys.version_info[:3]),
        "mne": getattr(mne, "__version__", "unknown"),
        "pingouin": getattr(pingouin, "__version__", "unknown"),
        "statsmodels": getattr(statsmodels, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown"),
        "pandas": getattr(pd, "__version__", "unknown"),
    }


def save_json(path: Union[str, Path], payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def compute_cohens_d_ci(
    *,
    x: np.ndarray,
    y: np.ndarray,
    confidence: float = 0.95,
    paired: bool = True,
) -> tuple[float, float, float]:
    """Cohen's d with CI (paired by default).

    Ported from the reference pipeline to support effect-size CI plots.
    """
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    if paired:
        if len(x) != len(y):
            raise ValueError("Paired samples must have equal length")
        n = len(x)
        if n < 2:
            return float("nan"), float("nan"), float("nan")
        diff = x - y
        mean_diff = float(np.mean(diff))
        sd_diff = float(np.std(diff, ddof=1))
        if sd_diff == 0:
            return float("nan"), float("nan"), float("nan")
        d = mean_diff / sd_diff
        se = np.sqrt((1.0 / n) + (d**2 / (2.0 * n)))
        df = n - 1
        alpha = 1.0 - float(confidence)
        t_crit = float(stats.t.ppf(1 - alpha / 2.0, df))
        return float(d), float(d - t_crit * se), float(d + t_crit * se)

    # Independent samples
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan"), float("nan")
    mean_diff = float(np.mean(x) - np.mean(y))
    var1 = float(np.var(x, ddof=1))
    var2 = float(np.var(y, ddof=1))
    pooled_sd = float(np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)))
    if pooled_sd == 0:
        return float("nan"), float("nan"), float("nan")
    d = mean_diff / pooled_sd
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2.0 * (n1 + n2)))
    df = n1 + n2 - 2
    alpha = 1.0 - float(confidence)
    t_crit = float(stats.t.ppf(1 - alpha / 2.0, df))
    return float(d), float(d - t_crit * se), float(d + t_crit * se)


__all__ = [
    "ERPStatistics",
    "get_library_versions",
    "save_json",
    "compute_cohens_d_ci",
]

