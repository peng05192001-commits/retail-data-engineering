""" cleansing, Retail, integration: Customer profiling customer profiling.

Read the final three cleaned CSVs. No cleaning, input modification, CSV export,
network access or clustering is performed. See the companion notebook for the
analysis narrative and run this file with --help for a text-only entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd


BATCHES = ("dirty", "missing", "outlier")
ORDER_COLUMNS = [
    "order_id", "customer_id", "date", "nearest_warehouse", "shopping_cart",
    "order_price", "delivery_charges", "customer_lat", "customer_long",
    "coupon_discount", "order_total", "season", "is_expedited_delivery",
    "distance_to_nearest_warehouse", "latest_customer_review", "is_happy_customer",
]
FEATURES = ["order_count", "mean_order_value", "mean_delivery_distance"]
LABELS = ["Order count", "Mean order value", "Mean delivery distance"]
UNITS = ["orders", "AUD", "km"]
METHODS = ["Original", "Z-score", "Min-max", "Log1p value + Z-score"]


class ProfileInputError(ValueError):
    """An input condition prevents a defensible Customer profiling calculation."""


def input_paths(clean_dir: str | Path, dataset_id: str = "Retail") -> dict:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", dataset_id):
        raise ProfileInputError("Use a safe dataset alias, e.g. Retail.")
    return {batch: Path(clean_dir) / f"{dataset_id}_{batch}_clean.csv" for batch in BATCHES}


def input_status(clean_dir: str | Path, dataset_id: str = "Retail") -> pd.DataFrame:
    return pd.DataFrame([
        {"batch": batch, "required_file": path.name, "exists": path.is_file()}
        for batch, path in input_paths(clean_dir, dataset_id).items()
    ])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cleaned_orders(clean_dir, dataset_id="Retail", raw_dir=None):
    """Validate essential Customer profiling prerequisites, retaining batch provenance.

    This is not a substitute for the business-rule checks in Audit validation. Optional
    raw_dir verifies retained IDs and unchanged customer assignments against
    the allocated original batches. A missing file never falls back to raw data.
    """
    paths = input_paths(clean_dir, dataset_id)
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Customer profiling is waiting for final cleaned CSVs:\n" + "\n".join(missing)
            + "\nDo not rename *_data.csv files to *_clean.csv."
        )
    frames, batch_counts, checks, fingerprints = [], [], [], {}
    for batch, path in paths.items():
        payload = path.read_bytes()
        fingerprints[path.resolve()] = hashlib.sha256(payload).hexdigest()
        frame = pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False)
        if list(frame.columns) != ORDER_COLUMNS:
            raise ProfileInputError(f"{path.name}: expected exactly the original 16 columns in order.")
        if batch in ("dirty", "missing") and len(frame) != 500:
            raise ProfileInputError(f"{path.name}: expected all 500 orders, received {len(frame)}.")
        if batch == "outlier" and len(frame) > 500:
            raise ProfileInputError(f"{path.name}: cannot contain more than 500 retained orders.")
        blank_columns = [c for c in frame if frame[c].str.strip().eq("").any()]
        if blank_columns:
            raise ProfileInputError(f"{path.name}: unresolved empty fields in {blank_columns}.")
        if frame["order_id"].duplicated().any():
            raise ProfileInputError(f"{path.name}: duplicate order_id values.")
        for field in ("order_total", "distance_to_nearest_warehouse"):
            values = pd.to_numeric(frame[field], errors="coerce")
            if not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0).any():
                raise ProfileInputError(f"{path.name}: {field} must be finite and non-negative.")
            frame[field] = values.astype(float)
        if raw_dir is not None:
            raw_path = Path(raw_dir) / f"{dataset_id}_{batch}_data.csv"
            original = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
            if not {"order_id", "customer_id"}.issubset(original.columns):
                raise ProfileInputError(f"{raw_path.name}: original ID columns are missing.")
            if len(original) != 500 or original["order_id"].duplicated().any():
                raise ProfileInputError(f"{raw_path.name}: expected 500 distinct source orders.")
            supplied, retained = set(original.order_id), set(frame.order_id)
            if not retained <= supplied or (batch != "outlier" and retained != supplied):
                raise ProfileInputError(f"{path.name}: retained IDs disagree with the original batch.")
            customer_map = original.set_index("order_id")["customer_id"]
            if not frame.order_id.map(customer_map).equals(frame.customer_id):
                raise ProfileInputError(f"{path.name}: an original customer_id has changed.")
        frame = frame.assign(source_batch=batch, source_file=path.name)
        frames.append(frame)
        batch_counts.append({
            "source_batch": batch, "source_file": path.name,
            "retained_rows": len(frame), "distinct_orders": frame.order_id.nunique(),
            "customers_in_batch": frame.customer_id.nunique(),
            "removed_orders": 500 - len(frame) if batch == "outlier" else 0,
        })
        checks.append({"check": f"Input schema, IDs and profile fields: {batch}", "result": "PASS"})
    orders = pd.concat(frames, ignore_index=True)
    if orders.order_id.duplicated().any():
        raise ProfileInputError("Duplicate order_id across the cleaned batches; do not silently deduplicate.")
    audit_path = Path(clean_dir) / "audit_log.csv"
    if audit_path.is_file():
        fingerprints[audit_path.resolve()] = sha256(audit_path)
    checks.append({"check": "Order IDs unique across all batches", "result": "PASS"})
    checks.append({
        "check": "Original ID sets and customer assignments",
        "result": "PASS" if raw_dir is not None else "NOT RUN: RAW_DIR not configured; use Audit validation evidence",
    })
    return orders, pd.DataFrame(batch_counts), pd.DataFrame(checks), fingerprints


def build_profiles(orders: pd.DataFrame) -> pd.DataFrame:
    """Select exactly three specified attributes; customer_id remains an index."""
    if orders.order_id.duplicated().any():
        raise ProfileInputError("Each retained order must contribute exactly once.")
    profiles = orders.groupby("customer_id", sort=True).agg(
        order_count=("order_id", "nunique"),
        mean_order_value=("order_total", "mean"),
        mean_delivery_distance=("distance_to_nearest_warehouse", "mean"),
    )
    if profiles.empty:
        raise ProfileInputError("No customers are available for analysis.")
    if int(profiles.order_count.sum()) != len(orders):
        raise ProfileInputError("Customer order counts do not reconcile to retained orders.")
    if not np.isfinite(profiles.to_numpy(dtype=float)).all():
        raise ProfileInputError("Non-finite aggregated features.")
    return profiles[FEATURES]


def describe_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Population SD (ddof=0); sample skewness is undefined for constants."""
    records = []
    for name in frame:
        x = frame[name].astype(float)
        constant = bool(x.max() == x.min())
        records.append({
            "attribute": name, "n": len(x), "min": x.min(), "q25": x.quantile(.25),
            "median": x.median(), "mean": x.mean(), "q75": x.quantile(.75),
            "p95": x.quantile(.95), "max": x.max(), "std_ddof0": x.std(ddof=0),
            "skewness": np.nan if constant or len(x) < 3 else x.skew(),
            "zero_count": int(x.eq(0).sum()), "constant": constant,
        })
    return pd.DataFrame(records).set_index("attribute")


def scale_features(frame: pd.DataFrame):
    """Fit descriptive scaling on the retained sample; no predictive claim.

    Constants map to zero and therefore contribute no customer separation.
    Values are not clipped or rounded. Parameters are returned for inspection.
    """
    x = frame.astype(float)
    if not np.isfinite(x.to_numpy()).all():
        raise ProfileInputError("Scaling requires finite numeric values.")
    minimum, maximum = x.min(), x.max()
    mean, sd = x.mean(), x.std(ddof=0)
    span = maximum - minimum
    constant = span.eq(0)
    zscore = (x - mean) / sd.mask(constant, 1.0)
    minmax = (x - minimum) / span.mask(constant, 1.0)
    zscore.loc[:, constant] = 0.0
    minmax.loc[:, constant] = 0.0
    params = pd.DataFrame({"mean": mean, "std_ddof0": sd, "min": minimum,
                           "max": maximum, "constant": constant})
    return zscore, minmax, params


def compare_transformations(profiles: pd.DataFrame):
    baseline = profiles.copy(deep=True)
    zscore, minmax, parameters = scale_features(baseline)
    if (baseline.mean_order_value < 0).any():
        raise ProfileInputError("This comparison assumes non-negative paid values in AUD.")
    nonlinear = baseline.astype(float).copy()
    nonlinear["mean_order_value"] = np.log1p(nonlinear["mean_order_value"])
    log_scaled, _, log_parameters = scale_features(nonlinear)
    versions = dict(zip(METHODS, [baseline, zscore, minmax, log_scaled]))
    scaling_summary = pd.concat(
        {name: describe_features(versions[name]) for name in METHODS[:3]},
        names=["method", "attribute"],
    )
    nonlinear_summary = pd.concat({
        "Original AUD": describe_features(baseline[["mean_order_value"]]),
        "Log1p(AUD / 1 AUD)": describe_features(nonlinear[["mean_order_value"]]),
    }, names=["method", "attribute"])
    correlations = {name: frame.corr() for name, frame in versions.items()}
    pd.testing.assert_frame_equal(baseline, profiles)
    return {
        "versions": versions, "parameters": parameters, "log_parameters": log_parameters,
        "scaling_summary": scaling_summary, "nonlinear_values": nonlinear,
        "nonlinear_summary": nonlinear_summary, "correlations": correlations,
    }


def distance_effects(versions: dict) -> pd.DataFrame:
    """Exact all-customer diagnostics for cleansing (at most 1,500 customers).

    Across all unordered pairs, feature squared-distance share = variance share.
    Pairwise Spearman correlation uses average ranks for ties. It measures how
    relative distances change, not whether a representation is correct.
    """
    original = versions["Original"]
    n = len(original)
    i, j = np.triu_indices(n, k=1)
    baseline_distances = None
    rows = []
    for name, frame in versions.items():
        if not frame.index.equals(original.index) or list(frame.columns) != list(original.columns):
            raise ProfileInputError("All transformation variants must align on customers and attributes.")
        a = frame.to_numpy(dtype=float)
        variance = np.var(a, axis=0)
        shares = variance / variance.sum() if variance.sum() > 0 else np.full(a.shape[1], np.nan)
        distances = pd.Series(np.sqrt(np.sum((a[i] - a[j]) ** 2, axis=1)))
        ranks = distances.rank(method="average")
        if name == "Original":
            baseline_distances = ranks
        defined = (len(ranks) >= 2 and ranks.nunique() > 1
                   and baseline_distances.nunique() > 1)
        corr = ranks.corr(baseline_distances) if defined else np.nan
        row = {"method": name, "customers": n, "pairs": len(i),
               "pairwise_distance_rank_corr_vs_original": corr}
        row.update({f"{feature}_squared_distance_share": share
                    for feature, share in zip(FEATURES, shares)})
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def recommendation_table(profiles, comparisons, effects):
    """A transparent policy: preserve absolute increments; equal marginal variance.

    This is one defensible exploratory starting point, not an optimisation result.
    The log alternative is measured but not selected solely to reduce skewness.
    """
    stats = describe_features(profiles)
    records = []
    for feature in FEATURES:
        s = stats.loc[feature]
        raw_share = effects.loc["Original", f"{feature}_squared_distance_share"]
        mm_share = effects.loc["Min-max", f"{feature}_squared_distance_share"]
        evidence = (
            f"Tables T3-B/T3-D: range {s['min']:.4g} to {s['max']:.4g}; "
            f"population SD {s['std_ddof0']:.4g}. T3-F: original squared-distance "
            f"share {raw_share:.1%}; min-max share {mm_share:.1%}."
        )
        methods = "Original; z-score; min-max"
        if s["constant"]:
            treatment = "Nonlinear: none; scaling: none; exclude from distance (zero separation)."
            reason = "Constant in this retained sample. Both tested scalings map it to zero. Reassess in a new sample."
        else:
            treatment = "Nonlinear: none; scaling: z-score; order: original -> z-score."
            reason = (
                "Use equal marginal variance as an explicit starting weight across activity, "
                "spending and geography. Preserve absolute within-feature increments. "
                "Min-max equalises ranges but not variance; either scaler remains sensitive "
                "to extremes. This does not remove correlation or prove useful clusters."
            )
        if feature == "mean_order_value":
            methods += "; log1p; log1p then z-score"
            before = comparisons["nonlinear_summary"].loc[("Original AUD", feature), "skewness"]
            after = comparisons["nonlinear_summary"].loc[("Log1p(AUD / 1 AUD)", feature), "skewness"]
            evidence += f" T3-E: skewness {before:.3g} -> {after:.3g} after log1p."
            if not s["constant"]:
                reason += (
                    " The log alternative compresses high-value gaps. Keep the linear value "
                    "scale for this stated absolute-AUD comparison; use log1p then z-score "
                    "if relative spending differences become the business priority. "
                    "Lower skewness alone is not the selection rule."
                )
        records.append({"attribute": feature, "methods_compared": methods,
                        "observed_effect_or_evidence": evidence,
                        "recommended_treatment": treatment, "justification_and_limitation": reason})
    return pd.DataFrame(records)


def interpretation(profiles, comparisons, effects):
    """Generate concise numerical observations rather than frozen sample claims."""
    shares = effects.loc["Original", [f"{c}_squared_distance_share" for c in FEATURES]]
    if shares.notna().any():
        feature = shares.idxmax().removesuffix("_squared_distance_share")
        scale_text = (
            f"On the original scales, {feature} has the largest aggregate contribution "
            f"to squared Euclidean distances ({shares.max():.1%}). This is a consequence "
            "of the observed spread and units, not evidence of greater business importance."
        )
    else:
        scale_text = "All selected features are constant; this retained sample has no profile-based separation."
    corr = comparisons["correlations"]["Original"]
    pairs = [(FEATURES[i], FEATURES[j], corr.iloc[i, j])
             for i in range(3) for j in range(i + 1, 3) if pd.notna(corr.iloc[i, j])]
    if pairs:
        a, b, r = max(pairs, key=lambda row: abs(row[2]))
        relation_text = (
            f"The largest absolute Pearson correlation is {a} versus {b} (r={r:.3f}). "
            "Review the scatter plots for redundant information; scaling alone does not remove it."
        )
    else:
        relation_text = "Correlations are undefined where features are constant or too few customers are available."
    rows = [scale_text, relation_text]
    for name in METHODS[1:]:
        r = effects.loc[name, "pairwise_distance_rank_corr_vs_original"]
        rows.append(f"{name}: pairwise-distance rank correlation with the original representation = "
                    + (f"{r:.4f}." if pd.notna(r) else "undefined (insufficient variation)."))
    rows.append(
        "Z-score gives every nonconstant feature unit population variance. Min-max puts "
        "each nonconstant feature between 0 and 1; its relative variance can still differ. "
        "A changed distance ranking is an observed effect, not a quality score."
    )
    return rows


def analysis_checks(orders, profiles, comparisons, fingerprints):
    rows = []
    def record(name, result, observed):
        rows.append({"check": name, "result": "PASS" if result else "FAIL", "observed": str(observed)})
    record("One row per customer", profiles.index.is_unique and len(profiles) == orders.customer_id.nunique(), len(profiles))
    record("Order counts reconcile", profiles.order_count.sum() == len(orders), len(orders))
    for source, profile in [("order_total", "mean_order_value"),
                            ("distance_to_nearest_warehouse", "mean_delivery_distance")]:
        recovered = (profiles[profile] * profiles.order_count).sum()
        actual = orders[source].sum()
        record(f"Weighted customer means reconcile: {source}",
               np.isclose(recovered, actual, rtol=1e-12, atol=1e-8), recovered - actual)
    variable = ~comparisons["parameters"]["constant"]
    for name, frame in comparisons["versions"].items():
        record(f"Finite transformed values: {name}", np.isfinite(frame.to_numpy(dtype=float)).all(), frame.shape)
        record(f"Customer IDs preserved: {name}", frame.index.equals(profiles.index), len(frame))
    z = comparisons["versions"]["Z-score"]
    mm = comparisons["versions"]["Min-max"]
    record("Z-score mean zero", np.allclose(z.mean(), 0, atol=1e-10), z.mean().round(12).to_dict())
    record("Z-score variable-feature SD one", np.allclose(z.loc[:, variable].std(ddof=0), 1), z.std(ddof=0).to_dict())
    record("Min-max values inside [0,1]", ((mm >= -1e-12) & (mm <= 1 + 1e-12)).all().all(), "all attributes")
    record("Min-max variable-feature endpoints", np.allclose(mm.loc[:, variable].min(), 0)
           and np.allclose(mm.loc[:, variable].max(), 1), "0 and 1")
    record("Constants map to zero", (z.loc[:, ~variable] == 0).all().all()
           and (mm.loc[:, ~variable] == 0).all().all(), list(profiles.columns[~variable]))
    original = comparisons["versions"]["Original"]
    record("Untransformed baseline preserved", original.equals(profiles), "exact DataFrame equality")
    recovered = np.expm1(comparisons["nonlinear_values"].mean_order_value)
    record("Log1p reverses to original value", np.allclose(recovered, profiles.mean_order_value), "expm1(log1p(x))")
    for path, before in fingerprints.items():
        record(f"Input bytes unchanged: {path.name}", path.is_file() and sha256(path) == before, "SHA-256 comparison")
    result = pd.DataFrame(rows)
    if result.result.eq("FAIL").any():
        raise AssertionError(result.loc[result.result.eq("FAIL")].to_string(index=False))
    return result


def make_figures(profiles, comparisons, effects):
    """Return matplotlib figures; nothing is written to disk by this function."""
    import matplotlib.pyplot as plt
    colors = ["#27647B", "#CA7548", "#57886E"]
    figures = {}
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 110}):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
        for k, (feature, label, unit) in enumerate(zip(FEATURES, LABELS, UNITS)):
            axes[0, k].hist(profiles[feature], bins="auto", color=colors[k], alpha=.85)
            axes[0, k].set(xlabel=f"{label} ({unit})", ylabel="Customers")
        for ax, (i, j) in zip(axes[1], [(0, 1), (0, 2), (1, 2)]):
            ax.scatter(profiles.iloc[:, i], profiles.iloc[:, j], s=18, alpha=.5, color=colors[j])
            ax.set(xlabel=f"{LABELS[i]} ({UNITS[i]})", ylabel=f"{LABELS[j]} ({UNITS[j]})")
        fig.suptitle("T3-Figure 1 | Retained customer profiles: distributions and relationships", fontsize=14)
        figures["baseline"] = fig
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for ax, name in zip(axes, METHODS[1:3]):
            ax.boxplot([comparisons["versions"][name][c] for c in FEATURES],
                       tick_labels=["Orders", "Order value", "Distance"], showmeans=True)
            ax.set(title=name, ylabel="Transformed feature value")
            ax.grid(axis="y", alpha=.2)
        fig.suptitle("T3-Figure 2 | Scaling changes units and weighting; extremes remain", fontsize=14)
        figures["scaling"] = fig
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        x = profiles.mean_order_value
        log_x = comparisons["nonlinear_values"].mean_order_value
        axes[0].hist(x, bins="auto", color=colors[1], alpha=.85)
        axes[0].set(xlabel="Mean order value (AUD)", ylabel="Customers", title="Original")
        axes[1].hist(log_x, bins="auto", color=colors[0], alpha=.85)
        axes[1].set(xlabel="ln(1 + value / 1 AUD)", ylabel="Customers", title="Log1p")
        axes[2].scatter(x, log_x, color=colors[0], s=18, alpha=.5)
        axes[2].set(xlabel="Mean order value (AUD)", ylabel="ln(1 + value / 1 AUD)", title="Large gaps are compressed")
        fig.suptitle("T3-Figure 3 | Nonlinear alternative for spending", fontsize=14)
        figures["nonlinear"] = fig
        fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
        bottom = np.zeros(len(effects))
        for feature, label, color in zip(FEATURES, LABELS, colors):
            values = effects[f"{feature}_squared_distance_share"].to_numpy(dtype=float) * 100
            ax.bar(np.arange(len(effects)), np.nan_to_num(values), bottom=bottom, label=label, color=color)
            bottom += np.nan_to_num(values)
        ax.set(xticks=np.arange(len(effects)), xticklabels=["Original", "Z-score", "Min-max", "Log1p value\n+ Z-score"],
               ylim=(0, 105), ylabel="Share of aggregate squared distance (%)")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.18), ncol=3, frameon=False)
        if np.isnan(effects.filter(like="_share").to_numpy()).all():
            ax.text(.5, .5, "Undefined: every customer profile is identical", ha="center", transform=ax.transAxes)
        fig.suptitle("T3-Figure 4 | How the representation weights customer differences", fontsize=14)
        figures["distance_shares"] = fig
    return figures


def run_profiles(clean_dir, dataset_id="Retail", raw_dir=None):
    orders, counts, inputs, hashes = load_cleaned_orders(clean_dir, dataset_id, raw_dir)
    profiles = build_profiles(orders)
    comparisons = compare_transformations(profiles)
    effects = distance_effects(comparisons["versions"])
    return {"orders": orders, "batch_counts": counts, "input_checks": inputs,
            "profiles": profiles, "baseline_summary": describe_features(profiles),
            "comparisons": comparisons, "distance_effects": effects,
            "recommendations": recommendation_table(profiles, comparisons, effects),
            "interpretation": interpretation(profiles, comparisons, effects),
            "checks": analysis_checks(orders, profiles, comparisons, hashes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", type=Path, required=True, help="Directory containing the final three *_clean.csv files")
    parser.add_argument("--group-id", default="Retail")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Optional original package directory for retained-ID checks")
    args = parser.parse_args()
    try:
        result = run_profiles(args.clean_dir, args.dataset_id, args.raw_dir)
    except (FileNotFoundError, ProfileInputError) as exc:
        parser.exit(2, f"Customer profiling cannot run: {exc}\n")
    for name in ("batch_counts", "baseline_summary", "distance_effects", "recommendations", "checks"):
        print(f"\n{name}\n{result[name].to_string()}")
    print("\n" + "\n".join(result["interpretation"]))
    print("\nAll features remain in memory. Cleaned CSVs and audit log were not modified.")


if __name__ == "__main__":
    main()
