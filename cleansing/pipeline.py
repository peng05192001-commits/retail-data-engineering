"""Integration entry points; no cleaning decision is inferred from saved outputs."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import json

import numpy as np
import pandas as pd


def run_dirty(data_dir, output_dir):
    from dirty_adapter import run
    return run(data_dir, output_dir)


def run_missing(data_dir, output_dir):
    from missing_adapter import run
    return run(data_dir, output_dir)


def run_outlier(data_dir, output_dir, dataset_id="Retail"):
    from outliers import run
    log = io.StringIO()
    with redirect_stdout(log):
        result = run(data_dir, output_dir, dataset_id)
    result["execution_log"] = log.getvalue()
    dirty = json.loads((Path(output_dir) / "dirty_validation.json").read_text(encoding="utf-8"))
    comparisons = []
    for season, row in result["coefficients"].iterrows():
        before = np.asarray(dirty["evidence"]["seasonal_models"][season]["coefficients"], dtype=float)
        after = row.to_numpy(dtype=float)
        difference = float(np.max(np.abs(before - after)))
        if difference > 1e-10:
            raise ValueError(f"Dirty / Outlier diagnostic model disagreement: {season}")
        comparisons.append({"season": season, "max_abs_coefficient_difference": difference, "passed": True})
    result["dirty_model_alignment"] = pd.DataFrame(comparisons)
    missing = json.loads((Path(output_dir) / "missing_validation.json").read_text(encoding="utf-8"))
    prices1 = dirty["evidence"]["price_inference"]["prices"]
    prices2 = missing["evidence"]["price_inference"]["prices"]
    if prices1 != prices2:
        raise ValueError("Dirty / Missing current-price estimates disagree")
    result["current_price_alignment"] = {"products": len(prices1), "passed": True}
    return result


def initial_investigation(data_dir, dataset_id="Retail"):
    rows, fields, categories, distributions, arithmetic = [], [], [], [], []
    sets = []
    for batch in ("dirty", "missing", "outlier"):
        frame = pd.read_csv(Path(data_dir) / f"{dataset_id}_{batch}_data.csv", dtype=str, keep_default_na=False)
        ids = set(frame.order_id)
        sets.append(ids)
        rows.append({"batch": batch, "rows": len(frame), "columns": len(frame.columns),
                     "distinct_order_ids": len(ids), "distinct_customers": frame.customer_id.nunique(),
                     "empty_cells": int(frame.eq("").sum().sum())})
        for field in frame:
            fields.append({"batch": batch, "field": field, "dtype_as_read": str(frame[field].dtype),
                           "empty_cells": int(frame[field].eq("").sum()), "distinct_values": frame[field].nunique()})
        for field in ("season", "nearest_warehouse", "is_expedited_delivery", "is_happy_customer"):
            categories.append({"batch": batch, "field": field,
                               "counts_including_empty": str(frame[field].value_counts().to_dict())})
        for field in ("order_price", "delivery_charges", "distance_to_nearest_warehouse"):
            numeric = pd.to_numeric(frame[field].replace("", np.nan), errors="coerce")
            distributions.append({"batch": batch, "field": field, "observed_numeric": int(numeric.notna().sum()),
                                  "minimum": numeric.min(), "median": numeric.median(), "maximum": numeric.max(),
                                  "unparseable_nonempty": int((numeric.isna() & frame[field].ne("")).sum())})
        money = frame[["order_price", "coupon_discount", "delivery_charges", "order_total"]].apply(pd.to_numeric, errors="coerce")
        complete = money.dropna()
        expected = (complete.order_price * (1 - complete.coupon_discount / 100) + complete.delivery_charges).round(2)
        arithmetic.append({"batch": batch, "complete_arithmetic_rows": len(complete),
                           "total_disagreements_over_one_cent": int((abs(complete.order_total - expected) > .01 + 1e-9).sum())})
    overlap = sum(len(a & b) for i, a in enumerate(sets) for b in sets[i + 1:])
    if overlap:
        raise ValueError("Original order IDs overlap across batches")
    return {"Initial batch dimensions, keys and missingness": pd.DataFrame(rows),
            "Field-level raw schema and missingness": pd.DataFrame(fields),
            "Raw categories": pd.DataFrame(categories),
            "Raw numeric distributions": pd.DataFrame(distributions),
            "Raw total-arithmetic diagnostics (not field attribution)": pd.DataFrame(arithmetic)}
