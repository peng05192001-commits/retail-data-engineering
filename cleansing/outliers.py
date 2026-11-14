"""Outlier calculations, extracted from the delivered notebook.
Integration adaptation: configurable paths, callable function and captured evidence.
The seasonal fitting, residual threshold and retention decisions are unchanged.
"""
from pathlib import Path

def run(data_dir, output_dir, dataset_id="Retail"):
    evidence, figures = [], []
    def display(value):
        evidence.append({"cell": section, "value": value})
    for batch in ("dirty", "missing"):
        if not (Path(output_dir) / f"{dataset_id}_{batch}_clean.csv").is_file():
            raise FileNotFoundError(f"Generate the {batch} cleaned output before cross-file validation.")

    # Original Outlier code cell 0.
    section = 0
    import pandas as pd

    DATASET_ID = dataset_id

    # Paths are relative to the folder containing this notebook.
    DATA_DIR = Path(data_dir)
    OUTPUT_DIR = Path(output_dir)
    OUTLIER_OUTPUT = OUTPUT_DIR / f"{DATASET_ID}_outlier_clean.csv"
    AUDIT_OUTPUT = OUTPUT_DIR / "audit_log_outlier.csv"

    missing_raw = pd.read_csv(
        DATA_DIR / f"{DATASET_ID}_missing_data.csv",
        dtype={"order_id": str, "customer_id": str},
        keep_default_na=False
    )

    outlier_raw = pd.read_csv(
        DATA_DIR / f"{DATASET_ID}_outlier_data.csv",
        dtype={"order_id": str, "customer_id": str},
        keep_default_na=False
    )

    print("Missing data shape:", missing_raw.shape)
    print("Outlier data shape:", outlier_raw.shape)

    display(outlier_raw.head())

    # Original Outlier code cell 1.
    section = 1
    MODEL_FIELDS = [
        "season",
        "distance_to_nearest_warehouse",
        "is_expedited_delivery",
        "is_happy_customer",
        "delivery_charges"
    ]

    # Select records with all model fields observed.
    # The missing-data batch's observed values are correct under the data contract.
    model_data = missing_raw[MODEL_FIELDS].replace("", pd.NA).dropna().copy()

    # Convert numeric fields.
    for field in ["distance_to_nearest_warehouse", "delivery_charges"]:
        model_data[field] = pd.to_numeric(model_data[field], errors="raise")

    # Encode booleans consistently as 0/1.
    boolean_map = {"true": 1, "false": 0, "1": 1, "0": 0}

    for field in ["is_expedited_delivery", "is_happy_customer"]:
        encoded = model_data[field].astype(str).str.lower().map(boolean_map)
        if encoded.isna().any():
            raise ValueError(f"Invalid boolean value in {field}")
        model_data[field] = encoded.astype(int)

    print("Eligible model records:", len(model_data))

    display(
        model_data.groupby("season")
        .size()
        .reindex(["Summer", "Autumn", "Winter", "Spring"])
        .rename("eligible_records")
        .to_frame()
    )

    # Original Outlier code cell 2.
    section = 2
    import numpy as np

    SEASONS = ["Summer", "Autumn", "Winter", "Spring"]
    SEED = 5196

    PREDICTORS = [
        "distance_to_nearest_warehouse",
        "is_expedited_delivery",
        "is_happy_customer"
    ]

    seasonal_models = {}
    evaluation_rows = []
    coefficient_rows = []

    for season in SEASONS:
        season_data = model_data.loc[model_data["season"] == season]

        # Include an intercept and three predictors.
        X = np.column_stack([
            np.ones(len(season_data)),
            season_data[PREDICTORS].to_numpy(dtype=float)
        ])
        y = season_data["delivery_charges"].to_numpy(dtype=float)

        # Reproduce Dirty's seasonal 80/20 split.
        indices = np.random.default_rng(SEED).permutation(len(season_data))
        split = int(0.8 * len(season_data))
        train_idx = indices[:split]
        test_idx = indices[split:]

        coefficients, _, rank, _ = np.linalg.lstsq(
            X[train_idx], y[train_idx], rcond=None
        )

        if rank != X.shape[1]:
            raise ValueError(f"Rank-deficient training matrix for {season}")

        # Evaluate only on observations excluded from fitting.
        predictions = X[test_idx] @ coefficients
        residuals = y[test_idx] - predictions
        total_variation = np.sum(
            (y[test_idx] - y[test_idx].mean()) ** 2
        )

        if total_variation == 0:
            raise ValueError(f"Undefined held-out R-squared for {season}")

        seasonal_models[season] = coefficients

        evaluation_rows.append({
            "season": season,
            "training_n": len(train_idx),
            "heldout_n": len(test_idx),
            "heldout_r2": 1 - np.sum(residuals ** 2) / total_variation,
            "heldout_mae_aud": np.mean(np.abs(residuals)),
            "heldout_max_error_aud": np.max(np.abs(residuals))
        })

        coefficient_rows.append({
            "season": season,
            "intercept": coefficients[0],
            "distance_coefficient": coefficients[1],
            "expedited_coefficient": coefficients[2],
            "happy_coefficient": coefficients[3]
        })

    evaluation_table = pd.DataFrame(evaluation_rows).set_index("season")
    coefficient_table = pd.DataFrame(coefficient_rows).set_index("season")

    display(evaluation_table.round(9))
    display(coefficient_table.round(6))

    # Original Outlier code cell 3.
    section = 3
    # Work on a separate diagnostic copy.
    outlier_analysis = outlier_raw.copy()

    for field in ["distance_to_nearest_warehouse", "delivery_charges"]:
        outlier_analysis[field] = pd.to_numeric(
            outlier_analysis[field], errors="raise"
        )

    for field in ["is_expedited_delivery", "is_happy_customer"]:
        encoded = (
            outlier_analysis[field]
            .astype(str)
            .str.lower()
            .map(boolean_map)
        )
        if encoded.isna().any():
            raise ValueError(f"Invalid boolean value in {field}")
        outlier_analysis[field] = encoded.astype(int)

    if not outlier_analysis["season"].isin(SEASONS).all():
        raise ValueError("Unexpected season value")

    outlier_analysis["predicted_delivery"] = np.nan

    for season in SEASONS:
        mask = outlier_analysis["season"] == season
        X = np.column_stack([
            np.ones(mask.sum()),
            outlier_analysis.loc[mask, PREDICTORS].to_numpy(dtype=float)
        ])

        outlier_analysis.loc[mask, "predicted_delivery"] = (
            X @ seasonal_models[season]
        )

    # Positive residuals indicate charges above the model prediction.
    outlier_analysis["residual"] = (
        outlier_analysis["delivery_charges"]
        - outlier_analysis["predicted_delivery"]
    )
    outlier_analysis["absolute_residual"] = (
        outlier_analysis["residual"].abs()
    )

    display(
        outlier_analysis.groupby("season")["absolute_residual"]
        .agg(["count", "median", "max"])
        .round(6)
    )

    # Explore threshold sensitivity without removing any records.
    threshold_comparison = pd.DataFrame({
        "threshold_aud": [0.01, 0.02, 0.05, 0.10, 0.50, 1.00]
    })
    threshold_comparison["flagged_orders"] = [
        int((outlier_analysis["absolute_residual"] > threshold).sum())
        for threshold in threshold_comparison["threshold_aud"]
    ]

    display(threshold_comparison)

    # Original Outlier code cell 4.
    section = 4
    import matplotlib.pyplot as plt

    THRESHOLD_AUD = 0.02

    # This flag is for diagnosis; no records are removed here.
    outlier_analysis["is_outlier"] = (
        outlier_analysis["absolute_residual"] > THRESHOLD_AUD
    )

    flagged = outlier_analysis["is_outlier"]
    retained = ~flagged

    decision_summary = pd.DataFrame({
        "measure": [
            "Flagged orders",
            "Retained orders",
            "Maximum retained absolute residual (AUD)",
            "Minimum flagged absolute residual (AUD)"
        ],
        "value": [
            int(flagged.sum()),
            int(retained.sum()),
            outlier_analysis.loc[retained, "absolute_residual"].max(),
            outlier_analysis.loc[flagged, "absolute_residual"].min()
        ]
    })

    display(decision_summary)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for mask, label, color in [
        (retained, "Retained candidate", "steelblue"),
        (flagged, "Flagged candidate", "crimson")
    ]:
        rows = outlier_analysis.loc[mask]

        axes[0].scatter(
            rows["predicted_delivery"],
            rows["delivery_charges"],
            label=label, color=color, alpha=0.7, s=25
        )
        axes[1].scatter(
            rows["predicted_delivery"],
            rows["residual"],
            label=label, color=color, alpha=0.7, s=25
        )

    limits = [
        min(outlier_analysis["predicted_delivery"].min(),
            outlier_analysis["delivery_charges"].min()),
        max(outlier_analysis["predicted_delivery"].max(),
            outlier_analysis["delivery_charges"].max())
    ]

    axes[0].plot(limits, limits, "--", color="black", label="Observed = predicted")
    axes[0].set_xlabel("Predicted delivery charge (AUD)")
    axes[0].set_ylabel("Observed delivery charge (AUD)")
    axes[0].set_title("Observed versus predicted charges")

    axes[1].axhline(THRESHOLD_AUD, linestyle="--", color="black")
    axes[1].axhline(-THRESHOLD_AUD, linestyle="--", color="black")
    axes[1].axhline(0, color="grey", linewidth=0.8)
    axes[1].set_yscale("symlog", linthresh=THRESHOLD_AUD)
    axes[1].set_xlabel("Predicted delivery charge (AUD)")
    axes[1].set_ylabel("Residual (AUD; symmetric log scale)")
    axes[1].set_title("Residuals and the ±0.02 AUD threshold")

    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)

    plt.tight_layout()
    figures.append(fig)
    plt.close(fig)

    # Original Outlier code cell 6.
    section = 6
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    AUDIT_COLUMNS = [
        "source_file", "record_key", "field_name", "issue_type",
        "original_value", "corrected_value", "method",
        "evidence", "confidence"
    ]

    # Ensure diagnostic flags still correspond to the original rows.
    assert outlier_analysis.index.equals(outlier_raw.index)
    assert outlier_raw["order_id"].is_unique
    assert outlier_analysis["order_id"].equals(outlier_raw["order_id"])

    flagged = outlier_analysis["absolute_residual"] > THRESHOLD_AUD

    # Select original rows so diagnostic columns and encoded values are excluded.
    outlier_clean = outlier_raw.loc[~flagged].copy()

    audit_entries = []

    for index in outlier_analysis.index[flagged]:
        diagnostic = outlier_analysis.loc[index]
        original = outlier_raw.loc[index]
        season = original["season"]
        evaluation = evaluation_table.loc[season]

        audit_entries.append({
            "source_file": f"{DATASET_ID}_outlier_data.csv",
            "record_key": original["order_id"],
            "field_name": "delivery_charges",
            "issue_type": "outlier_row",
            "original_value": str(original["delivery_charges"]),
            "corrected_value": "ROW_REMOVED",
            "method": "seasonal_linear_model_absolute_residual",
            "evidence": (
                f"season={season}; "
                f"predicted_aud={diagnostic['predicted_delivery']:.9f}; "
                f"residual_aud={diagnostic['residual']:.9f}; "
                f"absolute_residual_threshold_aud={THRESHOLD_AUD:.2f}; "
                f"heldout_n={int(evaluation['heldout_n'])}; "
                f"heldout_r2={evaluation['heldout_r2']:.9f}; "
                f"heldout_mae_aud={evaluation['heldout_mae_aud']:.9f}; "
                "see seasonal model evaluation, residual plots "
                "and threshold comparison"
            ),
            "confidence": "high"
        })

    audit_outlier = pd.DataFrame(audit_entries, columns=AUDIT_COLUMNS)

    outlier_clean.to_csv(OUTLIER_OUTPUT, index=False, encoding="utf-8")
    audit_outlier.to_csv(AUDIT_OUTPUT, index=False, encoding="utf-8")

    print("Retained orders:", len(outlier_clean))
    print("Removed orders:", int(flagged.sum()))
    print("Audit entries:", len(audit_outlier))
    print("Output folder:", OUTPUT_DIR)

    display(audit_outlier.head())

    # Original Outlier code cell 7.
    section = 7
    # Read source and exported files as strings to preserve identifiers.
    source_check = pd.read_csv(
        DATA_DIR / f"{DATASET_ID}_outlier_data.csv",
        dtype=str,
        keep_default_na=False
    )

    clean_check = pd.read_csv(
        OUTLIER_OUTPUT, dtype=str, keep_default_na=False
    )

    audit_check = pd.read_csv(
        AUDIT_OUTPUT, dtype=str, keep_default_na=False
    )

    EXPECTED_COLUMNS = [
        "order_id", "customer_id", "date", "nearest_warehouse",
        "shopping_cart", "order_price", "delivery_charges",
        "customer_lat", "customer_long", "coupon_discount",
        "order_total", "season", "is_expedited_delivery",
        "distance_to_nearest_warehouse",
        "latest_customer_review", "is_happy_customer"
    ]

    NUMERIC_FIELDS = [
        "order_price", "delivery_charges", "customer_lat",
        "customer_long", "coupon_discount", "order_total",
        "distance_to_nearest_warehouse"
    ]

    validation_results = []

    def record_check(name, passed, observed):
        validation_results.append({
            "check": name,
            "passed": bool(passed),
            "observed": observed
        })

    record_check(
        "Exact output schema and column order",
        clean_check.columns.tolist() == EXPECTED_COLUMNS,
        f"{len(clean_check.columns)} columns"
    )

    record_check(
        "Exact audit schema and column order",
        audit_check.columns.tolist() == AUDIT_COLUMNS,
        f"{len(audit_check.columns)} columns"
    )

    source_ids = set(source_check["order_id"])
    retained_ids = set(clean_check["order_id"])
    removed_ids = source_ids - retained_ids
    logged_ids = set(audit_check["record_key"])

    record_check(
        "Unique, nonempty retained order IDs",
        clean_check["order_id"].is_unique
        and clean_check["order_id"].ne("").all(),
        f"{len(clean_check)} rows; {len(retained_ids)} unique IDs"
    )

    record_check(
        "Retained IDs belong to the original batch",
        retained_ids <= source_ids,
        f"{len(retained_ids - source_ids)} unexpected IDs"
    )

    record_check(
        "Complete exported values",
        not clean_check.eq("").any().any(),
        f"{int(clean_check.eq('').sum().sum())} empty cells"
    )

    expected_removed_ids = set(
        outlier_analysis.loc[
            outlier_analysis["absolute_residual"] > THRESHOLD_AUD,
            "order_id"
        ]
    )

    record_check(
        "Removed IDs agree with the residual decision rule",
        removed_ids == expected_removed_ids,
        f"{len(removed_ids)} removed IDs"
    )

    record_check(
        "One audit entry per removed order",
        logged_ids == removed_ids
        and audit_check["record_key"].is_unique,
        f"{len(audit_check)} entries; {len(logged_ids)} unique IDs"
    )

    # Compare retained values with the source, allowing equivalent numeric formatting.
    source_by_id = source_check.set_index("order_id")
    clean_by_id = clean_check.set_index("order_id")
    original_retained = source_by_id.loc[clean_by_id.index]

    changed_fields = {}
    for field in clean_by_id.columns:
        if field in NUMERIC_FIELDS:
            before = pd.to_numeric(original_retained[field], errors="raise")
            after = pd.to_numeric(clean_by_id[field], errors="raise")
            same = np.isclose(
                before.to_numpy(), after.to_numpy(),
                rtol=0, atol=1e-9
            )
        else:
            same = (
                original_retained[field].to_numpy()
                == clean_by_id[field].to_numpy()
            )

        if not same.all():
            changed_fields[field] = int((~same).sum())

    record_check(
        "Original values preserved in retained rows",
        not changed_fields,
        str(changed_fields)
    )

    audit_metadata_ok = (
        audit_check["source_file"].eq(f"{DATASET_ID}_outlier_data.csv").all()
        and audit_check["field_name"].eq("delivery_charges").all()
        and audit_check["issue_type"].eq("outlier_row").all()
        and audit_check["corrected_value"].eq("ROW_REMOVED").all()
        and audit_check["method"].ne("").all()
        and audit_check["evidence"].ne("").all()
        and audit_check["confidence"].isin(["high", "medium", "low"]).all()
    )

    record_check(
        "Audit metadata follows the required conventions",
        audit_metadata_ok,
        f"{len(audit_check)} entries checked"
    )

    original_charges = pd.to_numeric(
        source_by_id.loc[audit_check["record_key"], "delivery_charges"],
        errors="raise"
    ).to_numpy()

    logged_charges = pd.to_numeric(
        audit_check["original_value"], errors="raise"
    ).to_numpy()

    record_check(
        "Logged original charges agree with source values",
        np.isclose(
            original_charges, logged_charges, rtol=0, atol=1e-9
        ).all(),
        f"{len(logged_charges)} original charges checked"
    )

    # Recalculate residuals from the exported values.
    export_residuals = []

    for season in SEASONS:
        rows = clean_check.loc[clean_check["season"] == season]

        X = np.column_stack([
            np.ones(len(rows)),
            pd.to_numeric(
                rows["distance_to_nearest_warehouse"], errors="raise"
            ).to_numpy(),
            rows["is_expedited_delivery"].str.lower().map(boolean_map).to_numpy(),
            rows["is_happy_customer"].str.lower().map(boolean_map).to_numpy()
        ]).astype(float)

        charges = pd.to_numeric(
            rows["delivery_charges"], errors="raise"
        ).to_numpy()

        export_residuals.extend(charges - X @ seasonal_models[season])

    export_residuals = np.asarray(export_residuals)

    record_check(
        "All exported orders pass the residual rule",
        len(export_residuals) == len(clean_check)
        and np.isfinite(export_residuals).all()
        and (np.abs(export_residuals) <= THRESHOLD_AUD).all(),
        f"Maximum absolute residual: {np.abs(export_residuals).max():.9f} AUD"
    )

    validation_table = pd.DataFrame(validation_results)
    display(validation_table)

    if not validation_table["passed"].all():
        raise ValueError("Read-back validation failed; inspect the results above.")

    print("All read-back checks passed.")

    # Original Outlier code cell 8.
    section = 8
    import sys
    import json
    import hashlib
    import csv
    from types import SimpleNamespace

    DIRTY_DIR = Path(__file__).resolve().parent

    if not (DIRTY_DIR / "dirty.py").is_file():
        raise FileNotFoundError(f"Dirty helpers not found: {DIRTY_DIR}")

    if str(DIRTY_DIR) not in sys.path:
        sys.path.insert(0, str(DIRTY_DIR))

    from dirty import (
        calendar_date,
        season_for,
        boolean,
        nearest_location,
        make_sentiment_scorer
    )
    from monetary import infer_current_prices, cart_price, order_total

    # Load the supplied warehouse reference.
    with (DATA_DIR / "warehouses.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        warehouses = list(csv.DictReader(handle))

    # Verify and load the frozen sentiment resource.
    resource_contract = json.loads(
        (DATA_DIR / "resource_contract.json").read_text(encoding="utf-8")
    )

    lexicon_bytes = (
        DATA_DIR / resource_contract["vader_lexicon_relative_path"]
    ).read_bytes()

    if hashlib.sha256(lexicon_bytes).hexdigest() != (
        resource_contract["vader_lexicon_sha256"]
    ):
        raise ValueError("Frozen VADER resource checksum mismatch")

    sentiment_score = make_sentiment_scorer(
        SimpleNamespace(
            contract=resource_contract,
            lexicon_zip=lexicon_bytes
        )
    )

    # Outlier-batch carts and order prices are correct under the data contract.
    # Reproduce Dirty's 350-record price-training selection.
    source_records = source_check.to_dict("records")
    price_indices = np.random.default_rng(SEED).permutation(len(source_records))

    price_fit = infer_current_prices([
        source_records[int(i)] for i in price_indices[:350]
    ])
    unit_prices = price_fit.prices

    print("Shared helpers loaded.")
    print("Frozen VADER resource verified and loaded.")
    print("Identifiable products:", len(unit_prices))
    print("Price-training records:", price_fit.training_count)

    # Original Outlier code cell 9.
    section = 9
    # Independently check prices on the 150 orders excluded from price fitting.
    price_holdout_errors = []

    for index in price_indices[350:]:
        row = source_records[int(index)]
        price_holdout_errors.append(
            abs(cart_price(row["shopping_cart"], unit_prices)
                - float(row["order_price"]))
        )

    price_holdout_errors = np.asarray(price_holdout_errors)

    print("Price held-out records:", len(price_holdout_errors))
    print(
        "Maximum held-out cart-price error (AUD):",
        price_holdout_errors.max()
    )

    # Validate business rules using the exported file.
    RULE_NAMES = [
        "Finite numeric values and valid booleans",
        "Valid geographic coordinate ranges",
        "ISO calendar date",
        "Season agrees with date",
        "Nearest warehouse agrees with haversine calculation",
        "Distance agrees with haversine calculation",
        "Happiness agrees with frozen VADER rule",
        "Cart structure and current-price calculation",
        "Discounted total calculation"
    ]

    row_results = []
    evaluation_errors = []

    for row in clean_check.to_dict("records"):
        try:
            numbers = {field: float(row[field]) for field in NUMERIC_FIELDS}
            expedited = boolean(row["is_expedited_delivery"])
            happy = boolean(row["is_happy_customer"])

            parsed_date = calendar_date(row["date"])
            expected_warehouse, expected_distance = nearest_location(
                numbers["customer_lat"],
                numbers["customer_long"],
                warehouses
            )
            expected_happy = sentiment_score(
                row["latest_customer_review"]
            )["happy"]

            expected_price = cart_price(row["shopping_cart"], unit_prices)
            expected_total = order_total(
                numbers["order_price"],
                numbers["coupon_discount"],
                numbers["delivery_charges"]
            )

            outcomes = {
                "Finite numeric values and valid booleans":
                    np.isfinite(list(numbers.values())).all()
                    and expedited in (0, 1) and happy in (0, 1),

                "Valid geographic coordinate ranges":
                    -90 <= numbers["customer_lat"] <= 90
                    and -180 <= numbers["customer_long"] <= 180,

                "ISO calendar date":
                    row["date"] == parsed_date.isoformat(),

                "Season agrees with date":
                    row["season"] == season_for(parsed_date),

                "Nearest warehouse agrees with haversine calculation":
                    row["nearest_warehouse"] == expected_warehouse,

                "Distance agrees with haversine calculation":
                    abs(numbers["distance_to_nearest_warehouse"]
                        - expected_distance) <= 0.0001 + 1e-9,

                "Happiness agrees with frozen VADER rule":
                    happy == int(expected_happy),

                "Cart structure and current-price calculation":
                    abs(numbers["order_price"] - expected_price) <= 1e-9,

                "Discounted total calculation":
                    abs(numbers["order_total"] - expected_total) <= 1e-9
            }

        except (ValueError, KeyError, TypeError, OverflowError) as error:
            # An unevaluated row cannot be counted as passing.
            outcomes = {name: False for name in RULE_NAMES}
            evaluation_errors.append({
                "order_id": row["order_id"],
                "error": str(error)
            })

        row_results.append({"order_id": row["order_id"], **outcomes})

    business_details = pd.DataFrame(row_results)

    business_validation_table = pd.DataFrame([
        {
            "check": name,
            "checked_rows": len(business_details),
            "failed_rows": int((~business_details[name]).sum()),
            "passed": bool(business_details[name].all())
        }
        for name in RULE_NAMES
    ])

    display(business_validation_table)

    if evaluation_errors:
        display(pd.DataFrame(evaluation_errors))

    failed_rows = business_details.loc[
        ~business_details[RULE_NAMES].all(axis=1)
    ]

    if not failed_rows.empty:
        display(failed_rows)

    if (
        not business_validation_table["passed"].all()
        or not np.isfinite(price_holdout_errors).all()
        or (price_holdout_errors > 1e-9).any()
    ):
        raise ValueError("Business validation failed; inspect the results above.")

    print("All business-rule checks passed.")

    # Original Outlier code cell 12.
    section = 12
    # Initial investigation of the outlier batch: schema, completeness, keys and categories.
    outlier_eda = pd.DataFrame({
        "dtype_as_read": outlier_raw.dtypes.astype(str),
        "empty_cells": outlier_raw.eq("").sum(),
        "distinct_values": outlier_raw.nunique()
    })
    display(outlier_eda)

    print("Column order matches the 16-column data contract:",
          outlier_raw.columns.tolist() == EXPECTED_COLUMNS)
    print("Unique order IDs:", outlier_raw["order_id"].is_unique)
    print("Total empty cells:", int(outlier_raw.eq("").sum().sum()))
    print("Season values:", outlier_raw["season"].value_counts().to_dict())
    print("Warehouse values:", outlier_raw["nearest_warehouse"].value_counts().to_dict())

    # Order IDs must be distinct across the three batches.
    other_ids = set(missing_raw["order_id"]) | set(
        pd.read_csv(DATA_DIR / f"{DATASET_ID}_dirty_data.csv", dtype=str,
                    keep_default_na=False)["order_id"]
    )
    print("Outlier IDs also present in another batch:",
          len(set(outlier_raw["order_id"]) & other_ids))

    display(outlier_analysis.groupby("season")["delivery_charges"]
            .describe().reindex(SEASONS).round(2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].boxplot(
        [outlier_analysis.loc[outlier_analysis["season"] == s, "delivery_charges"]
         for s in SEASONS]
    )
    axes[0].set_xticks(range(1, len(SEASONS) + 1), SEASONS)
    axes[0].set_title("Raw delivery charges by season (outlier batch)")
    axes[0].set_ylabel("Delivery charge (AUD)")

    for value, label, colour in [(0, "Standard", "steelblue"), (1, "Expedited", "darkorange")]:
        rows = outlier_analysis.loc[outlier_analysis["is_expedited_delivery"] == value]
        axes[1].scatter(rows["distance_to_nearest_warehouse"], rows["delivery_charges"],
                        s=15, alpha=0.6, color=colour, label=label)
    axes[1].set_xlabel("Distance to nearest warehouse (km)")
    axes[1].set_ylabel("Delivery charge (AUD)")
    axes[1].set_title("Charge depends on distance and expedited status")
    axes[1].legend()
    plt.tight_layout()
    figures.append(fig)
    plt.close(fig)

    # Original Outlier code cell 15.
    section = 15
    # (1) order_total arithmetic is independent of the model and threshold.
    # The outlier batch only corrupts delivery_charges, so an injected charge
    # should leave order_total consistent with the expected charge instead.
    def total_matches(row, charge):
        expected = order_total(float(row["order_price"]),
                               float(row["coupon_discount"]), charge)
        return abs(expected - float(row["order_total"])) <= 0.01 + 1e-9

    corroboration = pd.DataFrame({
        "group": np.where(outlier_analysis["is_outlier"], "Flagged", "Retained"),
        "total_matches_observed_charge": [
            total_matches(row, float(row["delivery_charges"]))
            for _, row in outlier_raw.iterrows()
        ],
        "total_matches_predicted_charge": [
            total_matches(row, round(pred, 2))
            for (_, row), pred in zip(outlier_raw.iterrows(),
                                      outlier_analysis["predicted_delivery"])
        ]
    })
    display(corroboration.groupby("group").agg(
        orders=("group", "size"),
        total_matches_observed_charge=("total_matches_observed_charge", "sum"),
        total_matches_predicted_charge=("total_matches_predicted_charge", "sum")
    ))

    # (2) Raw-charge screens ignore the predictors and misclassify orders.
    charges = outlier_analysis["delivery_charges"]
    q1, q3 = charges.quantile([0.25, 0.75])
    iqr_flag = (charges < q1 - 1.5 * (q3 - q1)) | (charges > q3 + 1.5 * (q3 - q1))
    z_flag = ((charges - charges.mean()) / charges.std()).abs() > 3
    residual_flag = outlier_analysis["is_outlier"]

    display(pd.DataFrame([
        {"screen": name,
         "flagged": int(flag.sum()),
         "agree_with_residual_rule": int((flag & residual_flag).sum()),
         "would_wrongly_remove": int((flag & ~residual_flag).sum()),
         "would_miss": int((~flag & residual_flag).sum())}
        for name, flag in [("Global IQR (1.5x)", iqr_flag),
                           ("Global |z| > 3", z_flag),
                           ("Seasonal residual > AUD 0.02", residual_flag)]
    ]))

    print("Lowest flagged charge (AUD):", charges[residual_flag].min())
    print("Highest retained charge (AUD):", charges[~residual_flag].max())
    print("Retained orders charged more than the lowest flagged charge:",
          int((charges[~residual_flag] > charges[residual_flag].min()).sum()))

    # (3) Refit on the retained outlier-batch rows; agreement with the
    # missing-batch models shows the training evidence was not unrepresentative.
    refit_rows = []
    for season in SEASONS:
        rows = outlier_analysis.loc[(outlier_analysis["season"] == season) & ~residual_flag]
        X = np.column_stack([np.ones(len(rows)), rows[PREDICTORS].to_numpy(dtype=float)])
        refit = np.linalg.lstsq(X, rows["delivery_charges"].to_numpy(dtype=float), rcond=None)[0]
        refit_rows.append({"season": season, "retained_n": len(rows),
                           "max_abs_coefficient_difference":
                               np.abs(refit - seasonal_models[season]).max()})
    display(pd.DataFrame(refit_rows).set_index("season").round(6))

    # Original Outlier code cell 20.
    section = 20
    VALUE_KIND = {field: "numeric" for field in NUMERIC_FIELDS}
    VALUE_KIND.update({"is_expedited_delivery": "boolean", "is_happy_customer": "boolean"})

    def same_value(field, a, b):
        """Equality in meaning: 8 == 8.00, True == true; other fields compare as text."""
        if a == "" or b == "":
            return a == b
        if VALUE_KIND.get(field) == "numeric":
            return abs(float(a) - float(b)) <= 1e-9
        if VALUE_KIND.get(field) == "boolean":
            return boolean(a) == boolean(b)
        return a == b

    def field_differences(source, clean, source_file):
        """Every (order_id, field) whose meaning differs between input and output."""
        src = source.set_index("order_id")
        out = clean.set_index("order_id")
        diffs = []
        for order_id in out.index:
            for field in EXPECTED_COLUMNS[1:]:
                before, after = src.at[order_id, field], out.at[order_id, field]
                if not same_value(field, before, after):
                    diffs.append({"source_file": source_file, "record_key": order_id,
                                  "field_name": field, "before": before, "after": after})
        return pd.DataFrame(diffs, columns=["source_file", "record_key",
                                            "field_name", "before", "after"])

    def business_rule_failures(clean):
        """Row-level business rules, including the seasonal residual screen."""
        failures = {name: 0 for name in RULE_NAMES + ["Seasonal residual within threshold"]}
        for row in clean.to_dict("records"):
            numbers = {f: float(row[f]) for f in NUMERIC_FIELDS}
            parsed = calendar_date(row["date"])
            warehouse, distance = nearest_location(numbers["customer_lat"],
                                                   numbers["customer_long"], warehouses)
            x = np.array([1, numbers["distance_to_nearest_warehouse"],
                          boolean(row["is_expedited_delivery"]), boolean(row["is_happy_customer"])])
            outcomes = {
                "Finite numeric values and valid booleans": np.isfinite(list(numbers.values())).all(),
                "Valid geographic coordinate ranges": -90 <= numbers["customer_lat"] <= 90
                    and -180 <= numbers["customer_long"] <= 180,
                "ISO calendar date": row["date"] == parsed.isoformat(),
                "Season agrees with date": row["season"] == season_for(parsed),
                "Nearest warehouse agrees with haversine calculation": row["nearest_warehouse"] == warehouse,
                "Distance agrees with haversine calculation":
                    abs(numbers["distance_to_nearest_warehouse"] - distance) <= 0.0001 + 1e-9,
                "Happiness agrees with frozen VADER rule":
                    boolean(row["is_happy_customer"]) == int(sentiment_score(row["latest_customer_review"])["happy"]),
                "Cart structure and current-price calculation":
                    abs(numbers["order_price"] - cart_price(row["shopping_cart"], unit_prices)) <= 1e-9,
                "Discounted total calculation":
                    abs(numbers["order_total"] - order_total(numbers["order_price"],
                        numbers["coupon_discount"], numbers["delivery_charges"])) <= 1e-9,
                "Seasonal residual within threshold":
                    abs(numbers["delivery_charges"] - x @ seasonal_models[row["season"]]) <= THRESHOLD_AUD,
            }
            for name, ok in outcomes.items():
                failures[name] += int(not ok)
        return failures

    def reconcile_audit(source, clean, audit, source_file, row_rule):
        """Compare actual input/output differences with the audit entries for one file."""
        entries = audit.loc[audit["source_file"] == source_file]
        results = []
        if row_rule == "all_rows":
            diffs = field_differences(source, clean, source_file)
            actual = set(zip(diffs["record_key"], diffs["field_name"]))
            logged = set(zip(entries["record_key"], entries["field_name"]))
            merged = entries.merge(diffs, on=["source_file", "record_key", "field_name"])
            values_ok = all(
                same_value(r.field_name, "" if r.original_value == "<MISSING>" else r.original_value, r.before)
                and same_value(r.field_name, r.corrected_value, r.after)
                for r in merged.itertuples()
            )
            results += [
                ("Row count and ID set retained",
                 len(clean) == len(source) and set(clean["order_id"]) == set(source["order_id"]),
                 f"{len(clean)} rows"),
                ("Every actual change is logged", actual <= logged, f"{len(actual - logged)} unlogged"),
                ("Every log entry is an actual change", logged <= actual, f"{len(logged - actual)} not in output"),
                ("Logged original/corrected values match files", values_ok, f"{len(merged)} entries compared"),
            ]
        else:  # removed rows only
            removed = set(source["order_id"]) - set(clean["order_id"])
            diffs = field_differences(source, clean, source_file)
            results += [
                ("Removed IDs equal logged ROW_REMOVED IDs",
                 removed == set(entries.loc[entries["corrected_value"] == "ROW_REMOVED", "record_key"]),
                 f"{len(removed)} removed"),
                ("Retained + removed = input IDs",
                 len(clean) + len(removed) == len(source) and removed.isdisjoint(clean["order_id"]),
                 f"{len(clean)} + {len(removed)} = {len(source)}"),
                ("No field changed in retained rows", diffs.empty, f"{len(diffs)} changed cells"),
            ]
        results.append(("No duplicate audit entries",
                         not entries.duplicated(["record_key", "field_name"]).any(),
                         f"{len(entries)} entries"))
        return pd.DataFrame(results, columns=["check", "passed", "observed"]).assign(file=source_file)

    # Original Outlier code cell 21.
    section = 21
    DIRTY_DIR = OUTPUT_DIR
    MISSING_DIR = OUTPUT_DIR   # adjust when Missing delivers

    BATCHES = [
        # (input file, cleaned output, component audit log, row rule)
        (f"{DATASET_ID}_dirty_data.csv", DIRTY_DIR / f"{DATASET_ID}_dirty_clean.csv",
         DIRTY_DIR / "audit_log_dirty.csv", "all_rows"),
        (f"{DATASET_ID}_missing_data.csv", MISSING_DIR / f"{DATASET_ID}_missing_clean.csv",
         MISSING_DIR / "audit_log_missing.csv", "all_rows"),
        (f"{DATASET_ID}_outlier_data.csv", OUTLIER_OUTPUT, AUDIT_OUTPUT, "removed_rows"),
    ]

    read = lambda path: pd.read_csv(path, dtype=str, keep_default_na=False)

    available = [b for b in BATCHES if b[1].is_file() and b[2].is_file()]
    pending = [b[0] for b in BATCHES if b not in available]

    combined_audit = pd.concat([read(b[2]) for b in available], ignore_index=True)[AUDIT_COLUMNS]
    COMBINED_AUDIT_OUTPUT = OUTPUT_DIR / "audit_log.csv"
    combined_audit.to_csv(COMBINED_AUDIT_OUTPUT, index=False, encoding="utf-8")
    combined_check = read(COMBINED_AUDIT_OUTPUT)

    print("Pending batches (not yet validated):", pending or "none")
    print("Combined audit entries:", combined_check.groupby("source_file").size().to_dict())

    audit_level = pd.DataFrame([
        ("Audit schema and column order", combined_check.columns.tolist() == AUDIT_COLUMNS, "9 columns"),
        ("issue_type values valid",
         combined_check["issue_type"].isin(["dirty", "missing", "outlier_row"]).all(),
         str(combined_check["issue_type"].value_counts().to_dict())),
        ("confidence values valid",
         combined_check["confidence"].isin(["high", "medium", "low"]).all(),
         str(combined_check["confidence"].value_counts().to_dict())),
        ("method and evidence non-empty",
         combined_check[["method", "evidence"]].ne("").all().all(), f"{len(combined_check)} entries"),
    ], columns=["check", "passed", "observed"]).assign(file="audit_log.csv")

    file_level, rule_level = [audit_level], []
    for source_file, output_path, _, row_rule in available:
        source, clean = read(DATA_DIR / source_file), read(output_path)
        structure = pd.DataFrame([
            ("Exact schema and column order", clean.columns.tolist() == EXPECTED_COLUMNS, f"{clean.shape[1]} columns"),
            ("Unique order IDs, no unexpected IDs",
             clean["order_id"].is_unique and set(clean["order_id"]) <= set(source["order_id"]), f"{len(clean)} rows"),
            ("No empty cells", not clean.eq("").any().any(), f"{int(clean.eq('').sum().sum())} empty"),
        ], columns=["check", "passed", "observed"]).assign(file=output_path.name)
        file_level += [structure, reconcile_audit(source, clean, combined_check, source_file, row_rule)]
        rule_level.append(pd.Series(business_rule_failures(clean), name=output_path.name))

    cross_file_validation = pd.concat(file_level, ignore_index=True)[["file", "check", "passed", "observed"]]
    display(cross_file_validation)

    business_failures = pd.concat(rule_level, axis=1)
    business_failures.index.name = "failed rows per rule"
    display(business_failures)

    if not cross_file_validation["passed"].all() or business_failures.to_numpy().any():
        raise ValueError("Cross-file validation failed; inspect the tables above.")
    print("All available outputs passed; pending:", pending or "none")

    if pending:
        raise RuntimeError(f"Cross-file validation is incomplete: {pending}")
    return {
        "evidence": evidence, "figures": figures, "evaluation": evaluation_table,
        "coefficients": coefficient_table, "decisions": decision_summary,
        "thresholds": threshold_comparison, "outlier_analysis": outlier_analysis,
        "outlier_eda": outlier_eda, "audit": combined_check,
        "cross_file_validation": cross_file_validation,
        "business_failures": business_failures,
        "outlier_readback_checks": validation_table,
        "outlier_business_checks": business_validation_table,
        "price_holdout_errors": price_holdout_errors,
        "corroboration": corroboration.groupby("group").sum(),
        "refit_comparison": pd.DataFrame(refit_rows),
    }
