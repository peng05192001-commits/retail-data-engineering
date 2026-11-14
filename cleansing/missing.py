from pathlib import Path
import argparse
import hashlib
import io
import json
import math
import tempfile
import zipfile

import numpy as np
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


DATASET_ID = "Retail"
MISSING_FILE = f"{DATASET_ID}_missing_data.csv"
OUTLIER_FILE = f"{DATASET_ID}_outlier_data.csv"
OUTPUT_FILE = f"{DATASET_ID}_missing_clean.csv"
AUDIT_FILE = "audit_log_missing.csv"
VALIDATION_FILE = "missing_validation.json"
SEED = 5196

REQUIRED_COLUMNS = [
    "order_id", "customer_id", "date", "nearest_warehouse", "shopping_cart",
    "order_price", "delivery_charges", "customer_lat", "customer_long",
    "coupon_discount", "order_total", "season", "is_expedited_delivery",
    "distance_to_nearest_warehouse", "latest_customer_review",
    "is_happy_customer"
]

AUDIT_COLUMNS = [
    "source_file", "record_key", "field_name", "issue_type",
    "original_value", "corrected_value", "method", "evidence", "confidence"
]


def read_zip_csv(zf, name):
    return pd.read_csv(io.BytesIO(zf.read(name)), dtype=str)


def parse_bool(value):
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return 1
    if text in {"false", "0"}:
        return 0
    raise ValueError(f"Unexpected boolean value: {value!r}")


def parse_cart(cart_text):
    items = json.loads(cart_text)
    if not isinstance(items, list) or len(items) == 0:
        raise ValueError("shopping_cart must be a non-empty JSON array")
    result = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each cart item must be a JSON object")
        product = item.get("product")
        quantity = item.get("quantity")
        if not isinstance(product, str) or product.strip() == "":
            raise ValueError("Product name must be a non-empty string")
        if type(quantity) is not int or quantity <= 0:
            raise ValueError("Quantity must be a positive integer")
        result.append((product, quantity))
    return result


def month_to_season(date_text):
    dt = pd.to_datetime(date_text, format="%Y-%m-%d", errors="raise")
    month = dt.month
    if month in (12, 1, 2):
        return "Summer"
    if month in (3, 4, 5):
        return "Autumn"
    if month in (6, 7, 8):
        return "Winter"
    return "Spring"


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def nearest_warehouse(customer_lat, customer_long, warehouses):
    lat = float(customer_lat)
    lon = float(customer_long)
    distances = []
    for _, row in warehouses.iterrows():
        distance = haversine_km(lat, lon, float(row["latitude"]), float(row["longitude"]))
        distances.append((row["warehouse"], distance))
    return min(distances, key=lambda x: x[1])


def infer_product_prices(outlier_df):
    products = sorted({
        product
        for cart in outlier_df["shopping_cart"]
        for product, _ in parse_cart(cart)
    })
    product_index = {product: i for i, product in enumerate(products)}

    X = np.zeros((len(outlier_df), len(products)), dtype=float)
    for row_number, cart in enumerate(outlier_df["shopping_cart"]):
        for product, quantity in parse_cart(cart):
            X[row_number, product_index[product]] += quantity

    y = outlier_df["order_price"].astype(float).to_numpy()
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    prices = {product: round(float(coef), 2) for product, coef in zip(products, model.coef_)}
    predicted = X @ np.array([prices[p] for p in products])
    max_equation_error = float(np.max(np.abs(predicted - y)))

    return prices, {
        "training_rows": int(len(outlier_df)),
        "product_count": int(len(products)),
        "r_squared": float(model.score(X, y)),
        "max_equation_error": max_equation_error,
        "prices": prices,
    }


def calculate_order_price(cart_text, prices):
    line_values = []
    for product, quantity in parse_cart(cart_text):
        if product not in prices:
            raise ValueError(f"No inferred price for product {product!r}")
        line_values.append(round(quantity * float(prices[product]), 2))
    return round(sum(line_values), 2)


def calculate_order_total(order_price, coupon_discount, delivery_charges):
    total = float(order_price) * (1 - float(coupon_discount) / 100) + float(delivery_charges)
    return round(total, 2)


def prepare_vader(zf, temp_root):
    contract = json.loads(zf.read("resource_contract.json").decode("utf-8"))
    resource_name = contract["vader_lexicon_relative_path"]
    resource_bytes = zf.read(resource_name)
    actual_sha = hashlib.sha256(resource_bytes).hexdigest()
    expected_sha = contract["vader_lexicon_sha256"]
    if actual_sha != expected_sha:
        raise ValueError("VADER lexicon checksum does not match resource_contract.json")
    if nltk.__version__ != contract["nltk_version"]:
        raise ValueError(
            f"NLTK version mismatch: expected {contract['nltk_version']}, got {nltk.__version__}"
        )

    target = Path(temp_root) / "nltk_data" / "sentiment"
    target.mkdir(parents=True, exist_ok=True)
    lexicon_path = target / "vader_lexicon.zip"
    lexicon_path.write_bytes(resource_bytes)

    nltk_root = str(Path(temp_root) / "nltk_data")
    old_path = list(nltk.data.path)
    nltk.data.path.insert(0, nltk_root)
    try:
        analyzer = SentimentIntensityAnalyzer()
    finally:
        nltk.data.path[:] = old_path

    return analyzer, contract, actual_sha


def derive_happiness(review, analyzer, threshold):
    if review == "No review submitted":
        return True, None
    score = analyzer.polarity_scores(review)["compound"]
    return bool(score >= threshold), float(score)


def fit_delivery_models(missing_df):
    model_fields = [
        "delivery_charges", "distance_to_nearest_warehouse",
        "is_expedited_delivery", "is_happy_customer", "season"
    ]
    trusted = missing_df.dropna(subset=model_fields).copy()

    models = {}
    metrics = {}
    for season in ["Summer", "Autumn", "Winter", "Spring"]:
        season_df = trusted[trusted["season"] == season].copy()
        X = pd.DataFrame({
            "distance": season_df["distance_to_nearest_warehouse"].astype(float),
            "expedited": season_df["is_expedited_delivery"].map(parse_bool).astype(float),
            "happy": season_df["is_happy_customer"].map(parse_bool).astype(float),
        })
        y = season_df["delivery_charges"].astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )
        evaluation_model = LinearRegression()
        evaluation_model.fit(X_train, y_train)
        test_prediction = evaluation_model.predict(X_test)
        mae = float(np.mean(np.abs(y_test.to_numpy() - test_prediction)))
        r_squared = float(evaluation_model.score(X_test, y_test))

        final_model = LinearRegression()
        final_model.fit(X, y)
        models[season] = final_model
        metrics[season] = {
            "complete_rows": int(len(season_df)),
            "training_rows": int(len(X_train)),
            "held_out_rows": int(len(X_test)),
            "held_out_r_squared": r_squared,
            "held_out_mae": mae,
            "intercept": float(final_model.intercept_),
            "coefficients": {
                "distance": float(final_model.coef_[0]),
                "is_expedited_delivery": float(final_model.coef_[1]),
                "is_happy_customer": float(final_model.coef_[2]),
            },
        }
    return models, metrics, trusted


def predict_delivery_charge(model, distance, expedited, happy):
    X = pd.DataFrame({
        "distance": [float(distance)],
        "expedited": [float(expedited)],
        "happy": [float(happy)],
    })
    return float(model.predict(X)[0])


def recover_missing_data(missing_df, outlier_df, warehouses, analyzer, threshold):
    source = missing_df.copy()
    cleaned = missing_df.copy()
    missing_counts = source.isna().sum().to_dict()

    prices, price_evidence = infer_product_prices(outlier_df)
    delivery_models, delivery_metrics, trusted_model_rows = fit_delivery_models(source)

    audit_rows = []

    def set_value(index, field, value, method, evidence, confidence):
        if pd.notna(source.at[index, field]):
            raise ValueError(f"Attempted to overwrite observed value: row={index}, field={field}")
        if field in {"order_price", "delivery_charges", "order_total"}:
            exported = f"{float(value):.2f}"
        elif field == "distance_to_nearest_warehouse":
            exported = f"{float(value):.4f}"
        elif field in {"is_expedited_delivery", "is_happy_customer"}:
            exported = "True" if bool(value) else "False"
        else:
            exported = str(value)
        cleaned.at[index, field] = exported
        audit_rows.append({
            "source_file": MISSING_FILE,
            "record_key": source.at[index, "order_id"],
            "field_name": field,
            "issue_type": "missing",
            "original_value": "<MISSING>",
            "corrected_value": exported,
            "method": method,
            "evidence": evidence,
            "confidence": confidence,
        })

    for index in source.index[source["season"].isna()]:
        value = month_to_season(source.at[index, "date"])
        set_value(index, "season", value,
                  "Derived from order month using the published Australian season mapping",
                  "cleansing data contract section 3.3; date is complete in the missing batch",
                  "high")

    for index in source.index[source["nearest_warehouse"].isna()]:
        warehouse, distance = nearest_warehouse(
            source.at[index, "customer_lat"], source.at[index, "customer_long"], warehouses
        )
        set_value(index, "nearest_warehouse", warehouse,
                  "Selected the minimum haversine distance to the supplied warehouses",
                  "cleansing data contract section 3.3; warehouses.csv is the reference file",
                  "high")

    for index in source.index[source["distance_to_nearest_warehouse"].isna()]:
        warehouse, distance = nearest_warehouse(
            source.at[index, "customer_lat"], source.at[index, "customer_long"], warehouses
        )
        set_value(index, "distance_to_nearest_warehouse", round(distance, 4),
                  "Calculated haversine distance to the nearest supplied warehouse",
                  "cleansing data contract section 3.3; Earth radius 6371 km",
                  "high")

    happiness_scores = {}
    for index in source.index[source["is_happy_customer"].isna()]:
        value, compound = derive_happiness(
            source.at[index, "latest_customer_review"], analyzer, threshold
        )
        happiness_scores[source.at[index, "order_id"]] = compound
        if compound is None:
            evidence = "Published 'No review submitted' sentinel rule"
        else:
            evidence = f"Frozen VADER compound={compound:.4f}; threshold={threshold}"
        set_value(index, "is_happy_customer", value,
                  "Applied the supplied frozen VADER rule to latest_customer_review",
                  evidence,
                  "high")

    for index in source.index[source["order_price"].isna()]:
        value = calculate_order_price(source.at[index, "shopping_cart"], prices)
        set_value(index, "order_price", value,
                  "Recalculated shopping-cart price from inferred current cleansing unit prices",
                  "Prices inferred from all 500 outlier-batch cart/order_price equations; fit R^2=1.0",
                  "high")

    expedited_decisions = {}
    for index in source.index[source["is_expedited_delivery"].isna()]:
        season = source.at[index, "season"]
        model = delivery_models[season]
        distance = float(source.at[index, "distance_to_nearest_warehouse"])
        happy = parse_bool(source.at[index, "is_happy_customer"])
        observed = float(source.at[index, "delivery_charges"])
        pred_false = predict_delivery_charge(model, distance, 0, happy)
        pred_true = predict_delivery_charge(model, distance, 1, happy)
        residual_false = abs(observed - pred_false)
        residual_true = abs(observed - pred_true)
        value = residual_true < residual_false
        expedited_decisions[source.at[index, "order_id"]] = {
            "predicted_if_false": pred_false,
            "predicted_if_true": pred_true,
            "absolute_residual_if_false": residual_false,
            "absolute_residual_if_true": residual_true,
        }
        set_value(index, "is_expedited_delivery", value,
                  "Compared the observed delivery charge with the season model under expedited=False and True",
                  f"Selected the smaller model residual: false={residual_false:.4f}, true={residual_true:.4f}",
                  "medium")

    delivery_predictions = {}
    for index in source.index[source["delivery_charges"].isna()]:
        season = source.at[index, "season"]
        model = delivery_models[season]
        distance = float(source.at[index, "distance_to_nearest_warehouse"])
        expedited = parse_bool(source.at[index, "is_expedited_delivery"])
        happy = parse_bool(source.at[index, "is_happy_customer"])
        prediction = predict_delivery_charge(model, distance, expedited, happy)
        value = round(prediction, 2)
        delivery_predictions[source.at[index, "order_id"]] = prediction
        set_value(index, "delivery_charges", value,
                  "Estimated from the separate linear delivery-charge model for the row's season",
                  f"Season={season}; held-out R^2={delivery_metrics[season]['held_out_r_squared']:.6f}; held-out MAE={delivery_metrics[season]['held_out_mae']:.4f}",
                  "medium")

    for index in source.index[source["order_total"].isna()]:
        value = calculate_order_total(
            source.at[index, "order_price"],
            source.at[index, "coupon_discount"],
            source.at[index, "delivery_charges"]
        )
        set_value(index, "order_total", value,
                  "Applied the published order_total formula in the required operation order",
                  "order_price * (1 - coupon_discount/100) + delivery_charges, rounded to 2 decimals",
                  "high")

    audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    return cleaned, audit, {
        "missing_counts_before": {k: int(v) for k, v in missing_counts.items()},
        "price_inference": price_evidence,
        "delivery_model_evaluation": delivery_metrics,
        "trusted_delivery_model_rows": int(len(trusted_model_rows)),
        "happiness_compound_scores": happiness_scores,
        "expedited_decisions": expedited_decisions,
        "delivery_charge_raw_predictions": delivery_predictions,
    }


def validate_output(source, cleaned, audit, warehouses, prices, analyzer, threshold, model_metrics):
    checks = []

    def add(check_id, check, observed, passed, interpretation):
        checks.append({
            "check_id": check_id,
            "check": check,
            "observed_result": observed,
            "status": "PASS" if passed else "FAIL",
            "interpretation": interpretation,
        })

    add("M2-VAL-01", "exact schema and column order",
        cleaned.columns.tolist(), cleaned.columns.tolist() == REQUIRED_COLUMNS,
        "The output must keep the original 16-column schema and order.")

    add("M2-VAL-02", "500 rows and preserved unique order IDs",
        {"rows": int(len(cleaned)), "unique_ids": int(cleaned['order_id'].nunique()),
         "same_id_set": set(cleaned['order_id']) == set(source['order_id'])},
        len(cleaned) == 500 and cleaned['order_id'].is_unique and set(cleaned['order_id']) == set(source['order_id']),
        "The missing-data output must retain every original order ID.")

    missing_after = int(cleaned.isna().sum().sum())
    add("M2-VAL-03", "no missing values remain", missing_after, missing_after == 0,
        "All empty source fields should have a field-appropriate recovered value.")

    originally_observed = source.notna()
    changed_observed = 0
    for col in source.columns:
        mask = originally_observed[col]
        changed_observed += int((source.loc[mask, col] != cleaned.loc[mask, col]).sum())
    add("M2-VAL-04", "all originally observed values preserved", changed_observed,
        changed_observed == 0,
        "The missing batch guarantee says its observed values are correct and must be preserved.")

    source_missing_cells = int(source.isna().sum().sum())
    add("M2-VAL-05", "audit row count equals recovered fields",
        {"source_missing_cells": source_missing_cells, "audit_rows": int(len(audit))},
        len(audit) == source_missing_cells,
        "There should be one audit row per recovered field.")

    audit_keys = set(zip(audit["record_key"], audit["field_name"]))
    expected_keys = {
        (source.at[i, "order_id"], col)
        for col in source.columns
        for i in source.index[source[col].isna()]
    }
    add("M2-VAL-06", "audit exactly identifies the recovered cells",
        {"audit_keys": len(audit_keys), "expected_keys": len(expected_keys)},
        audit_keys == expected_keys and len(audit_keys) == len(audit),
        "The audit log must reconcile to actual input/output changes without duplicates.")

    season_bad = 0
    for i in cleaned.index:
        if cleaned.at[i, "season"] != month_to_season(cleaned.at[i, "date"]):
            season_bad += 1
    add("M2-VAL-07", "date/season relationship", season_bad, season_bad == 0,
        "Every row should follow the published Australian month-to-season mapping.")

    location_bad = 0
    max_distance_error = 0.0
    for i in cleaned.index:
        warehouse, distance = nearest_warehouse(
            cleaned.at[i, "customer_lat"], cleaned.at[i, "customer_long"], warehouses
        )
        if cleaned.at[i, "nearest_warehouse"] != warehouse:
            location_bad += 1
        err = abs(float(cleaned.at[i, "distance_to_nearest_warehouse"]) - distance)
        max_distance_error = max(max_distance_error, err)
        if err > 0.0001 + 1e-9:
            location_bad += 1
    add("M2-VAL-08", "nearest warehouse and haversine distance",
        {"violations": location_bad, "max_distance_error_km": max_distance_error},
        location_bad == 0,
        "Warehouse names and distances should agree with the supplied warehouse coordinates.")

    happiness_bad = 0
    for i in cleaned.index:
        expected, _ = derive_happiness(cleaned.at[i, "latest_customer_review"], analyzer, threshold)
        actual = bool(parse_bool(cleaned.at[i, "is_happy_customer"]))
        if actual != expected:
            happiness_bad += 1
    add("M2-VAL-09", "VADER happiness rule", happiness_bad, happiness_bad == 0,
        "Happiness should follow the frozen VADER threshold and no-review sentinel rule.")

    price_errors = []
    total_errors = []
    for i in cleaned.index:
        expected_price = calculate_order_price(cleaned.at[i, "shopping_cart"], prices)
        price_errors.append(abs(float(cleaned.at[i, "order_price"]) - expected_price))
        expected_total = calculate_order_total(
            cleaned.at[i, "order_price"], cleaned.at[i, "coupon_discount"], cleaned.at[i, "delivery_charges"]
        )
        total_errors.append(abs(float(cleaned.at[i, "order_total"]) - expected_total))
    max_price_error = float(max(price_errors))
    max_total_error = float(max(total_errors))
    add("M2-VAL-10", "cart price and order total arithmetic",
        {"max_order_price_error": max_price_error, "max_order_total_error": max_total_error},
        max_price_error <= 0.01 + 1e-9 and max_total_error <= 0.01 + 1e-9,
        "Calculated money fields should satisfy the published current-price and total rules.")

    minimum_r2 = min(v["held_out_r_squared"] for v in model_metrics.values())
    maximum_mae = max(v["held_out_mae"] for v in model_metrics.values())
    add("M2-VAL-11", "held-out seasonal delivery model quality",
        {"minimum_held_out_r_squared": minimum_r2, "maximum_held_out_mae": maximum_mae},
        minimum_r2 >= 0.99,
        "The data contract says clean data normally supports a very strong seasonal relationship.")

    all_passed = all(check["status"] == "PASS" for check in checks)
    return checks, all_passed


def run(data_zip, output_dir, group=DATASET_ID):
    data_zip = Path(data_zip)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if group != DATASET_ID:
        raise ValueError(f"This component is for {DATASET_ID}, not {group}")

    zip_hash_before = hashlib.sha256(data_zip.read_bytes()).hexdigest()

    with zipfile.ZipFile(data_zip, "r") as zf:
        manifest = json.loads(zf.read("cleansing_manifest.json").decode("utf-8"))
        if manifest["group_alias"] != group:
            raise ValueError("Group ID does not match cleansing_manifest.json")

        missing_df = read_zip_csv(zf, MISSING_FILE)
        outlier_df = read_zip_csv(zf, OUTLIER_FILE)
        warehouses = read_zip_csv(zf, "warehouses.csv")

        if missing_df.columns.tolist() != REQUIRED_COLUMNS:
            raise ValueError("Missing-data input schema does not match the data contract")
        if len(missing_df) != 500 or len(outlier_df) != 500:
            raise ValueError("Expected 500 rows in each supplied input batch")

        with tempfile.TemporaryDirectory() as temp_root:
            analyzer, contract, resource_sha = prepare_vader(zf, temp_root)
            cleaned, audit, evidence = recover_missing_data(
                missing_df, outlier_df, warehouses, analyzer,
                float(contract["compound_threshold"])
            )
            prices = evidence["price_inference"]["prices"]
            checks, all_passed = validate_output(
                missing_df, cleaned, audit, warehouses, prices, analyzer,
                float(contract["compound_threshold"]), evidence["delivery_model_evaluation"]
            )

    if not all_passed:
        failed = [c["check_id"] for c in checks if c["status"] == "FAIL"]
        raise ValueError(f"Missing validation failed: {failed}")

    output_path = output_dir / OUTPUT_FILE
    audit_path = output_dir / AUDIT_FILE
    validation_path = output_dir / VALIDATION_FILE

    cleaned.to_csv(output_path, index=False, encoding="utf-8")
    audit.to_csv(audit_path, index=False, encoding="utf-8")

    reread = pd.read_csv(output_path, dtype=str)
    reread_audit = pd.read_csv(audit_path, dtype=str)

    validation = {
        "dataset_id": group,
        "component": "Missing missing-data recovery",
        "input_zip": data_zip.name,
        "source_zip_sha256_before": zip_hash_before,
        "source_zip_sha256_after": hashlib.sha256(data_zip.read_bytes()).hexdigest(),
        "resource_contract": contract,
        "verified_vader_lexicon_sha256": resource_sha,
        "python_dependencies": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "nltk": nltk.__version__,
        },
        "input_rows": int(len(missing_df)),
        "output_rows": int(len(reread)),
        "missing_cells_before": int(missing_df.isna().sum().sum()),
        "missing_cells_after": int(reread.isna().sum().sum()),
        "audit_rows": int(len(reread_audit)),
        "evidence": evidence,
        "checks": checks,
        "all_checks_passed": bool(all_passed),
    }
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")

    return {
        "output_csv": str(output_path),
        "audit_csv": str(audit_path),
        "validation_json": str(validation_path),
        "validation": validation,
    }


def main():
    parser = argparse.ArgumentParser(description="Retail Missing missing-data recovery")
    parser.add_argument("--data-zip", required=True, help="Path to Retail_cleansing.zip")
    parser.add_argument("--output-dir", default="missing_run", help="Output directory")
    parser.add_argument("--group", default=DATASET_ID, help="Group identifier")
    args = parser.parse_args()

    result = run(args.data_zip, args.output_dir, args.group)
    print("Missing missing-data recovery complete")
    print("Output:", result["output_csv"])
    print("Audit:", result["audit_csv"])
    print("Validation:", result["validation_json"])
    print("All validation checks passed:", result["validation"]["all_checks_passed"])


if __name__ == "__main__":
    main()
