"""Integration checks independent of saved Notebook outputs and claimed statuses."""
from pathlib import Path
from collections import Counter
from decimal import Decimal
import hashlib
import json

import pandas as pd


NUMERIC = {"order_price", "delivery_charges", "customer_lat", "customer_long",
           "coupon_discount", "order_total", "distance_to_nearest_warehouse"}
BOOL = {"is_expedited_delivery", "is_happy_customer"}
AUDIT_COLUMNS = ["source_file", "record_key", "field_name", "issue_type",
                 "original_value", "corrected_value", "method", "evidence", "confidence"]


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equivalent(field, before, after):
    if before == "" or after == "":
        return before == after
    if field in NUMERIC:
        return Decimal(str(before)) == Decimal(str(after))
    if field in BOOL:
        conversion = {"true": True, "false": False, "1": True, "0": False}
        return conversion[str(before).lower()] == conversion[str(after).lower()]
    if field == "shopping_cart":
        def items(value):
            return Counter((item["product"], item["quantity"]) for item in json.loads(value))
        return items(before) == items(after)
    return str(before) == str(after)


def verify_input_manifest(data_dir, dataset_id="Retail"):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "cleansing_manifest.json").read_text(encoding="utf-8"))
    if manifest["group_alias"] != dataset_id:
        raise ValueError("Allocated input group differs from configured group")
    records = []
    for item in manifest["files"]:
        path = (data_dir / item["path"]).resolve()
        if not path.is_relative_to(data_dir.resolve()):
            raise ValueError("Manifest path is outside the input package")
        digest = fingerprint(path)
        ok = digest == item["sha256"] and path.stat().st_size == item["bytes"]
        records.append({"file": item["path"], "sha256": digest, "passed": ok})
    if not all(r["passed"] for r in records):
        raise ValueError("An input file does not match the allocated manifest")
    return records


def reconcile_outputs(data_dir, output_dir, dataset_id="Retail"):
    """Compare each actual cell change and removal with the regenerated audit."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    audit = read(output_dir / "audit_log.csv")
    if audit.columns.tolist() != AUDIT_COLUMNS:
        raise ValueError("Audit schema/order mismatch")
    expected_files = {f"{dataset_id}_{b}_data.csv" for b in ("dirty", "missing", "outlier")}
    if set(audit.source_file) != expected_files:
        raise ValueError("Audit is missing a batch or contains an unknown source file")
    if audit.duplicated(["source_file", "record_key", "field_name"]).any():
        raise ValueError("Duplicate audit entries")
    if not audit.confidence.isin(["high", "medium", "low"]).all():
        raise ValueError("Invalid confidence label")
    if audit[["method", "evidence"]].eq("").any().any():
        raise ValueError("Every correction/removal requires method and evidence")
    rows = []
    all_output_ids = []
    for batch in ("dirty", "missing", "outlier"):
        filename = f"{dataset_id}_{batch}_data.csv"
        source = read(data_dir / filename)
        output = read(output_dir / f"{dataset_id}_{batch}_clean.csv")
        if list(output.columns) != list(source.columns) or len(output.columns) != 16:
            raise ValueError(f"{batch}: incorrect output schema")
        if not output.order_id.is_unique or output.eq("").any().any():
            raise ValueError(f"{batch}: duplicate IDs or missing values")
        src, dst = source.set_index("order_id"), output.set_index("order_id")
        if not set(dst.index) <= set(src.index):
            raise ValueError(f"{batch}: unexpected order ID")
        removed = set(src.index) - set(dst.index)
        if batch != "outlier" and (removed or len(output) != 500):
            raise ValueError(f"{batch}: all 500 original IDs must remain")
        actual = {}
        preserved_observed = True
        for key, row in dst.iterrows():
            for field in dst.columns:
                before, after = src.at[key, field], row[field]
                if not equivalent(field, before, after):
                    actual[(key, field)] = (before, after)
                    if batch == "missing" and before != "":
                        preserved_observed = False
        if not preserved_observed:
            raise ValueError("Originally observed missing-batch data changed")
        entries = audit.loc[audit.source_file.eq(filename)]
        if batch == "outlier":
            if actual:
                raise ValueError("Retained outlier rows were modified")
            if set(entries.record_key) != removed or len(entries) != len(removed):
                raise ValueError("Outlier removal IDs and audit disagree")
            for entry in entries.itertuples():
                if not (entry.field_name == "delivery_charges" and entry.issue_type == "outlier_row"
                        and entry.corrected_value == "ROW_REMOVED"
                        and equivalent("delivery_charges", entry.original_value, src.at[entry.record_key, "delivery_charges"])):
                    raise ValueError("Incorrect outlier audit entry")
        else:
            if set(actual) != set(zip(entries.record_key, entries.field_name)):
                raise ValueError(f"{batch}: changes and audit keys do not reconcile")
            for entry in entries.itertuples():
                before, after = actual[(entry.record_key, entry.field_name)]
                original = "" if entry.original_value == "<MISSING>" else entry.original_value
                if entry.issue_type != batch or not equivalent(entry.field_name, before, original) or not equivalent(entry.field_name, after, entry.corrected_value):
                    raise ValueError(f"{batch}: audit values disagree with source/output")
        all_output_ids += list(dst.index)
        rows.append({"batch": batch, "input_rows": len(source), "retained_rows": len(output),
                     "removed_rows": len(removed), "changed_cells": len(actual),
                     "audit_entries": len(entries), "passed": True})
    if len(all_output_ids) != len(set(all_output_ids)):
        raise ValueError("Order IDs overlap across cleaned batches")
    return pd.DataFrame(rows)


def compare_with_handoff(actual_path, supplied_path):
    actual, supplied = read(actual_path).set_index("order_id"), read(supplied_path).set_index("order_id")
    if list(actual.columns) != list(supplied.columns) or set(actual.index) != set(supplied.index):
        raise ValueError(f"Reproduced output disagrees with handoff structure: {Path(actual_path).name}")
    differences = [(key, field) for key, row in actual.iterrows() for field in actual.columns
                   if not equivalent(field, row[field], supplied.at[key, field])]
    return {"file": Path(actual_path).name, "rows": len(actual), "semantic_differences": len(differences),
            "passed": not differences, "actual_sha256": fingerprint(actual_path),
            "handoff_sha256": fingerprint(supplied_path)}
