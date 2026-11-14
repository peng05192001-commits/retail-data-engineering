"""Run Missing's unchanged recovery functions from a local extracted package."""
from pathlib import Path
import hashlib
import json
import tempfile

import pandas as pd
import missing as m2


class LocalPackage:
    """Minimal read interface for Missing's verified local VADER loader."""
    def __init__(self, directory):
        self.root = Path(directory).resolve()

    def read(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Resource path escapes the allocated data directory")
        return path.read_bytes()


def run(data_dir, output_dir):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    read = lambda name: pd.read_csv(data_dir / name, dtype=str)
    original = read(m2.MISSING_FILE)
    outlier = read(m2.OUTLIER_FILE)
    warehouses = read("warehouses.csv")
    with tempfile.TemporaryDirectory(prefix="Retail_vader_") as temporary:
        scorer, contract, resource_hash = m2.prepare_vader(LocalPackage(data_dir), temporary)
        cleaned, audit, evidence = m2.recover_missing_data(
            original, outlier, warehouses, scorer, float(contract["compound_threshold"])
        )
        cleaned.to_csv(output_dir / m2.OUTPUT_FILE, index=False, encoding="utf-8")
        audit.to_csv(output_dir / m2.AUDIT_FILE, index=False, encoding="utf-8")
        # Unlike the original component, run all validations on disk read-back.
        disk = pd.read_csv(output_dir / m2.OUTPUT_FILE, dtype=str)
        disk_audit = pd.read_csv(output_dir / m2.AUDIT_FILE, dtype=str)
        checks, passed = m2.validate_output(
            original, disk, disk_audit, warehouses, evidence["price_inference"]["prices"],
            scorer, float(contract["compound_threshold"]), evidence["delivery_model_evaluation"]
        )
    if not passed:
        raise ValueError("Missing read-back validation failed")
    # Independent arithmetic corroboration for the model-imputed delivery charges.
    selected = original.delivery_charges.isna()
    algebraic_charge = (
        disk.loc[selected, "order_total"].astype(float)
        - disk.loc[selected, "order_price"].astype(float)
        * (1 - disk.loc[selected, "coupon_discount"].astype(float) / 100)
    ).round(2)
    charge_error = (algebraic_charge - disk.loc[selected, "delivery_charges"].astype(float)).abs()
    if not (charge_error <= .01 + 1e-9).all():
        raise ValueError("Model-imputed charges disagree with independent observed-total arithmetic")
    result = {
        "evidence": evidence, "checks": checks, "all_checks_passed": passed,
        "resource_contract": contract, "verified_vader_lexicon_sha256": resource_hash,
        "independent_charge_check": {"rows": int(selected.sum()), "max_abs_difference_aud": float(charge_error.max())},
    }
    (output_dir / "missing_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
