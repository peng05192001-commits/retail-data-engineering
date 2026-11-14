"""Unit and read-only Retail integration tests. See --help for reproduction.

Only --evidence writes a file: a JSON verification record, never a cleaned CSV.
The data ZIP is always opened read-only. No dirty-batch repairs are attempted.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import unittest
import zipfile

import numpy as np
from monetary import cart_price, infer_current_prices, order_total, parse_cart


def make_cart(*items):
    return json.dumps([{"product": product, "quantity": quantity}
                       for product, quantity in items])


class MonetaryUnitTests(unittest.TestCase):
    def test_published_example(self):
        self.assertEqual(order_total(100, 10, 8), 98.0)

    def test_coupon_boundaries_and_delivery_not_discounted(self):
        self.assertEqual(order_total(100, 0, 8), 108.0)
        self.assertEqual(order_total(100, 100, 8), 8.0)
        self.assertEqual(order_total(100, 25, 8), 83.0)

    def test_numeric_csv_strings(self):
        self.assertEqual(order_total("100.00", "10", "8.00"), 98.0)

    def test_multiple_items_and_quantities(self):
        self.assertEqual(cart_price(make_cart(("A", 3), ("B", 2)),
                                    {"A": 1.25, "B": 2.1}), 7.95)

    def test_line_amounts_are_rounded_before_sum(self):
        # Synthetic fractional-cent prices isolate the rounding sequence.
        actual = cart_price(make_cart(("A", 1), ("B", 1)),
                            {"A": 0.335, "B": 0.335})
        self.assertEqual(actual, 0.68)
        self.assertNotEqual(actual, round(0.335 + 0.335, 2))

    def test_multiply_quantity_before_rounding(self):
        self.assertEqual(cart_price(make_cart(("A", 3)), {"A": 0.335}), 1.01)
        self.assertNotEqual(1.01, round(round(0.335, 2) * 3, 2))

    def test_repeated_entries_keep_line_boundaries(self):
        self.assertEqual(cart_price(make_cart(("A", 1), ("A", 1)),
                                    {"A": 0.335}), 0.68)

    def test_no_premature_discounted_subtotal_rounding(self):
        self.assertEqual(order_total(0.03, 50, 0.01), 0.03)
        early = round(round(0.03 * (1 - 50 / 100), 2) + 0.01, 2)
        self.assertEqual(early, 0.02)

    def test_binary_float_rounding_is_intentional(self):
        self.assertEqual(order_total(5.35, 50, 0), 2.67)
        decimal_result = (Decimal("5.35") * Decimal("0.5")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_EVEN)
        self.assertEqual(decimal_result, Decimal("2.68"))

    def test_invalid_cart_quantities_and_shapes_rejected(self):
        for value in (0, -1, 1.5, True, "2"):
            with self.subTest(quantity=value), self.assertRaises(ValueError):
                parse_cart(make_cart(("A", value)))
        for raw in ("not json", "[]", "{}", '[{"product":"A"}]'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_cart(raw)

    def test_missing_product_price_rejected(self):
        with self.assertRaises(ValueError):
            cart_price(make_cart(("unknown", 1)), {"A": 1.0})

    def test_invalid_monetary_inputs_rejected(self):
        for values in ((-1, 10, 8), (100, -1, 8), (100, 101, 8),
                       (100, 10, -1), ("", 10, 8), (True, 10, 8),
                       (float("nan"), 10, 8), (100, 10, float("inf"))):
            with self.subTest(values=values), self.assertRaises(ValueError):
                order_total(*values)

    def test_helpers_do_not_mutate_arguments(self):
        cart = make_cart(("A", 2))
        prices = {"A": 1.25}
        before = copy.deepcopy(prices)
        self.assertEqual(cart_price(cart, prices), 2.5)
        self.assertEqual(prices, before)
        self.assertEqual(cart, make_cart(("A", 2)))

    def test_inference_recovers_known_synthetic_prices(self):
        rows = [{"shopping_cart": make_cart(("A", 1)), "order_price": "1.25"},
                {"shopping_cart": make_cart(("B", 2)), "order_price": "5.00"},
                {"shopping_cart": make_cart(("A", 2), ("B", 1)), "order_price": "5.00"}]
        before = copy.deepcopy(rows)
        result = infer_current_prices(rows)
        self.assertEqual(result.prices, {"A": 1.25, "B": 2.5})
        self.assertEqual(rows, before)

    def test_nonidentifiable_prices_rejected(self):
        rows = [{"shopping_cart": make_cart(("A", 1), ("B", 1)),
                 "order_price": "5.00"}]
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            infer_current_prices(rows)

    def test_inconsistent_training_evidence_rejected(self):
        rows = [{"shopping_cart": make_cart(("A", 1)), "order_price": value}
                for value in ("1.00", "3.00")]
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            infer_current_prices(rows)

    def test_unsupported_cent_snapping_rejected(self):
        rows = [{"shopping_cart": make_cart(("A", 2)), "order_price": "2.01"}]
        with self.assertRaisesRegex(ValueError, "cent-valued"):
            infer_current_prices(rows)


def read_evidence(archive_path: Path) -> dict:
    archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive_path, "r") as archive:
        manifest = json.loads(archive.read("cleansing_manifest.json"))
        if manifest["group_alias"] != "Retail":
            raise ValueError("This review is for Retail, not another group")
        for entry in manifest["files"]:
            raw = archive.read(entry["path"])
            if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError(f"Input checksum mismatch: {entry['path']}")
        # Dirty data is not parsed or used in this component.
        rows = {}
        for batch in ("outlier", "missing"):
            name = f"Retail_{batch}_data.csv"
            text = archive.read(name).decode("utf-8-sig")
            rows[batch] = list(csv.DictReader(io.StringIO(text)))
    if any(len(batch) != 500 for batch in rows.values()):
        raise ValueError("Expected 500 orders in each supplied evidence batch")
    all_ids = [row["order_id"] for batch in rows.values() for row in batch]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Evidence batches have duplicate order IDs")
    seed = 5196
    indices = np.random.default_rng(seed).permutation(500)
    train = [rows["outlier"][int(i)] for i in indices[:350]]
    heldout = [rows["outlier"][int(i)] for i in indices[350:]]
    before = copy.deepcopy(rows)
    fit = infer_current_prices(train)  # Only these 350 rows enter fitting.

    def price_check(check_rows):
        predicted = [cart_price(r["shopping_cart"], fit.prices) for r in check_rows]
        observed = [float(r["order_price"]) for r in check_rows]
        differences = [abs(p - y) for p, y in zip(predicted, observed)]
        return {"n": len(check_rows), "exact_matches": sum(p == y for p, y in zip(predicted, observed)),
                "max_absolute_error_aud": max(differences, default=0.0)}

    missing_prices = [r for r in rows["missing"] if r["order_price"] != ""]
    monetary_fields = ("order_price", "coupon_discount", "delivery_charges", "order_total")
    trusted_totals = [r for r in rows["missing"] if all(r[f] != "" for f in monetary_fields)]
    total_errors = []
    decimal_differences = []
    for row in trusted_totals:
        predicted = order_total(row["order_price"], row["coupon_discount"], row["delivery_charges"])
        total_errors.append(abs(predicted - float(row["order_total"])))
        alternative = (Decimal(row["order_price"]) *
                       (1 - Decimal(row["coupon_discount"]) / 100) +
                       Decimal(row["delivery_charges"])).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
        delta = alternative - Decimal(row["order_total"])
        if delta:
            decimal_differences.append({
                "order_id": row["order_id"],
                **{f: row[f] for f in monetary_fields},
                "python_result": predicted, "decimal_half_even_result": str(alternative),
                "decimal_minus_observed_aud": str(delta),
            })
    report = {
        "group": "Retail", "scope": "Price inference and pure monetary helpers only",
        "archive_sha256_before": archive_hash,
        "archive_sha256_after": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "input_manifest_checksums_verified": True,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "split": {"method": "numpy default_rng(seed).permutation over supplied outlier row order",
                  "seed": seed, "training_source": "Retail_outlier_data.csv",
                  "training_ids": [r["order_id"] for r in train],
                  "heldout_ids": [r["order_id"] for r in heldout],
                  "disjoint": not bool({r["order_id"] for r in train} & {r["order_id"] for r in heldout})},
        "fit": asdict(fit),
        "price_validation": {"training": price_check(train), "heldout": price_check(heldout),
                             "missing_observed": price_check(missing_prices)},
        "total_validation": {"source": "Retail_missing_data.csv", "n": len(trusted_totals),
                             "exact_matches": sum(error == 0 for error in total_errors),
                             "max_absolute_error_aud": max(total_errors),
                             "validation_ids": [r["order_id"] for r in trusted_totals]},
        "decimal_comparison": {"n_different": len(decimal_differences), "cases": decimal_differences},
        "in_memory_source_values_unchanged": rows == before,
        "source_archive_unchanged": archive_hash == hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "no_cleaned_csvs_written": True,
    }
    return report


def integration_tests(report):
    class RetailIntegrationTests(unittest.TestCase):
        def test_training_is_full_rank(self):
            self.assertEqual(report["fit"]["training_count"], 350)
            self.assertEqual(report["fit"]["rank"], 12)
            self.assertEqual(len(report["fit"]["products"]), 12)
            self.assertLess(report["fit"]["max_distance_from_cent"], 1e-7)

        def test_training_prices(self):
            self.assertEqual(report["price_validation"]["training"],
                             {"n": 350, "exact_matches": 350, "max_absolute_error_aud": 0.0})

        def test_unused_outlier_orders(self):
            self.assertTrue(report["split"]["disjoint"])
            self.assertEqual(report["price_validation"]["heldout"],
                             {"n": 150, "exact_matches": 150, "max_absolute_error_aud": 0.0})

        def test_independent_missing_batch_prices(self):
            self.assertEqual(report["price_validation"]["missing_observed"],
                             {"n": 489, "exact_matches": 489, "max_absolute_error_aud": 0.0})

        def test_trusted_observed_totals(self):
            result = report["total_validation"]
            self.assertEqual((result["n"], result["exact_matches"], result["max_absolute_error_aud"]),
                             (469, 469, 0.0))

        def test_decimal_is_not_an_equivalent_substitution(self):
            # Regression evidence for this supplied package, not an defined count rule.
            self.assertEqual(report["decimal_comparison"]["n_different"], 9)
            for case in report["decimal_comparison"]["cases"]:
                self.assertEqual(abs(Decimal(case["decimal_minus_observed_aud"])), Decimal("0.01"))

        def test_source_preservation(self):
            self.assertTrue(report["input_manifest_checksums_verified"])
            self.assertTrue(report["source_archive_unchanged"])
            self.assertTrue(report["in_memory_source_values_unchanged"])

    return RetailIntegrationTests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, help="Optional JSON test/inference record")
    args = parser.parse_args()
    if args.evidence and args.evidence.suffix.lower() != ".json":
        parser.error("--evidence must name a JSON file")
    report = read_evidence(args.data_zip)
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite([
        loader.loadTestsFromTestCase(MonetaryUnitTests),
        loader.loadTestsFromTestCase(integration_tests(report)),
    ])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report["tests"] = {"executed_at_utc": datetime.now(timezone.utc).isoformat(),
                       "run": result.testsRun, "failures": len(result.failures),
                       "errors": len(result.errors), "skipped": len(result.skipped),
                       "passed": result.wasSuccessful()}
    if args.evidence:
        args.evidence.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prices_aud": report["fit"]["prices"],
                      "price_validation": report["price_validation"],
                      "total_validation": {k:v for k,v in report["total_validation"].items() if k != "validation_ids"},
                      "decimal_differences": report["decimal_comparison"]["n_different"],
                      "tests": report["tests"]}, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
