"""Contract checks for the final Retail source-to-target mapping."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestFinalMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.template = pd.read_csv(
            PROJECT_ROOT / "templates" / "mapping_schema.csv",
            keep_default_na=False,
            dtype=str,
        )
        cls.mapping = pd.read_csv(
            PROJECT_ROOT / "source_to_target_mapping.csv",
            keep_default_na=False,
            dtype=str,
        )
        cls.fill_columns = [
            "source_format",
            "json_source_path",
            "xml_source_path",
            "transformation_or_derivation",
            "overlap_or_conflict_rule",
            "notebook_evidence",
        ]
        cls.catalogue_mask = cls.mapping["output_table"].isin(["customers", "products"])

    def test_official_rows_and_order_are_preserved(self) -> None:
        self.assertEqual(self.mapping.shape, (111, 9))
        self.assertEqual(self.mapping["mapping_id"].tolist(), self.template["mapping_id"].tolist())
        self.assertTrue(
            self.mapping[["output_table", "target_field"]].equals(
                self.template[["output_table", "target_field"]]
            )
        )

    def test_exactly_41_catalogue_rows_are_completed(self) -> None:
        self.assertEqual(int(self.catalogue_mask.sum()), 41)
        core_columns = [
            "source_format",
            "transformation_or_derivation",
            "overlap_or_conflict_rule",
            "notebook_evidence",
        ]
        self.assertTrue(self.mapping.loc[self.catalogue_mask, core_columns].ne("").all().all())

    def test_structural_paths_follow_source_authority(self) -> None:
        customers = self.mapping["output_table"].eq("customers")
        products = self.mapping["output_table"].eq("products")
        self.assertTrue(self.mapping.loc[customers, "json_source_path"].ne("").all())
        self.assertTrue(self.mapping.loc[customers, "xml_source_path"].eq("").all())
        self.assertTrue(self.mapping.loc[products, "xml_source_path"].ne("").all())
        self.assertTrue(self.mapping.loc[products, "json_source_path"].eq("").all())

    def test_all_final_mapping_rows_are_completed(self) -> None:
        core_columns = [
            "source_format",
            "transformation_or_derivation",
            "overlap_or_conflict_rule",
            "notebook_evidence",
        ]
        self.assertTrue(self.mapping[core_columns].ne("").all().all())

        json_rows = self.mapping["source_format"].isin(["JSON", "both"])
        xml_rows = self.mapping["source_format"].isin(["XML", "both"])
        self.assertTrue(self.mapping.loc[json_rows, "json_source_path"].ne("").all())
        self.assertTrue(self.mapping.loc[xml_rows, "xml_source_path"].ne("").all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
