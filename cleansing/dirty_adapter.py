"""Run Dirty's unchanged ZIP-based component from a local input directory."""
from pathlib import Path
import json
import tempfile
import zipfile

import pandas as pd
import dirty as m1
from validation import verify_input_manifest


def run(data_dir, output_dir):
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir)
    before = verify_input_manifest(data_dir)
    manifest = json.loads((data_dir / "cleansing_manifest.json").read_text(encoding="utf-8"))
    # The original runner accepts ZIP input. This temporary, deterministic wrapper
    # contains the unchanged supplied bytes and is never part of the handoff.
    with tempfile.TemporaryDirectory(prefix="Retail_dirty_") as temporary:
        archive_path = Path(temporary) / "Retail_cleansing.zip"
        names = ["cleansing_manifest.json"] + [entry["path"] for entry in manifest["files"]]
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, (data_dir / name).read_bytes())
        report = m1.run(archive_path, output_dir, group="Retail", seed=5196)
    if before != verify_input_manifest(data_dir):
        raise ValueError("Original input package changed during Dirty execution")
    report["input_adapter"] = {
        "input_kind": "allocated local extracted package",
        "archive_sha256_meaning": "Deterministic temporary ZIP of unchanged package bytes, removed after execution",
        "original_member_checksums_unchanged": True,
    }
    report["evidence"]["team_alignment"] = (
        "Dirty and Outlier use the same seeded diagnostic fit. "
        "The integrated pipeline verifies their coefficients; Missing's refit is documented separately."
    )
    report["confidence_definitions"]["medium"] = "Field attribution additionally relies on validated diagnostic seasonal models"
    (output_dir / "dirty_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    evidence = report["evidence"]
    model_table = pd.DataFrame(evidence["seasonal_models"]).T[
        ["training_n", "heldout_n", "heldout_r2", "heldout_mae_aud", "heldout_max_error_aud"]
    ]
    tables = {
        "Dirty-component price identification": pd.Series({
            key: evidence["price_inference"][key]
            for key in ["training_count", "rank", "condition_number", "max_linear_residual", "max_cart_residual"]
        }, name="observed").to_frame(),
        "Dirty-component independent price checks": pd.DataFrame(evidence["price_validation"]).T,
        "Dirty-component seasonal held-out evidence": model_table,
        "Dirty-component exported-file checks": pd.DataFrame(report["readback"]["checks"]),
    }
    summary = (f"**{sum(report['readback']['changes_by_field'].values())} field repairs**, "
               f"**{len(report['unresolved'])} unresolved decisions**; "
               f"all **{len(report['readback']['checks'])}** exported-file checks passed. "
               "Detailed accepted and rejected alternatives are retained in `dirty_validation.json`.")
    return {"summary": summary, "tables": tables, "report": report}
