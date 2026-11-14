"""Corresponding executable source for solution.ipynb.
Generated from the notebook code cells; helper modules are included alongside it.
"""
import argparse
import os

def run_solution(data_dir="./data", output_dir="./outputs"):
    #   — Retail

    # 0. Configuration, dependencies and input provenance

    from pathlib import Path
    import os, sys, json, importlib.metadata as metadata
    import numpy as np
    import pandas as pd
    from IPython.display import display, Markdown, HTML

    DATASET_ID = "Retail"
    DATA_DIR = Path(data_dir)
    OUTPUT_DIR = Path(output_dir)
    KNOWN_INPUT_LIMITATIONS = []

    from pipeline import run_dirty, run_missing, run_outlier, initial_investigation
    from validation import verify_input_manifest, reconcile_outputs

    def show_table(title, frame, precision=4):
        display(Markdown(f"**{title}**"))
        with pd.option_context("display.max_colwidth", None):
            table = frame.to_html(border=0, classes="Retail-table", na_rep="Undefined",
                                  float_format=lambda value: f"{value:.{precision}f}", escape=True)
        css = "<style>.Retail-table td,.Retail-table th{padding:7px;vertical-align:top;text-align:left;max-width:320px;white-space:normal;} .Retail-table thead{background:#edf3f5;}</style>"
        display(HTML(css + '<div style="overflow-x:auto">' + table + '</div>'))

    print("Python", sys.version.split()[0])
    show_table("Runtime packages", pd.DataFrame([
        {"package": name, "version": metadata.version(name)}
        for name in ["numpy", "pandas", "matplotlib", "nltk", "scikit-learn"]
    ]))
    input_manifest_before = verify_input_manifest(DATA_DIR, DATASET_ID)
    show_table("Allocated input checksums", pd.DataFrame(input_manifest_before))


    # Cleansing — Data cleansing and explanation

    eda = initial_investigation(DATA_DIR, DATASET_ID)
    for name, table in eda.items():
        show_table(name, table)


    # 1.1 Dirty data — field attribution and preservation

    dirty_result = run_dirty(DATA_DIR, OUTPUT_DIR)
    dirty_audit = pd.read_csv(OUTPUT_DIR / "audit_log_dirty.csv", dtype=str, keep_default_na=False)
    show_table("Dirty repairs by field", dirty_audit.groupby("field_name").size().rename("repaired_cells").to_frame())
    show_table("Examples of traceable dirty-data decisions", dirty_audit.head(12))
    display(Markdown(dirty_result["summary"]))
    for title, table in dirty_result["tables"].items():
        show_table(title, table)


    # 1.2 Missing data — field-appropriate recovery

    missing_result = run_missing(DATA_DIR, OUTPUT_DIR)
    missing_counts = pd.Series(missing_result["evidence"]["missing_counts_before"])
    show_table("Missing fields before recovery", missing_counts[missing_counts.gt(0)].rename("missing_cells").to_frame())
    show_table("Missing-component price inference", pd.Series(missing_result["evidence"]["price_inference"]["prices"], name="current_unit_price_AUD").to_frame())
    show_table("Missing-component held-out seasonal performance", pd.DataFrame(missing_result["evidence"]["delivery_model_evaluation"]).T)
    show_table("Missing-component read-back validation", pd.DataFrame(missing_result["checks"]))
    display(Markdown(f"Independent observed-total check: **{missing_result['independent_charge_check']['rows']}** imputed charges; maximum difference **AUD {missing_result['independent_charge_check']['max_abs_difference_aud']:.2f}**."))


    # 1.3 Outlier data — seasonal residuals and independent corroboration

    outlier_result = run_outlier(DATA_DIR, OUTPUT_DIR, DATASET_ID)
    show_table("Outlier batch: initial field investigation", outlier_result["outlier_eda"])
    show_table("Outlier model: held-out performance", outlier_result["evaluation"], 9)
    show_table("Outlier model: fitted coefficients", outlier_result["coefficients"], 6)
    show_table("Dirty / Outlier diagnostic model agreement", outlier_result["dirty_model_alignment"], 12)
    print("Dirty / Missing price agreement:", outlier_result["current_price_alignment"])
    show_table("Residual decision and separation", outlier_result["decisions"], 6)
    show_table("Threshold sensitivity", outlier_result["thresholds"])
    show_table("Independent arithmetic corroboration", outlier_result["corroboration"])
    show_table("Retained-row coefficient refit", outlier_result["refit_comparison"], 6)
    print("Maximum held-out current-price equation error (AUD):", outlier_result["price_holdout_errors"].max())
    for figure in outlier_result["figures"]:
        display(figure)
    # Include Outlier's raw-charge-screen comparison table.
    for evidence in outlier_result["evidence"]:
        value = evidence["value"]
        if isinstance(value, pd.DataFrame) and "screen" in value.columns:
            show_table("Alternative raw-charge screens", value)


    # 1.4 Export boundaries

    show_table("Cross-file read-back and audit checks", outlier_result["cross_file_validation"])
    show_table("Business-rule failure counts by output", outlier_result["business_failures"])
    integration_reconciliation = reconcile_outputs(DATA_DIR, OUTPUT_DIR, DATASET_ID)
    show_table("Independent actual-change / audit reconciliation", integration_reconciliation)
    show_table("Combined audit counts", outlier_result["audit"].groupby(["source_file", "issue_type"]).size().rename("entries").to_frame())
    assert integration_reconciliation["passed"].all()
    assert outlier_result["cross_file_validation"]["passed"].all()
    assert not outlier_result["business_failures"].to_numpy().any()


    # Customer profiling — Customer profiling and transformation

    from pathlib import Path
    import os
    import sys
    import importlib.metadata as metadata

    # DATASET_ID is configured in section 0.
    CLEAN_DIR = OUTPUT_DIR
    RAW_DIR = DATA_DIR
    # KNOWN_INPUT_LIMITATIONS is configured in section 0.

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown, HTML
    from profiles import (
        FEATURES, METHODS, input_status, load_cleaned_orders, build_profiles,
        describe_features, compare_transformations, distance_effects,
        recommendation_table, interpretation, analysis_checks, make_figures,
    )

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 200)
    print("Python:", sys.version.split()[0])
    print({name: metadata.version(name) for name in ("numpy", "pandas", "matplotlib", "ipykernel")})
    print("Clean input directory:", CLEAN_DIR.resolve())

    def show_table(title, frame, precision=4):
        display(Markdown(f"**{title}**"))
        with pd.option_context("display.max_colwidth", None):
            table = frame.to_html(border=0, classes="profiles-table", na_rep="Undefined",
                                  float_format=lambda value: f"{value:.{precision}f}", escape=True)
        css = "<style>.profiles-table th,.profiles-table td {padding:8px;vertical-align:top;text-align:left;max-width:310px;white-space:normal;} .profiles-table thead {background:#edf3f5;}</style>"
        display(HTML(css + '<div style="overflow-x:auto">' + table + '</div>'))

    def pending():
        display(Markdown("**Pending:** provide the three final cleaned CSVs and rerun all cells. No sample results are substituted."))


    # 3.1 Build the customer-level analysis copy

    status = input_status(CLEAN_DIR, DATASET_ID)
    show_table("Required cleaned inputs", status)
    READY = bool(status["exists"].all())
    if READY:
        orders, batch_counts, input_checks, input_fingerprints = load_cleaned_orders(
            CLEAN_DIR, DATASET_ID, RAW_DIR
        )
        profiles = build_profiles(orders)
        baseline = profiles.copy(deep=True)
        show_table("T3-A. Retained input sample by batch", batch_counts)
        print(f"Combined distinct orders: {orders.order_id.nunique():,}")
        print(f"Distinct customers / profile rows: {len(profiles):,}")
        print(f"Removed outlier orders: {500 - int(batch_counts.loc[batch_counts.source_batch.eq('outlier'), 'retained_rows'].iloc[0]):,}")
        print("Batch customer counts overlap and must not be added together.")
        show_table("Input prerequisites (business-rule validation remains in Audit validation)", input_checks)
        show_table("Customer profiles — first 10 rows; full table remains in profiles", profiles.head(10))
        if KNOWN_INPUT_LIMITATIONS:
            display(Markdown("**Declared upstream limitations:**\n\n" + "\n".join(f"- {x}" for x in KNOWN_INPUT_LIMITATIONS)))
        else:
            print("No upstream limitations declared here. Refer to the group's Audit validation evidence.")
    else:
        pending()


    # 3.2 Original scales, distributions and relationships

    if READY:
        baseline_summary = describe_features(baseline)
        show_table("T3-B. Original distributions and scales", baseline_summary)
        show_table("T3-C. Original Pearson correlations", baseline.corr())
    else:
        pending()


    # 3.3 Compare z-score standardisation and min-max scaling

    if READY:
        comparisons = compare_transformations(baseline)
        transformed_features = comparisons["versions"]
        effects = distance_effects(transformed_features)
        figures = make_figures(baseline, comparisons, effects)
        display(figures["baseline"])
        show_table("T3-D1. Fitted scaling parameters", comparisons["parameters"])
        show_table("T3-D2. Original versus z-score versus min-max", comparisons["scaling_summary"])
        display(figures["scaling"])
        # Figures are displayed explicitly; close them to avoid duplicate inline output.
        for figure in figures.values():
            plt.close(figure)
    else:
        pending()


    # 3.4 Compare a nonlinear alternative and its distance effects

    if READY:
        show_table("T3-E. Mean order value before and after log1p", comparisons["nonlinear_summary"])
        display(figures["nonlinear"])
        show_table("T3-F. Relative distances and feature contributions", effects)
        display(figures["distance_shares"])
        show_table("T3-G. Pearson relationships after log1p value then z-score",
                   comparisons["correlations"]["Log1p value + Z-score"])
        for observation in interpretation(baseline, comparisons, effects):
            display(Markdown(observation))
    else:
        pending()


    # 3.5 Recommendations, boundaries and limitations

    if READY:
        recommendations = recommendation_table(baseline, comparisons, effects)
        show_table("T3-H. Per-attribute recommendations and evidence", recommendations)
    else:
        pending()


    # Reproducibility and preservation checks

    if READY:
        pd.testing.assert_frame_equal(profiles, baseline)
        checks = analysis_checks(orders, profiles, comparisons, input_fingerprints)
        show_table("T3-I. Executable Customer profiling checks", checks)
        print(f"Customer profiling executed on {len(orders):,} retained orders and {len(profiles):,} customers.")
        print("Review T3-H with the plots and upstream Audit validation findings before release.")
    else:
        display(Markdown("**Customer profiling result status: NOT YET GENERATED.** The code is ready, but final cleaned inputs are required. This notebook is not a completed cleansing release."))


    # Final reproducibility, sources and release boundary

    input_manifest_after = verify_input_manifest(DATA_DIR, DATASET_ID)
    assert input_manifest_before == input_manifest_after, "Original inputs were modified"
    assert READY, "Customer profiling requires all three final cleaned outputs"
    final_summary = {
        "dataset_id": DATASET_ID,
        "raw_input_checksums_unchanged": True,
        "retained_orders": int(len(orders)),
        "customers": int(len(profiles)),
        "audit_entries": int(len(outlier_result["audit"])),
        "reconciliation": integration_reconciliation.to_dict("records"),
        "profiles_checks_passed": bool(checks.result.eq("PASS").all()),
        "profiles_check_count": int(len(checks)),
        "dirty_outlier_model_alignment": outlier_result["dirty_model_alignment"].to_dict("records"),
        "dirty_missing_price_alignment": outlier_result["current_price_alignment"],
    }
    (OUTPUT_DIR / "integration_validation.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    display(Markdown(f"**Run complete:** {len(orders):,} retained orders, {len(profiles):,} customers, {len(outlier_result['audit']):,} audit entries. Original input checksums are unchanged."))

    return final_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("RETAIL_DATA_DIR", "./data"))
    parser.add_argument("--output-dir", default=os.environ.get("RETAIL_OUTPUT_DIR", "./outputs"))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    run_solution(args.data_dir, args.output_dir)
