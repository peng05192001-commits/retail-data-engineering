# Retail Data Engineering

A Python project for multi-source retail data integration, traceable cleansing, and customer profiling. It converts nested JSON/XML exports into relational tables, then processes a separate collection of orders using business rules, price equations, and seasonal delivery-charge models. Field-level audits and independent validation make each change inspectable.

## Project Structure

| Module | Implementation |
| --- | --- |
| `integration/` | JSON/XML parsing; six tables covering customers, products, orders, order items, deliveries, and reviews; field mapping, type standardisation, duplicate and overlap reconciliation, text cleaning, foreign-key validation, and exploratory analysis |
| `cleansing/` | Current-price inference from shopping carts; seasonal delivery-charge regression; single-field candidate repairs, missing-value recovery, and delivery-charge outlier removal; field-level auditing, export read-back validation, and customer-feature transformation comparisons |
| `scripts/` | Private-input verification, neutral filename preparation, and a data-free test entry point |

The two modules belong to one portfolio project but use different datasets. Raw orders are not joined across stages, and historical product prices are not reused for current-price inference. The three cleansing batches also contain different orders: retained orders are concatenated for customer profiling, not joined on order ID.

## Key Methods

- Parse JSON/XML with structured parsers and build relational tables at explicitly defined grains using Pandas. Preserve leading zeros in identifiers and postcodes.
- Document source paths, transformations, conflict-resolution rules, and implementation evidence in a source-to-target mapping.
- Construct a quantity matrix from shopping carts and infer product prices with least squares. Check full rank, held-out orders, and an independent batch.
- Fit delivery charges against distance, expedited delivery, and customer happiness separately for each season. Evaluate held-out residuals rather than relying on goodness of fit alone.
- Apply dirty-data repairs under an explicit input contract: each row has at most one incorrect field. Accept only a uniquely supported single-field repair; a failed check does not automatically identify the incorrect field.
- Validate sentiment labels with a frozen NLTK/VADER resource and follow an explicit rounding sequence for item amounts, discounts, delivery charges, and final totals.
- Audit actual changes and independently compare original records with re-read exports. Validate preserved information, input checksums, and cross-batch key uniqueness.
- Compare z-score, min-max, and log1p followed by standardisation for customer features, examining scales, correlations, and distance contributions. No clustering model is fitted.

## Running the Project

Local release verification passed: all three notebooks executed from start to finish, and all six relational tables and three cleaned tables matched the archived baseline values. The 35 data-free tests and 40 monetary/dirty-data regression tests passed. These suites overlap and should not be added together as a unique test count. Saved notebook outputs are deliberately cleared for publication; a full rerun requires authorised private inputs.

Use Python 3.12 in a dedicated environment:

```bash
python -m pip install -r requirements.txt
python scripts/test_public.py
```

The public tests use synthetic cases, field contracts, and boundary conditions without requiring original customer records. **Raw datasets, record-level cleaned outputs, and private conversations are not included.** Cloning this repository alone is therefore insufficient to reproduce the full data-backed run.

Run the commands in each subsection below from the repository root.

### 1. Multi-source Integration and Exploration

```bash
python scripts/prepare_inputs.py --stage integration --source /path/to/original/input-package --destination integration/raw_package
cd integration
python solution.py
```

Alternatively, run `integration/solution.ipynb` in order, followed by `integration/exploration.ipynb` to recreate the analytical figures. Generated tables and figures are stored in the module's `outputs/` and `figures/` directories, both excluded from Git.

### 2. Cleansing, Auditing, and Customer Profiling

```bash
python scripts/prepare_inputs.py --stage cleansing --source /path/to/original/input-package.zip --destination cleansing/data
cd cleansing
python solution.py --data-dir data --output-dir outputs
```

Alternatively, open `solution.ipynb` in that directory and run all cells. The input adapter verifies the original package checksums before preparing local copies under neutral filenames. It does not change record bytes or modify the original package. The frozen VADER resource is read from the private input package; no replacement is downloaded at runtime.

## Collaboration and Individual Contribution

This is a four-person collaborative project, not work authored entirely by the repository maintainer. The maintainer primarily contributed the customer/product master tables, their field mappings and validation, dirty-data diagnosis, current-price inference, diagnostic seasonal delivery-charge models, repairs, and preservation tests.

Other collaborators contributed transaction, delivery, and review processing; missing-value recovery; outlier detection; customer profiling; and overall integration. Shared interfaces and cross-component validation were coordinated by the team. AI assisted development, with results checked through executable code and tests.

## Publication Scope

This repository retains the processing and analytical implementations, runnable unit tests, field mappings, and dependency information. Personal identifiers, signatures, private conversations, teaching documents, and legacy project labels are excluded from the published files. Saved notebook outputs are cleared to avoid disclosing record-level data. The original complete archive remains private.

No open-source licence is attached. Publication of this code does not grant redistribution rights for third-party input data.
