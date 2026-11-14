""" cleansing Dirty: rule-based dirty repairs, actual-change audit and QA.

Run: python dirty.py --data-zip PATH/Retail_cleansing.zip --output-dir OUTPUT
Requires monetary.py alongside this file. No network access occurs at runtime.
Missing and outlier batches supply trustworthy evidence only; neither is cleaned.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
import hashlib
import io
import json
import csv
import math
from pathlib import Path
import platform
import re
import tempfile
from typing import Any
import zipfile

import numpy as np
from monetary import cart_price, infer_current_prices, order_total, parse_cart


SCHEMA = ['order_id', 'customer_id', 'date', 'nearest_warehouse', 'shopping_cart',
          'order_price', 'delivery_charges', 'customer_lat', 'customer_long',
          'coupon_discount', 'order_total', 'season', 'is_expedited_delivery',
          'distance_to_nearest_warehouse', 'latest_customer_review', 'is_happy_customer']
AUDIT_SCHEMA = ['source_file', 'record_key', 'field_name', 'issue_type',
                'original_value', 'corrected_value', 'method', 'evidence', 'confidence']
PROTECTED = {'order_id', 'customer_id', 'shopping_cart', 'customer_long',
             'coupon_discount', 'latest_customer_review'}
SEASONS = ('Summer', 'Autumn', 'Winter', 'Spring')
NUMERIC = {'order_price', 'delivery_charges', 'customer_lat', 'customer_long',
           'coupon_discount', 'order_total', 'distance_to_nearest_warehouse'}
BOOLEAN = {'is_expedited_delivery', 'is_happy_customer'}
MODEL_DEPENDENT = {'season', 'distance_to_nearest_warehouse', 'order_total',
                   'delivery_charges', 'is_expedited_delivery'}
METHODS = {
    'date': 'calendar_format', 'season': 'calendar_and_seasonal_crosscheck',
    'nearest_warehouse': 'nearest_warehouse_haversine', 'customer_lat': 'latitude_sign_location',
    'distance_to_nearest_warehouse': 'haversine_and_delivery', 'is_happy_customer': 'frozen_vader',
    'order_price': 'current_price_cart', 'order_total': 'verified_total_formula',
    'delivery_charges': 'charge_model_and_total', 'is_expedited_delivery': 'expedited_boolean_compare',
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv_bytes(data: bytes):
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline=''))
    return list(reader.fieldnames or []), list(reader)


@dataclass
class Package:
    group: str
    archive_path: Path
    archive_hash: str
    batches: dict
    warehouses: list
    contract: dict
    lexicon_zip: bytes
    member_checksums: dict


def load_package(path: Path, group: str = 'Retail') -> Package:
    path = Path(path).resolve()
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', group):
        raise ValueError('Use the allocated safe dataset alias')
    archive_bytes = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        manifest = json.loads(archive.read('cleansing_manifest.json'))
        if manifest['group_alias'] != group:
            raise ValueError('Requested group does not match the supplied manifest')
        checksums = {}
        for entry in manifest['files']:
            data = archive.read(entry['path'])
            actual = sha256(data)
            if actual != entry['sha256'] or len(data) != entry['bytes']:
                raise ValueError(f"Supplied-file checksum/size failed: {entry['path']}")
            checksums[entry['path']] = actual
        batches = {}
        for batch in ('dirty', 'missing', 'outlier'):
            header, rows = read_csv_bytes(archive.read(f'{group}_{batch}_data.csv'))
            if header != SCHEMA or len(rows) != 500 or any(set(row) != set(SCHEMA) for row in rows):
                raise ValueError(f'Invalid input schema/row structure: {batch}')
            batches[batch] = rows
        ids = [r['order_id'] for rows in batches.values() for r in rows]
        if len(ids) != len(set(ids)) or any(not value for value in ids):
            raise ValueError('Input order IDs are missing or duplicated')
        _, warehouses = read_csv_bytes(archive.read('warehouses.csv'))
        contract = json.loads(archive.read('resource_contract.json'))
        lexicon = archive.read(contract['vader_lexicon_relative_path'])
        if sha256(lexicon) != contract['vader_lexicon_sha256']:
            raise ValueError('Frozen VADER resource checksum failed')
    return Package(group, path, sha256(archive_bytes), batches, warehouses,
                   contract, lexicon, checksums)


def calendar_date(value: str):
    # Explicit formats avoid locale-dependent or month-first guessing.
    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%Y/%m/%d'):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    match = re.fullmatch(r'([A-Za-z]{3}) (\d{1,2}), (\d{4})', value)
    months = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
    if match and match[1].title() in months:
        return datetime(int(match[3]), months.index(match[1].title()) + 1, int(match[2])).date()
    raise ValueError(f'Unsupported calendar date: {value!r}')


def season_for(date):
    return ('Summer', 'Summer', 'Autumn', 'Autumn', 'Autumn', 'Winter', 'Winter',
            'Winter', 'Spring', 'Spring', 'Spring', 'Summer')[date.month - 1]


def boolean(value) -> int:
    text = str(value).lower()
    if text in ('true', '1'):
        return 1
    if text in ('false', '0'):
        return 0
    raise ValueError(f'Unparseable boolean: {value!r}')


def nearest_location(latitude, longitude, warehouses):
    lat, lon = np.radians([float(latitude), float(longitude)])
    wlat = np.radians([float(w['latitude']) for w in warehouses])
    wlon = np.radians([float(w['longitude']) for w in warehouses])
    a = np.sin((wlat-lat)/2)**2 + np.cos(lat)*np.cos(wlat)*np.sin((wlon-lon)/2)**2
    a = np.clip(a, 0, 1)
    distance = 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    index = int(np.argmin(distance))
    return warehouses[index]['warehouse'], round(float(distance[index]), 4)


def make_sentiment_scorer(package: Package):
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    if nltk.__version__ != package.contract['nltk_version']:
        raise ValueError(f"Use NLTK {package.contract['nltk_version']}; found {nltk.__version__}")
    # A unique explicit file URI avoids a previously cached/global NLTK lexicon.
    with zipfile.ZipFile(io.BytesIO(package.lexicon_zip)) as nested:
        lexicon_text = nested.read('vader_lexicon/vader_lexicon.txt')
    with tempfile.TemporaryDirectory(prefix='cleansing-vader-') as folder:
        root = Path(folder).resolve()
        if root.parent != Path(tempfile.gettempdir()).resolve():
            raise RuntimeError('Unexpected temporary resource directory')
        path = root / 'frozen_vader.txt'
        path.write_bytes(lexicon_text)
        analyser = SentimentIntensityAnalyzer(lexicon_file=path.as_uri())

    @lru_cache(maxsize=None)
    def score(review):
        if review == 'No review submitted':
            return {'compound': None, 'happy': True}
        compound = analyser.polarity_scores(review)['compound']
        return {'compound': compound, 'happy': compound >= package.contract['compound_threshold']}
    return score


def fit_seasonal_models(missing_rows, seed=5196):
    required = ('season', 'distance_to_nearest_warehouse', 'is_expedited_delivery',
                'is_happy_customer', 'delivery_charges')
    complete = [row for row in missing_rows if all(row[f] != '' for f in required)]
    models, evidence = {}, {}
    for season in SEASONS:
        rows = [row for row in complete if row['season'] == season]
        if len(rows) < 10:
            raise ValueError(f'Insufficient reliable observations for {season}')
        X = np.array([[1, float(r['distance_to_nearest_warehouse']), boolean(r['is_expedited_delivery']),
                       boolean(r['is_happy_customer'])] for r in rows])
        y = np.array([float(r['delivery_charges']) for r in rows])
        perm = np.random.default_rng(seed).permutation(len(rows))
        cut = int(.8 * len(rows))
        train, heldout = perm[:cut], perm[cut:]
        coefficients, _, rank, _ = np.linalg.lstsq(X[train], y[train], rcond=None)
        if rank != 4:
            raise ValueError(f'Non-identifiable seasonal model: {season}')
        errors = y[heldout] - X[heldout] @ coefficients
        denominator = float(np.sum((y[heldout] - np.mean(y[heldout]))**2))
        r2 = 1 - float(np.sum(errors**2)) / denominator if denominator else float('-inf')
        training_max = float(np.max(np.abs(y[train] - X[train] @ coefficients)))
        heldout_max = float(np.max(np.abs(errors)))
        if r2 < .99:
            raise ValueError(f'Investigate inadequate held-out fit for {season}: {r2}')
        models[season] = coefficients
        evidence[season] = {
            'source': 'missing batch; complete observed model fields only',
            'predictor_order': ['intercept', 'distance', 'expedited_0_1', 'happy_0_1'],
            'coefficients': coefficients.tolist(), 'seed': seed, 'rank': int(rank),
            'training_n': len(train), 'heldout_n': len(heldout), 'heldout_r2': r2,
            'heldout_mae_aud': float(np.mean(np.abs(errors))),
            'heldout_max_error_aud': heldout_max, 'training_max_error_aud': training_max,
            'training_ids': [rows[i]['order_id'] for i in train],
            'heldout_ids': [rows[i]['order_id'] for i in heldout],
        }
    return models, evidence


@dataclass
class Context:
    package: Package
    prices: dict
    score: Any
    models: dict
    metadata: dict
    model_screen: float = .02


def build_context(package: Package, seed=5196, model_screen=.02) -> Context:
    if not math.isfinite(model_screen) or model_screen <= 0:
        raise ValueError('Diagnostic model screen must be positive and finite')
    outlier = package.batches['outlier']
    perm = np.random.default_rng(seed).permutation(len(outlier))
    train = [outlier[int(i)] for i in perm[:350]]
    heldout = [outlier[int(i)] for i in perm[350:]]
    fit = infer_current_prices(train)
    price_checks = {}
    for label, rows in [('training', train), ('heldout', heldout),
                         ('independent_missing', [r for r in package.batches['missing'] if r['order_price']])]:
        errors = [abs(cart_price(r['shopping_cart'], fit.prices) - float(r['order_price'])) for r in rows]
        if max(errors, default=0) > 1e-7:
            raise ValueError(f'Price validation failed: {label}')
        price_checks[label] = {'n': len(rows), 'exact_matches': sum(x == 0 for x in errors),
                               'max_absolute_error_aud': max(errors, default=0)}
    score = make_sentiment_scorer(package)
    sentiment_checks = {}
    for batch in ('missing', 'outlier'):
        rows = [r for r in package.batches[batch] if r['is_happy_customer']]
        disagreements = [r['order_id'] for r in rows if boolean(r['is_happy_customer']) != int(score(r['latest_customer_review'])['happy'])]
        if disagreements:
            raise ValueError(f'Frozen sentiment disagrees with trusted observations: {batch}')
        sentiment_checks[batch] = {'n': len(rows), 'disagreements': len(disagreements)}
    models, model_evidence = fit_seasonal_models(package.batches['missing'], seed)
    max_reliable_error = max(max(x['training_max_error_aud'], x['heldout_max_error_aud']) for x in model_evidence.values())
    if max_reliable_error >= model_screen:
        raise ValueError('Reliable-model errors exceed the diagnostic screen; investigate before repairs')
    monetary_rows = [r for r in package.batches['missing'] if all(r[f] for f in ('order_price', 'coupon_discount', 'delivery_charges', 'order_total'))]
    total_errors = [abs(order_total(r['order_price'], r['coupon_discount'], r['delivery_charges']) - float(r['order_total'])) for r in monetary_rows]
    if any(total_errors):
        raise ValueError('Monetary helper no longer reproduces trusted observed totals')
    metadata = {'price_inference': asdict(fit), 'price_validation': price_checks,
                'price_training_ids': [r['order_id'] for r in train],
                'price_heldout_ids': [r['order_id'] for r in heldout],
                'sentiment_validation': sentiment_checks, 'nltk_version': package.contract['nltk_version'],
                'frozen_lexicon_sha256': sha256(package.lexicon_zip), 'seasonal_models': model_evidence,
                'model_screen_aud': model_screen,
                'model_screen_note': 'Diagnostic choice, not the AUD 0.08 export tolerance or teammate outlier threshold',
                'trusted_total_validation': {'n': len(monetary_rows), 'exact_matches': len(monetary_rows)},
                'team_alignment': 'Diagnostic models and shared calendar/location/sentiment helpers require agreement with team owners before full-group integration'}
    return Context(package, fit.prices, score, models, metadata, model_screen)


def inspect_row(row, context: Context):
    if set(row) != set(SCHEMA) or any(value is None or value == '' for value in row.values()):
        raise ValueError('Dirty rows must have the complete original schema and values')
    for field in NUMERIC:
        if not math.isfinite(float(row[field])):
            raise ValueError(f'Non-finite numeric value in {field}')
    date = calendar_date(row['date'])
    warehouse, distance = nearest_location(row['customer_lat'], row['customer_long'], context.package.warehouses)
    sentiment = context.score(row['latest_customer_review'])
    price = cart_price(row['shopping_cart'], context.prices)
    total = order_total(row['order_price'], row['coupon_discount'], row['delivery_charges'])
    x = np.array([1, float(row['distance_to_nearest_warehouse']), boolean(row['is_expedited_delivery']), boolean(row['is_happy_customer'])])
    predicted = float(x @ context.models[row['season']])
    residual = float(row['delivery_charges']) - predicted
    flags = []
    if row['date'] != date.isoformat(): flags.append('date_format')
    if row['season'] != season_for(date): flags.append('date_season')
    if float(row['customer_lat']) >= 0: flags.append('latitude_hemisphere')
    if row['nearest_warehouse'] != warehouse: flags.append('warehouse_geometry')
    if abs(float(row['distance_to_nearest_warehouse']) - distance) > 1e-7: flags.append('distance_geometry')
    if boolean(row['is_happy_customer']) != int(sentiment['happy']): flags.append('review_happiness')
    if abs(float(row['order_price']) - price) > 1e-7: flags.append('cart_price')
    if abs(float(row['order_total']) - total) > 1e-7: flags.append('total_arithmetic')
    if abs(residual) > context.model_screen: flags.append('delivery_model')
    return {'flags': flags, 'expected_date': date.isoformat(), 'expected_season': season_for(date),
            'expected_warehouse': warehouse, 'expected_distance': distance, 'expected_happy': sentiment['happy'],
            'vader_compound': sentiment['compound'], 'expected_price': price, 'expected_total': total,
            'predicted_delivery': predicted, 'delivery_residual': residual}


def replacement_text(field, value, original):
    if field in ('order_price', 'order_total', 'delivery_charges'):
        return f'{value:.2f}'
    if field == 'distance_to_nearest_warehouse':
        return f'{value:.4f}'
    if field == 'customer_lat':
        # Sign is the hypothesis: retain every original magnitude digit.
        return '-' + original.lstrip('+')
    return str(value)


def candidate_values(row, observation):
    flags = set(observation['flags'])
    candidates = []
    mappings = [('date_format', 'date', 'expected_date'), ('date_season', 'season', 'expected_season'),
                ('warehouse_geometry', 'nearest_warehouse', 'expected_warehouse'),
                ('distance_geometry', 'distance_to_nearest_warehouse', 'expected_distance'),
                ('review_happiness', 'is_happy_customer', 'expected_happy'),
                ('cart_price', 'order_price', 'expected_price'), ('total_arithmetic', 'order_total', 'expected_total')]
    for flag, field, key in mappings:
        if flag in flags:
            candidates.append((field, observation[key]))
    if 'latitude_hemisphere' in flags:
        candidates.append(('customer_lat', -float(row['customer_lat'])))
    if 'delivery_model' in flags:
        candidates.append(('is_expedited_delivery', not bool(boolean(row['is_expedited_delivery']))))
    if flags & {'delivery_model', 'total_arithmetic'}:
        inverse = float(row['order_total']) - float(row['order_price']) * (1 - float(row['coupon_discount']) / 100)
        # Candidate cents near both independent estimates. Final acceptance also
        # requires exact monetary consistency and every other row check.
        centres = [round(observation['predicted_delivery'], 2), round(inverse, 2)]
        values = {round(centre + offset/100, 2) for centre in centres for offset in range(-2, 3)}
        candidates.extend(('delivery_charges', value) for value in sorted(values) if value >= 0)
    result = []
    for field, value in candidates:
        text = replacement_text(field, value, row[field])
        same = boolean(row[field]) == boolean(text) if field in BOOLEAN else abs(float(row[field])-float(text)) < 1e-8 if field in NUMERIC else row[field] == text
        if not same and (field, text) not in result:
            result.append((field, text))
    return result


class AmbiguousDecision(ValueError):
    pass


def choose_unique(accepted):
    if len(accepted) != 1:
        raise AmbiguousDecision(f'Expected one defensible single-field repair; found {len(accepted)}')
    return accepted[0]


def diagnose_row(row, context: Context):
    before = inspect_row(row, context)
    if not before['flags']:
        return None
    accepted, alternatives = [], []
    for field, text in candidate_values(row, before):
        hypothetical = dict(row)
        hypothetical[field] = text
        try:
            after = inspect_row(hypothetical, context)
            record = {'field': field, 'original': row[field], 'replacement': text,
                      'remaining_flags': after['flags'], 'after_residual': after['delivery_residual']}
        except (ValueError, KeyError, OverflowError) as exc:
            record = {'field': field, 'original': row[field], 'replacement': text,
                      'remaining_flags': [str(exc)], 'after_residual': None}
        alternatives.append(record)
        if not record['remaining_flags']:
            accepted.append(record)
    selected = choose_unique(accepted)
    return {'selected': selected, 'before': before, 'alternatives': alternatives}


@dataclass
class RepairPlan:
    cleaned: list
    decisions: dict
    unresolved: list


def repair_rows(raw_rows, context: Context) -> RepairPlan:
    cleaned, decisions, unresolved = [], {}, []
    for row in raw_rows:
        result = dict(row)
        try:
            decision = diagnose_row(row, context)
            if decision:
                choice = decision['selected']
                if choice['field'] in PROTECTED:
                    raise ValueError('A proposed repair targets a protected field')
                result[choice['field']] = choice['replacement']
                decisions[row['order_id']] = decision
        except (ValueError, KeyError, OverflowError) as exc:
            unresolved.append({'order_id': row['order_id'], 'reason': str(exc)})
        cleaned.append(result)
    return RepairPlan(cleaned, decisions, unresolved)


def evidence_text(row, decision):
    choice, obs = decision['selected'], decision['before']
    field = choice['field']
    detail = {
        'date': f"same calendar day; required ISO {obs['expected_date']}",
        'season': f"date implies {obs['expected_season']}; charge corroborates seasonal choice",
        'nearest_warehouse': f"minimum haversine distance gives {obs['expected_warehouse']} at {obs['expected_distance']:.4f} km",
        'customer_lat': 'sign change reconciles original warehouse and distance; longitude retained',
        'distance_to_nearest_warehouse': f"haversine gives {obs['expected_distance']:.4f} km; original charge corroborates distance",
        'is_happy_customer': f"frozen VADER compound={obs['vader_compound']}; threshold >=0.05; exact sentinel handled",
        'order_price': f"validated cart equations give {obs['expected_price']:.2f}; original total preserved",
        'order_total': f"cart price verified; observed charge fits seasonal model; required total={obs['expected_total']:.2f}",
        'delivery_charges': f"model charge={obs['predicted_delivery']:.6f}; replacement preserves original total exactly",
        'is_expedited_delivery': 'only opposite boolean explains charge while preserving valid total and independently checked predictors',
    }[field]
    return (f"M1:{row['order_id']}; {detail}; residual before={obs['delivery_residual']:.6f}, "
            f"after={choice['after_residual']:.6f}; see dirty_validation.json decisions")


def make_audit(raw_rows, cleaned_rows, decisions, group):
    original = {r['order_id']: r for r in raw_rows}
    entries = []
    for row in cleaned_rows:
        raw = original[row['order_id']]
        # Diff actual rows, rather than logging every flag or candidate.
        for field in SCHEMA:
            if raw[field] == row[field]:
                continue
            decision = decisions[row['order_id']]
            choice = decision['selected']
            if (choice['field'], choice['replacement']) != (field, row[field]):
                raise ValueError('Actual change differs from the selected evidence-based decision')
            entries.append(dict(zip(AUDIT_SCHEMA, [f'{group}_dirty_data.csv', row['order_id'], field,
                'dirty', raw[field], row[field], METHODS[field], evidence_text(raw, decision),
                'medium' if field in MODEL_DEPENDENT else 'high'])))
    return entries


def validate(raw_rows, cleaned_rows, audit_rows, context, decisions,
             data_header=None, audit_header=None):
    checks = []
    def check(name, passed, observed):
        checks.append({'check': name, 'passed': bool(passed), 'observed': observed})
    header = SCHEMA if data_header is None else data_header
    audit_cols = AUDIT_SCHEMA if audit_header is None else audit_header
    schema_ok = header == SCHEMA and all(set(r) == set(SCHEMA) for r in cleaned_rows)
    audit_schema_ok = audit_cols == AUDIT_SCHEMA and all(set(r) == set(AUDIT_SCHEMA) for r in audit_rows)
    check('exact_16_column_schema_and_order', schema_ok, header)
    check('exact_9_column_audit_schema_and_order', audit_schema_ok, audit_cols)
    raw_ids = [r['order_id'] for r in raw_rows]
    ids = [r.get('order_id') for r in cleaned_rows]
    id_ok = len(cleaned_rows) == 500 and len(set(ids)) == len(ids) and set(ids) == set(raw_ids)
    check('all_500_original_unique_ids', id_ok, {'rows': len(ids), 'unique': len(set(ids)), 'same_id_set': set(ids) == set(raw_ids)})
    if not (schema_ok and audit_schema_ok and id_ok):
        return {'passed': False, 'checks': checks}
    original = {r['order_id']: r for r in raw_rows}
    diff, protected_changes, row_failures, changed_per_row = {}, [], {}, Counter()
    residuals = []
    for row in cleaned_rows:
        try:
            observation = inspect_row(row, context)
            if observation['flags']:
                row_failures[row['order_id']] = observation['flags']
            residuals.append(abs(observation['delivery_residual']))
        except (ValueError, KeyError, OverflowError) as exc:
            row_failures[row['order_id']] = [str(exc)]
        for field in SCHEMA:
            if original[row['order_id']][field] != row[field]:
                key = (row['order_id'], field)
                diff[key] = (original[row['order_id']][field], row[field])
                changed_per_row[row['order_id']] += 1
                if field in PROTECTED:
                    protected_changes.append(list(key))
    check('complete_parseable_values_and_business_relationships', not row_failures,
          {'checked_rows': len(cleaned_rows), 'failing_rows': row_failures})
    check('seasonal_residuals', len(residuals) == 500 and max(residuals, default=math.inf) <= context.model_screen,
          {'maximum_absolute_residual_aud': max(residuals, default=None), 'screen_aud': context.model_screen})
    check('six_error_free_fields_preserved', not protected_changes, protected_changes)
    check('at_most_one_changed_field_per_row', all(n <= 1 for n in changed_per_row.values()), dict(Counter(changed_per_row.values())))
    expected = {(id, d['selected']['field']): (d['selected']['original'], d['selected']['replacement']) for id, d in decisions.items()}
    check('only_supported_changes_all_other_values_identical', diff == expected,
          {'actual_changed_fields': len(diff), 'planned_changed_fields': len(expected),
           'unchanged_cells': 500*len(SCHEMA)-len(diff), 'unchanged_rows': 500-len(changed_per_row)})
    audit_map = {}
    audit_errors = []
    for entry in audit_rows:
        key = (entry['record_key'], entry['field_name'])
        if key in audit_map:
            audit_errors.append(f'Duplicate entry {key}')
        audit_map[key] = (entry['original_value'], entry['corrected_value'])
        field = entry['field_name']
        if (entry['source_file'] != f'{context.package.group}_dirty_data.csv' or entry['issue_type'] != 'dirty'
            or entry['method'] != METHODS.get(field)
            or entry['confidence'] != ('medium' if field in MODEL_DEPENDENT else 'high')
            or not entry['evidence'].startswith(f"M1:{entry['record_key']};")):
            audit_errors.append(f'Invalid audit metadata {key}')
        if key in expected:
            original_row = original[entry['record_key']]
            if entry['evidence'] != evidence_text(original_row, decisions[entry['record_key']]):
                audit_errors.append(f'Evidence does not match recorded decision {key}')
    check('audit_exactly_reconciles_actual_field_changes', audit_map == diff and not audit_errors,
          {'audit_rows': len(audit_rows), 'actual_differences': len(diff), 'metadata_errors': audit_errors})
    check('source_archive_bytes_unchanged', sha256(context.package.archive_path.read_bytes()) == context.package.archive_hash,
          context.package.archive_hash)
    return {'passed': all(c['passed'] for c in checks), 'checks': checks,
            'changes_by_field': dict(Counter(field for _, field in diff)),
            'confidence_counts': dict(Counter(row['confidence'] for row in audit_rows))}


def write_csv(path, header, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def export_plan(plan, context, output_dir, audit_name='audit_log_dirty.csv'):
    output_dir = Path(output_dir).resolve()
    if Path(audit_name).name != audit_name or not audit_name.endswith('.csv'):
        raise ValueError('audit_name must be a simple CSV filename')
    data_name = f'{context.package.group}_dirty_clean.csv'
    if audit_name == data_name or audit_name in context.package.member_checksums:
        raise ValueError('Audit filename conflicts with a data filename')
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'dirty_validation.json'
    if plan.unresolved:
        report_path.write_text(json.dumps({'passed': False, 'unresolved': plan.unresolved,
                               'csv_exports_created_this_run': False}, indent=2)+'\n', encoding='utf-8')
        raise AmbiguousDecision('Unresolved decisions: inspect dirty_validation.json; no CSVs written this run')
    raw = context.package.batches['dirty']
    audit = make_audit(raw, plan.cleaned, plan.decisions, context.package.group)
    before = validate(raw, plan.cleaned, audit, context, plan.decisions)
    if not before['passed']:
        raise ValueError(f'In-memory validation failed: {before}')
    audit_path = output_dir / audit_name
    if audit_path.exists():
        _, existing = read_csv_bytes(audit_path.read_bytes())
        if any(r.get('source_file') != f'{context.package.group}_dirty_data.csv' for r in existing):
            raise ValueError('Refusing to overwrite audit entries belonging to teammates')
    data_path = output_dir / data_name
    write_csv(data_path, SCHEMA, plan.cleaned)
    write_csv(audit_path, AUDIT_SCHEMA, audit)
    data_header, reread = read_csv_bytes(data_path.read_bytes())
    audit_header, reread_audit = read_csv_bytes(audit_path.read_bytes())
    readback = validate(raw, reread, reread_audit, context, plan.decisions, data_header, audit_header)
    report = {'component': 'Dirty dirty data only', 'group': context.package.group,
              'passed': readback['passed'], 'unresolved': plan.unresolved,
              'python_version': platform.python_version(), 'numpy_version': np.__version__,
              'archive_sha256': context.package.archive_hash,
              'input_member_checksums': context.package.member_checksums,
              'evidence': context.metadata, 'decisions': plan.decisions,
              'readback': readback, 'output_files': {data_name: sha256(data_path.read_bytes()),
                                                   audit_name: sha256(audit_path.read_bytes())},
              'confidence_definitions': {'high': 'Direct published calculation/reference rule with independent corroboration',
                                        'medium': 'Field attribution additionally relies on validated diagnostic seasonal models; team alignment pending',
                                        'low': 'Insufficient evidence; never auto-repaired by this workflow'},
              'audit_scope': 'Dirty entries only; merge with teammates into final audit_log.csv'}
    report_path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    if not readback['passed']:
        raise ValueError('Export read-back failed; do not use these files until resolved')
    return report


def run(data_zip, output_dir, group='Retail', audit_name='audit_log_dirty.csv', seed=5196):
    package = load_package(Path(data_zip), group)
    original_snapshot = json.dumps(package.batches, sort_keys=True)
    context = build_context(package, seed)
    plan = repair_rows(package.batches['dirty'], context)
    if json.dumps(package.batches, sort_keys=True) != original_snapshot:
        raise RuntimeError('Source records were unexpectedly mutated')
    return export_plan(plan, context, output_dir, audit_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-zip', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--group', default='Retail')
    parser.add_argument('--audit-name', default='audit_log_dirty.csv')
    parser.add_argument('--seed', type=int, default=5196)
    args = parser.parse_args()
    report = run(args.data_zip, args.output_dir, args.group, args.audit_name, args.seed)
    print(json.dumps({'passed': report['passed'], 'unresolved': report['unresolved'],
                      'readback': report['readback'], 'output_files': report['output_files']}, indent=2))


if __name__ == '__main__':
    main()
