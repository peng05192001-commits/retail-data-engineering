"""Run monetary tests and Dirty repair/preservation regression tests.

python test_dirty.py --data-zip PATH/Retail_cleansing.zip --output-dir OUTPUT
Tests read real exports. Tampering probes use copies; clean outputs are not edited.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import dirty as m
import test_monetary as monetary_tests


def make_tests(package, context, plan, audit, output_dir, scratch_dir):
    raw = package.batches['dirty']

    class DirtyRepairTests(unittest.TestCase):
        def validate(self, rows=None, entries=None, header=None):
            return m.validate(raw, plan.cleaned if rows is None else rows,
                              audit if entries is None else entries, context, plan.decisions,
                              data_header=header)

        def test_reviewed_repairs_and_no_unresolved_decisions(self):
            self.assertEqual(plan.unresolved, [])
            self.assertEqual(len(plan.decisions), 83)  # Observed dataset regression, not a repair rule.
            self.assertEqual(len(audit), 83)
            self.assertTrue(self.validate()['passed'])

        def test_raw_objects_remain_unchanged(self):
            snapshot = copy.deepcopy(package.batches)
            m.repair_rows(raw, context)
            self.assertEqual(package.batches, snapshot)
            self.assertEqual(m.sha256(package.archive_path.read_bytes()), package.archive_hash)

        def test_decisions_do_not_depend_on_order_ids_or_row_order(self):
            remapped = []
            old_for_new = {}
            for i, row in enumerate(reversed(raw)):
                new = dict(row)
                new_id = f'test-key-{i}'
                old_for_new[new_id] = row['order_id']
                new['order_id'] = new_id
                remapped.append(new)
            repeated = m.repair_rows(remapped, context)
            self.assertEqual(repeated.unresolved, [])
            observed = {old_for_new[id]: (d['selected']['field'], d['selected']['replacement'])
                        for id, d in repeated.decisions.items()}
            expected = {id: (d['selected']['field'], d['selected']['replacement'])
                        for id, d in plan.decisions.items()}
            self.assertEqual(observed, expected)

        def test_idempotence_no_second_round_of_changes(self):
            repeated = m.repair_rows(plan.cleaned, context)
            self.assertEqual(repeated.unresolved, [])
            self.assertEqual(repeated.decisions, {})
            self.assertEqual(repeated.cleaned, plan.cleaned)

        def test_actual_disk_exports_read_back_and_match_plan(self):
            header, rows = m.read_csv_bytes((output_dir/'Retail_dirty_clean.csv').read_bytes())
            audit_header, entries = m.read_csv_bytes((output_dir/'audit_log_dirty.csv').read_bytes())
            result = m.validate(raw, rows, entries, context, plan.decisions, header, audit_header)
            self.assertTrue(result['passed'])
            self.assertEqual(rows, plan.cleaned)
            self.assertEqual(entries, audit)

        def test_missing_duplicate_or_incorrect_audit_is_rejected(self):
            wrong_value = copy.deepcopy(audit)
            wrong_value[0]['corrected_value'] = 'not-the-exported-value'
            wrong_evidence = copy.deepcopy(audit)
            wrong_evidence[0]['evidence'] = 'M1:untraceable'
            for entries in (audit[:-1], audit+[audit[0]], wrong_value, wrong_evidence):
                with self.subTest(entries=len(entries)):
                    self.assertFalse(self.validate(entries=entries)['passed'])

        def test_unchanged_review_text_is_protected_even_if_sentiment_same(self):
            changed = copy.deepcopy(plan.cleaned)
            changed[0]['latest_customer_review'] += ' '
            result = self.validate(rows=changed)
            self.assertFalse(result['passed'])
            checks = {r['check']: r['passed'] for r in result['checks']}
            self.assertFalse(checks['six_error_free_fields_preserved'])
            self.assertFalse(checks['only_supported_changes_all_other_values_identical'])

        def test_duplicate_missing_and_unexpected_ids_are_rejected(self):
            duplicate = copy.deepcopy(plan.cleaned)
            duplicate[1]['order_id'] = duplicate[0]['order_id']
            unexpected = copy.deepcopy(plan.cleaned)
            unexpected[0]['order_id'] = 'unexpected-test-id'
            for rows in (duplicate, unexpected, plan.cleaned[:-1]):
                with self.subTest(rows=len(rows)):
                    self.assertFalse(self.validate(rows=rows)['passed'])

        def test_column_reordering_is_rejected(self):
            header = list(m.SCHEMA)
            header[0], header[1] = header[1], header[0]
            self.assertFalse(self.validate(header=header)['passed'])

        def test_zero_or_multiple_supported_candidates_are_not_guessed(self):
            for accepted in ([], [{'field': 'a'}, {'field': 'b'}]):
                with self.subTest(candidates=len(accepted)), self.assertRaises(m.AmbiguousDecision):
                    m.choose_unique(accepted)

        def test_unresolved_decision_blocks_csv_exports(self):
            blocked = m.RepairPlan(copy.deepcopy(raw), {}, [{'order_id': 'synthetic', 'reason': 'ambiguous evidence'}])
            with tempfile.TemporaryDirectory(prefix='m1-blocked-', dir=scratch_dir) as directory:
                temp = Path(directory).resolve()
                self.assertEqual(temp.parent, scratch_dir.resolve())
                with self.assertRaises(m.AmbiguousDecision):
                    m.export_plan(blocked, context, temp)
                self.assertEqual(list(temp.glob('*.csv')), [])
                self.assertFalse(json.loads((temp/'dirty_validation.json').read_text())['passed'])

        def test_unrecognised_date_is_flagged_instead_of_invented(self):
            row = dict(raw[0])
            row['date'] = 'unknown calendar day'
            result = m.repair_rows([row], context)
            self.assertEqual(len(result.unresolved), 1)
            self.assertEqual(result.cleaned, [row])
            self.assertEqual(result.decisions, {})

        def test_frozen_sentiment_matches_trusted_batches(self):
            checks = context.metadata['sentiment_validation']
            self.assertEqual(checks['missing'], {'n': 490, 'disagreements': 0})
            self.assertEqual(checks['outlier'], {'n': 500, 'disagreements': 0})
            self.assertEqual(context.score('No review submitted'), {'compound': None, 'happy': True})

        def test_diagnostic_screen_sensitivity(self):
            from dataclasses import replace
            expected = {id: (d['selected']['field'], d['selected']['replacement']) for id, d in plan.decisions.items()}
            for screen in (.01, .05):
                repeated = m.repair_rows(raw, replace(context, model_screen=screen))
                actual = {id: (d['selected']['field'], d['selected']['replacement']) for id, d in repeated.decisions.items()}
                self.assertEqual(repeated.unresolved, [])
                self.assertEqual(actual, expected)

        def test_second_export_is_byte_identical(self):
            with tempfile.TemporaryDirectory(prefix='m1-rerun-', dir=scratch_dir) as directory:
                temp = Path(directory).resolve()
                self.assertEqual(temp.parent, scratch_dir.resolve())
                second = m.export_plan(plan, context, temp)
                self.assertTrue(second['passed'])
                for name in ('Retail_dirty_clean.csv', 'audit_log_dirty.csv'):
                    self.assertEqual((temp/name).read_bytes(), (output_dir/name).read_bytes())

        def test_export_does_not_replace_teammate_audit_entries(self):
            with tempfile.TemporaryDirectory(prefix='m1-audit-guard-', dir=scratch_dir) as directory:
                temp = Path(directory).resolve()
                self.assertEqual(temp.parent, scratch_dir.resolve())
                entry = dict(audit[0])
                entry['source_file'] = 'Retail_missing_data.csv'
                target = temp/'audit_log_dirty.csv'
                m.write_csv(target, m.AUDIT_SCHEMA, [entry])
                previous = target.read_bytes()
                with self.assertRaises(ValueError):
                    m.export_plan(plan, context, temp)
                self.assertEqual(target.read_bytes(), previous)
                self.assertFalse((temp/'Retail_dirty_clean.csv').exists())

    return DirtyRepairTests


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcomes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.outcomes.append({'test': test.id(), 'passed': True})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.outcomes.append({'test': test.id(), 'passed': False, 'kind': 'failure'})

    def addError(self, test, err):
        super().addError(test, err)
        self.outcomes.append({'test': test.id(), 'passed': False, 'kind': 'error'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-zip', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scratch-dir', type=Path, default=Path(tempfile.gettempdir()))
    args = parser.parse_args()
    package = m.load_package(args.data_zip)
    context = m.build_context(package)
    plan = m.repair_rows(package.batches['dirty'], context)
    audit = m.make_audit(package.batches['dirty'], plan.cleaned, plan.decisions, package.group)
    monetary_report = monetary_tests.read_evidence(args.data_zip)
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite([
        loader.loadTestsFromTestCase(monetary_tests.MonetaryUnitTests),
        loader.loadTestsFromTestCase(monetary_tests.integration_tests(monetary_report)),
        loader.loadTestsFromTestCase(make_tests(package, context, plan, audit, args.output_dir.resolve(), args.scratch_dir.resolve())),
    ])
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(suite)
    report = {'executed_at_utc': datetime.now(timezone.utc).isoformat(), 'run': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
              'passed': result.wasSuccessful(), 'outcomes': result.outcomes}
    (args.output_dir/'dirty_test_results.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'outcomes'}, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
