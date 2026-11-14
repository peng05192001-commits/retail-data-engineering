"""Run only test classes that require no private source records."""
from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'tests'))
sys.path.insert(0, str(root / 'src'))
names = [
    'test_text_functions.TestPublishedTextContract',
    'test_Retail_mapping.TestFinalMapping',
    'test_catalogue.TestCatalogueUnitFunctions',
    'test_transactions.TestTransactionsUnitFunctions',
]
suite = unittest.TestLoader().loadTestsFromNames(names)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
