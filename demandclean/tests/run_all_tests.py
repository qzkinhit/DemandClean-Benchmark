"""
Test Runner
===========

Run all test modules.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import importlib


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("DemandClean full test suite")
    print("=" * 70)

    test_modules = [
        ('Config module', 'demandclean.tests.test_config'),
        ('Detector module', 'demandclean.tests.test_detectors'),
        ('Model adapters', 'demandclean.tests.test_models'),
        ('DQN Agent', 'demandclean.tests.test_agents'),
        ('Environments', 'demandclean.tests.test_environments'),
        ('API', 'demandclean.tests.test_api'),
        ('Integration', 'demandclean.tests.test_integration'),
    ]

    results = []

    for name, module_name in test_modules:
        print(f"\n{'=' * 70}")
        print(f"Running: {name}")
        print(f"{'=' * 70}")

        try:
            # Dynamically import and run
            module = importlib.import_module(module_name)

            # Run all test_ functions in the module
            test_functions = [
                (fn_name, getattr(module, fn_name))
                for fn_name in dir(module)
                if fn_name.startswith('test_') and callable(getattr(module, fn_name))
            ]

            passed = 0
            failed = 0

            for fn_name, fn in test_functions:
                try:
                    fn()
                    passed += 1
                except Exception as e:
                    print(f"✗ {fn_name} failed: {e}")
                    failed += 1

            results.append((name, passed, failed))
            print(f"\n{name}: {passed} passed, {failed} failed")

        except Exception as e:
            print(f"✗ Module {name} failed to load: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, 0, 1))

    # Summary
    print("\n" + "=" * 70)
    print("Test summary")
    print("=" * 70)

    total_passed = sum(r[1] for r in results)
    total_failed = sum(r[2] for r in results)

    for name, passed, failed in results:
        status = "✓" if failed == 0 else "✗"
        print(f"  {status} {name}: {passed} passed, {failed} failed")

    print(f"\nTotal: {total_passed} passed, {total_failed} failed")
    print("=" * 70)

    return total_failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
