"""
测试运行器
==========

运行所有测试模块。
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import importlib


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("DemandClean 完整测试套件")
    print("=" * 70)

    test_modules = [
        ('配置模块', 'demandclean.tests.test_config'),
        ('检测器模块', 'demandclean.tests.test_detectors'),
        ('模型适配器', 'demandclean.tests.test_models'),
        ('DQN Agent', 'demandclean.tests.test_agents'),
        ('环境', 'demandclean.tests.test_environments'),
        ('API', 'demandclean.tests.test_api'),
        ('集成测试', 'demandclean.tests.test_integration'),
    ]

    results = []

    for name, module_name in test_modules:
        print(f"\n{'=' * 70}")
        print(f"运行: {name}")
        print(f"{'=' * 70}")

        try:
            # 动态导入并运行
            module = importlib.import_module(module_name)

            # 运行模块中的所有 test_ 函数
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
                    print(f"✗ {fn_name} 失败: {e}")
                    failed += 1

            results.append((name, passed, failed))
            print(f"\n{name}: {passed} 通过, {failed} 失败")

        except Exception as e:
            print(f"✗ 模块 {name} 加载失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, 0, 1))

    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    total_passed = sum(r[1] for r in results)
    total_failed = sum(r[2] for r in results)

    for name, passed, failed in results:
        status = "✓" if failed == 0 else "✗"
        print(f"  {status} {name}: {passed} 通过, {failed} 失败")

    print(f"\n总计: {total_passed} 通过, {total_failed} 失败")
    print("=" * 70)

    return total_failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
