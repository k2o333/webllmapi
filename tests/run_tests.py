import unittest
import logging
from .test_basic import TestBasicFunctionality
from .test_health import TestHealthCheck
from .test_performance import TestPerformance
from .test_concurrency import TestConcurrency

async def run_tests():
    """运行所有测试并生成报告"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('TestRunner')

    # 创建测试套件
    test_suite = unittest.TestSuite()
    test_classes = [
        TestBasicFunctionality,
        TestHealthCheck,
        TestPerformance,
        TestConcurrency
    ]

    # 添加所有测试用例
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # 运行测试
    logger.info("Starting test suite...")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # 输出测试结果
    logger.info("\nTest Results:")
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.2f}%")

    return result.wasSuccessful()

if __name__ == "__main__":
    asyncio.run(run_tests())