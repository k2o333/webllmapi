import asyncio
import logging
import sys
import os
import pytest
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest import TestCase
from unittest.mock import patch, AsyncMock
from models import LLMSiteConfig
from browser_handler import LLMWebsiteAutomator

@pytest.mark.asyncio
class TestHealthCheck(TestCase):
    """测试健康检查功能"""
    
    def setUp(self):
        # 配置日志
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        
        # 创建测试配置
        self.config = LLMSiteConfig(
            id="test_site",
            name="Test Site",
            url="http://test.site",
            firefox_profile_dir="/path/to/profile",
            selectors={
                "input_area": "#input",
                "submit_button": "#submit",
                "response_area": "#response",
                "health_check_element": "#health"
            },
            response_handling={
                "type": "stream",
                "extraction_selector_key": "response_area",
                "stream_end_conditions": [
                    {
                        "type": "timeout",
                        "timeout_seconds": 30
                    }
                ]
            },
            health_check={
                "enabled": True,
                "check_element_selector_key": "health_check_element",
                "failure_threshold": 3,
                "interval_seconds": 300,
                "timeout_seconds": 30
            }
        )
        
    async def asyncSetUp(self):
        # 创建automator实例
        self.automator = LLMWebsiteAutomator(self.config)
        await self.automator.initialize()
        
    async def asyncTearDown(self):
        # 清理automator
        await self.automator.cleanup()
        
    async def test_health_check_pass(self):
        """测试健康检查通过"""
        with patch.object(self.automator, '_perform_health_check', 
                        new_callable=AsyncMock) as mock_check:
            # 设置模拟响应
            mock_check.return_value = True
            
            # 调用健康检查
            is_healthy = await self.automator.is_healthy()
            
            # 验证结果
            self.assertTrue(is_healthy)
            return None
            
    async def test_health_check_fail(self):
        """测试健康检查失败"""
        with patch.object(self.automator, '_perform_health_check', 
                        new_callable=AsyncMock) as mock_check:
            # 设置模拟响应
            mock_check.return_value = False
            
            # 调用健康检查
            is_healthy = await self.automator.is_healthy()
            
            # 验证结果
            self.assertFalse(is_healthy)
            return None
            
    async def test_instance_recovery(self):
        """测试实例回收机制"""
        # 模拟连续失败
        with patch.object(self.automator, '_perform_health_check', 
                        side_effect=Exception("Test error")):
            # 触发健康检查失败
            for _ in range(self.config.health_check.failure_threshold):
                await self.automator._health_check_loop()
                
            # 验证实例被回收重建
            self.assertEqual(self.automator.consecutive_failures, 0)
            return None

    def run_async_test(self, test_func):
        """运行异步测试的辅助方法"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(test_func())
        finally:
            loop.close()

    async def _run_test_case(self, test):
        """运行单个测试用例"""
        await self.asyncSetUp()
        try:
            await test()
            self.logger.info(f"Test {test.__name__} passed")
        finally:
            await self.asyncTearDown()

    def runTest(self):
        """运行所有测试"""
        tests = [
            self.test_health_check_pass,
            self.test_health_check_fail,
            self.test_instance_recovery
        ]
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            for test in tests:
                loop.run_until_complete(self._run_test_case(test))
        finally:
            loop.close()

if __name__ == "__main__":
    tester = TestHealthCheck()
    tester.runTest()