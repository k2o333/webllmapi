import asyncio
import time
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest import TestCase
from unittest.mock import patch, AsyncMock
from models import LLMSiteConfig
from browser_handler import LLMWebsiteAutomator

class TestPerformance(TestCase):
    """测试性能指标"""
    
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
                "response_area": "#response"
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
                "interval_seconds": 300,
                "timeout_seconds": 30,
                "failure_threshold": 3,
                "check_element_selector_key": "response_area"
            },
            pool_size=3,
            max_requests_per_instance=100,
            max_memory_per_instance_mb=512
        )
        
    async def asyncSetUp(self):
        # 创建automator实例
        self.automator = LLMWebsiteAutomator(self.config)
        await self.automator.initialize()
        
    async def asyncTearDown(self):
        # 清理automator
        await self.automator.cleanup()
        
    async def test_response_time(self):
        """测试响应时间"""
        with patch.object(self.automator, 'send_message', 
                        new_callable=AsyncMock) as mock_send:
            # 设置模拟响应
            mock_send.return_value = "Test response"
            
            # 测量响应时间
            start_time = time.time()
            await self.automator.send_prompt_and_get_response("Test prompt")
            elapsed = time.time() - start_time
            
            self.logger.info(f"Response time: {elapsed:.3f} seconds")
            self.assertLess(elapsed, 1.0)  # 响应时间应小于1秒
            
    async def test_memory_usage(self):
        """测试内存使用"""
        # 发送多个请求以观察内存变化
        with patch.object(self.automator, 'send_message', 
                        new_callable=AsyncMock) as mock_send:
            mock_send.return_value = "Test response"
            
            # 初始内存使用
            initial_memory = sum(
                [await i.get_memory_usage() 
                 for i in self.automator.browser_pool]
            )
            
            # 发送多个请求
            for _ in range(10):
                await self.automator.send_prompt_and_get_response("Test prompt")
                
            # 检查内存增长
            current_memory = sum(
                [await i.get_memory_usage() 
                 for i in self.automator.browser_pool]
            )
            memory_increase = current_memory - initial_memory
            
            self.logger.info(f"Memory increase: {memory_increase:.2f} MB")
            self.assertLess(memory_increase, 50)  # 内存增长应小于50MB

    def run_async_test(self, test_func):
        """运行异步测试的辅助方法"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(test_func())

    def runTest(self):
        """运行所有测试"""
        tests = [
            self.test_response_time,
            self.test_memory_usage
        ]
        
        for test in tests:
            self.run_async_test(test)
            self.logger.info(f"Test {test.__name__} passed")

if __name__ == "__main__":
    tester = TestPerformance()
    tester.runTest()