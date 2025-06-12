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
class TestConcurrency(TestCase):
    """测试并发处理能力"""
    
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
        
    async def send_test_request(self, request_id):
        """发送测试请求的辅助方法"""
        chunks = []
        async def mock_stream(*args, **kwargs):
            yield f"Response "
            yield f"to "
            yield f"request {request_id}"
            
        async def stream_callback(chunk: str) -> None:
            chunks.append(chunk)
            
        with patch.object(self.automator, 'send_message', 
                        new_callable=AsyncMock) as mock_send:
            # 为每个请求设置不同的响应
            mock_send.side_effect = mock_stream
            
            # 发送请求并返回响应
            response = await self.automator.send_prompt_and_get_response(
                f"Test prompt {request_id}",
                stream_callback=stream_callback
            )
            
            # 如果返回的是异步生成器，消费它
            if hasattr(response, '__aiter__'):
                async for _ in response:
                    pass  # 消费异步生成器
            
            return "".join(chunks)
            
    async def test_concurrent_requests(self):
        """测试并发请求处理"""
        # 创建并发任务
        tasks = [
            self.send_test_request(i) 
            for i in range(10)  # 发送10个并发请求
        ]
        
        # 运行并发任务
        responses = await asyncio.gather(*tasks)
        
        # 验证所有请求都得到正确处理
        for i, response in enumerate(responses):
            self.assertEqual(response, f"Response to request {i}")
            self.logger.debug(f"Request {i} got response: {response}")
            
        # 检查实例使用情况
        active_instances = sum(
            1 for instance in self.automator.browser_pool 
            if instance.request_count > 0
        )
        self.logger.info(f"Active instances: {active_instances}")
        self.assertGreaterEqual(active_instances, 1)  # 至少一个实例被使用
        
    async def test_instance_reuse(self):
        """测试实例重用"""
        # 发送多个顺序请求
        for i in range(5):
            response = await self.send_test_request(i)
            self.assertEqual(response, f"Response to request {i}")
            
        # 检查实例使用分布
        usage_counts = [
            instance.request_count 
            for instance in self.automator.browser_pool
        ]
        self.logger.info(f"Instance usage counts: {usage_counts}")
        
        # 验证请求被均衡分配到不同实例
        self.assertTrue(any(count > 1 for count in usage_counts))

    def run_async_test(self, test_func):
        """运行异步测试的辅助方法"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(test_func())

    def runTest(self):
        """运行所有测试"""
        tests = [
            self.test_concurrent_requests,
            self.test_instance_reuse
        ]
        
        for test in tests:
            self.run_async_test(test)
            self.logger.info(f"Test {test.__name__} passed")

if __name__ == "__main__":
    tester = TestConcurrency()
    tester.runTest()