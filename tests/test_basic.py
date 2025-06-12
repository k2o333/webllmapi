import asyncio
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch, AsyncMock
from models import LLMSiteConfig, AppConfig
from browser_handler import LLMWebsiteAutomator

class TestBasicFunctionality(IsolatedAsyncioTestCase):
    """测试基础功能"""
    
    async def asyncSetUp(self):
        # 配置日志
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        
        # 创建测试配置
        self.config = LLMSiteConfig(
            id="test_site",
            name="Test Site",
            url="http://example.com",
            firefox_profile_dir="/path/to/profile",
            selectors={
                "input_area": "input[name='q']",
                "submit_button": "button[type='submit']",
                "response_area": "div"
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
            }
        )
        
        # 创建automator实例
        self.automator = LLMWebsiteAutomator(self.config)
        await self.automator.initialize()
        
    async def asyncTearDown(self):
        # 清理automator
        await self.automator.cleanup()
        
    async def test_non_streaming_response(self):
        """测试非流式响应"""
        with patch.object(self.automator, 'send_message', 
                        new_callable=AsyncMock) as mock_send:
            # 设置模拟响应
            mock_send.return_value = "Test response"
            
            # 调用非流式接口
            response = await self.automator.send_prompt_and_get_response("Test prompt")
            
            # 验证响应
            self.assertEqual(response, "Test response")
            mock_send.assert_called_once_with("Test prompt")
            
    async def test_streaming_response(self):
        """测试流式响应"""
        chunks = []
        async def mock_stream(*args, **kwargs):
            yield "Stream "
            yield "response "
            yield "chunks"
            
        async def stream_callback(chunk: str) -> None:
            chunks.append(chunk)
            
        with patch.object(self.automator, 'send_message', 
                        new_callable=AsyncMock) as mock_send:
            mock_send.side_effect = mock_stream
            # 调用流式接口
            response = await self.automator.send_prompt_and_get_response(
                "Test prompt",
                stream_callback=stream_callback
            )
            
            # 如果返回的是异步生成器，消费它
            if hasattr(response, '__aiter__'):
                async for _ in response:
                    pass  # 消费异步生成器
            
            # 验证响应
            self.assertEqual("".join(chunks), "Stream response chunks")
            
    async def test_error_handling(self):
        """测试错误处理"""
        with patch.object(self.automator, 'send_message', 
                        side_effect=Exception("Test error")):
            # 验证抛出异常
            with self.assertRaises(Exception):
                await self.automator.send_prompt_and_get_response("Test prompt")