# wrapper1cobu
  # core
    # services
      # context_service.py
      
```
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel
import asyncio

class DialogTurn(BaseModel):
    role: str  # "user" or "bot"
    content: str
    timestamp: str = datetime.now().isoformat()

class ContextConfig:
    def __init__(self, max_history: int = 5, persist: bool = False):
        self.max_history = max_history
        self.persist = persist

class ContextService:
    def __init__(self, config: Optional[ContextConfig] = None):
        self.config = config or ContextConfig()
        self.contexts: Dict[str, List[DialogTurn]] = {}
        self.locks: Dict[str, asyncio.Lock] = {}

    async def _get_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self.locks:
            self.locks[session_id] = asyncio.Lock()
        return self.locks[session_id]

    async def add_turn(self, session_id: str, turn: DialogTurn) -> None:
        async with await self._get_lock(session_id):
            if session_id not in self.contexts:
                self.contexts[session_id] = []
            
            self.contexts[session_id].append(turn)
            
            if len(self.contexts[session_id]) > self.config.max_history:
                self.contexts[session_id].pop(0)

    async def get_context(self, session_id: str) -> List[DialogTurn]:
        return self.contexts.get(session_id, []).copy()

    async def clear_context(self, session_id: str) -> None:
        async with await self._get_lock(session_id):
            if session_id in self.contexts:
                del self.contexts[session_id]
      ```

    # context.py
    
```
from typing import Dict, List

class ContextManager:
    def __init__(self):
        self.sessions: Dict[str, List[dict]] = {}
    
    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"role": role, "content": content})
        
    def get_context(self, session_id: str, max_history=5) -> List[dict]:
        history = self.sessions.get(session_id, [])
        return history[-max_history:] if max_history > 0 else history.copy()
    ```

  # examples
    # streaming_example.py
    
```
#!/usr/bin/env python3
"""
流式响应示例脚本
演示如何使用流式响应功能与LLM进行交互
"""

import asyncio
import time
import sys
import json
import argparse
from typing import Dict, Any, List

# 添加项目根目录到路径
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from browser_automator import LLMWebsiteAutomator
from config_manager import get_llm_site_config

async def stream_demo(model_name: str, prompt: str, format_output: bool = True):
    """演示流式响应功能
    
    Args:
        model_name: 要使用的模型名称
        prompt: 要发送的提示
        format_output: 是否格式化输出（彩色、带时间戳）
    """
    # 初始化自动化器
    site_config = get_llm_site_config(model_name)
    automator = LLMWebsiteAutomator(site_config)
    
    try:
        # 连接到浏览器
        print(f"正在连接到 {model_name}...")
        await automator.initialize()
        
        # 准备性能指标
        start_time = time.time()
        chunk_count = 0
        total_chars = 0
        
        # 发送消息并处理流式响应
        print(f"\n>>> 用户: {prompt}\n")
        print(f">>> {model_name} (流式响应):")
        
        # 获取流式响应
        stream_gen = await automator.send_message(prompt, stream=True)
        
        # 处理流式响应
        async for chunk in stream_gen:
            chunk_count += 1
            total_chars += len(chunk)
            
            # 输出格式化或原始响应
            if format_output:
                # 计算已用时间
                elapsed = time.time() - start_time
                # 输出带有时间戳和颜色的文本
                sys.stdout.write(f"\033[32m{chunk}\033[0m")
                sys.stdout.flush()
            else:
                # 简单输出
                sys.stdout.write(chunk)
                sys.stdout.flush()
        
        # 输出性能统计
        elapsed = time.time() - start_time
        print(f"\n\n--- 性能统计 ---")
        print(f"总时间: {elapsed:.2f}秒")
        print(f"数据块数量: {chunk_count}")
        print(f"总字符数: {total_chars}")
        print(f"平均速度: {total_chars/elapsed:.2f} 字符/秒")
        print(f"平均块大小: {total_chars/chunk_count:.2f} 字符/块") if chunk_count > 0 else None
        
    except Exception as e:
        print(f"错误: {e}")
    finally:
        # 关闭浏览器
        await automator.close()

async def compare_streaming_vs_regular(model_name: str, prompt: str):
    """比较流式响应和常规响应的性能差异
    
    Args:
        model_name: 要使用的模型名称
        prompt: 要发送的提示
    """
    # 初始化自动化器
    site_config = get_llm_site_config(model_name)
    automator = LLMWebsiteAutomator(site_config)
    
    try:
        # 连接到浏览器
        print(f"正在连接到 {model_name}...")
        await automator.initialize()
        
        # 测试流式响应
        print(f"\n>>> 测试流式响应")
        stream_start = time.time()
        
        stream_gen = await automator.send_message(prompt, stream=True)
        stream_response = ""
        first_token_time = None
        
        async for chunk in stream_gen:
            if not first_token_time and chunk.strip():
                first_token_time = time.time()
            stream_response += chunk
            # 输出进度指示
            sys.stdout.write(".")
            sys.stdout.flush()
        
        stream_end = time.time()
        stream_total = stream_end - stream_start
        stream_ttft = first_token_time - stream_start if first_token_time else None
        
        print(f"\n流式响应完成，总时间: {stream_total:.2f}秒")
        if stream_ttft:
            print(f"首个token时间: {stream_ttft:.2f}秒")
        
        # 测试常规响应
        print(f"\n>>> 测试常规响应")
        regular_start = time.time()
        
        regular_response = await automator.send_message(prompt, stream=False)
        
        regular_end = time.time()
        regular_total = regular_end - regular_start
        
        print(f"常规响应完成，总时间: {regular_total:.2f}秒")
        
        # 比较结果
        print(f"\n--- 性能比较 ---")
        print(f"流式响应总时间: {stream_total:.2f}秒")
        print(f"常规响应总时间: {regular_total:.2f}秒")
        print(f"差异: {regular_total - stream_total:.2f}秒 ({(regular_total/stream_total - 1)*100:.1f}%)")
        
        if stream_ttft:
            print(f"流式响应首个token时间: {stream_ttft:.2f}秒")
            print(f"用户体验改善: {regular_total - stream_ttft:.2f}秒 ({(regular_total/stream_ttft - 1)*100:.1f}%)")
        
        # 验证响应内容一致性
        content_match = stream_response.strip() == regular_response.strip()
        print(f"响应内容一致: {'是' if content_match else '否'}")
        
    except Exception as e:
        print(f"错误: {e}")
    finally:
        # 关闭浏览器
        await automator.close()

async def api_streaming_demo(prompt: str, api_url: str = "http://localhost:8000/v1/chat/completions"):
    """演示通过API使用流式响应
    
    Args:
        prompt: 要发送的提示
        api_url: API端点URL
    """
    import aiohttp
    
    # 准备请求数据
    request_data = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }
    
    print(f"通过API发送流式请求: {api_url}")
    print(f"提示: {prompt}")
    print("\n响应:")
    
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, json=request_data) as response:
            if response.status != 200:
                print(f"API错误: {response.status}")
                print(await response.text())
                return
                
            # 处理SSE流
            buffer = ""
            async for line in response.content:
                line = line.decode('utf-8')
                buffer += line
                
                if buffer.endswith('\n\n'):
                    for event in buffer.strip().split('\n\n'):
                        if not event.startswith('data: '):
                            continue
                            
                        data = event[6:]  # 移除 'data: ' 前缀
                        
                        if data == '[DONE]':
                            continue
                            
                        try:
                            json_data = json.loads(data)
                            content = json_data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                            if content:
                                sys.stdout.write(content)
                                sys.stdout.flush()
                        except json.JSONDecodeError:
                            print(f"无法解析JSON: {data}")
                    
                    buffer = ""
    
    elapsed = time.time() - start_time
    print(f"\n\n总时间: {elapsed:.2f}秒")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="流式响应演示")
    parser.add_argument("--model", "-m", default="default", help="要使用的模型名称")
    parser.add_argument("--prompt", "-p", default="请写一篇关于人工智能的短文，包括其历史、现状和未来发展。", help="要发送的提示")
    parser.add_argument("--compare", "-c", action="store_true", help="比较流式和常规响应")
    parser.add_argument("--api", "-a", action="store_true", help="使用API进行流式响应")
    parser.add_argument("--api-url", default="http://localhost:8000/v1/chat/completions", help="API端点URL")
    parser.add_argument("--raw", "-r", action="store_true", help="输出原始响应，不带格式")
    
    args = parser.parse_args()
    
    if args.api:
        asyncio.run(api_streaming_demo(args.prompt, args.api_url))
    elif args.compare:
        asyncio.run(compare_streaming_vs_regular(args.model, args.prompt))
    else:
        asyncio.run(stream_demo(args.model, args.prompt, not args.raw))

if __name__ == "__main__":
    main()
    ```

  # tests
    # integration
    # __init__.py
    
```
# 使tests目录成为Python包
from .run_tests import run_tests

__all__ = ['run_tests']
    ```

    # run_tests.py
    
```
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
    ```

    # simple_test.py
    
```
import pytest
import asyncio
from browser_handler import LLMWebsiteAutomator
from models import LLMSiteConfig

# 测试配置
TEST_CONFIG = {
    "SKIP_BROWSER_TESTS": True,  # 强制跳过浏览器测试
    "TEST_MESSAGE": "测试消息",
    "EXPECTED_RESPONSE_CONTAINS": ["response", "answer", "reply"]
}

@pytest.fixture
def site_config():
    """测试用的站点配置"""
    return LLMSiteConfig(
        id="test-site",
        name="Test Site",
        url="about:blank",  # 使用空白页避免连接错误
        pool_size=1,
        max_requests_per_instance=10,
        max_memory_per_instance_mb=500,
        selectors={
            "input_area": "textarea",
            "submit_button": "button",
            "response_area": ".response"
        },
        response_handling={
            "type": "full",
            "extraction_selector_key": "response_area",
            "extraction_method": "textContent"
        },
        health_check={
            "enabled": False,
            "interval_seconds": 300,
            "timeout_seconds": 10,
            "failure_threshold": 3,
            "check_element_selector_key": "input_area"
        }
    )

@pytest.mark.asyncio
async def test_browser_initialization(site_config):
    """测试浏览器初始化"""
    pytest.skip("第一阶段跳过浏览器测试")

@pytest.mark.asyncio
async def test_basic_functionality(site_config):
    """测试基本功能"""
    pytest.skip("第一阶段跳过浏览器测试")

@pytest.mark.asyncio
async def test_error_handling(site_config):
    """测试错误处理"""
    pytest.skip("第一阶段跳过浏览器测试")

@pytest.mark.asyncio
async def test_resource_monitoring(site_config):
    """测试资源监控"""
    pytest.skip("第一阶段跳过浏览器测试")
    ```

    # test_basic.py
    
```
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
    ```

    # test_browser_handler copy.py
    
```
import asyncio
import pytest
import logging
import tenacity
from pathlib import Path
from typing import Optional

# 禁用测试中的重试装饰器
original_retry = tenacity.retry

def disable_retry(*decorator_args, **decorator_kwargs):
    """
    返回一个不执行重试的装饰器函数。
    这个函数本身可以接受 tenacity.retry 装饰器工厂的参数，但会忽略它们。
    """
    def decorator(func):
        return func  # 直接返回原函数，不添加重试逻辑
    return decorator

# 在测试环境中替换retry装饰器
tenacity.retry = disable_retry

from config_manager import get_llm_site_config, load_config
from browser_handler import LLMWebsiteAutomator
from utils import setup_logging

# 设置日志
logger = setup_logging(log_level="DEBUG")

# 测试配置
TEST_CONFIG = {
    "SKIP_BROWSER_TESTS": False,  # 启用浏览器测试
    "MOCK_MODE": True,  # 启用mock模式
    "TEST_MESSAGE": "Hello, this is a test message.",
    "EXPECTED_RESPONSE_CONTAINS": ["hello", "hi", "greetings"],  # 任意一个都可以
    "MOCK_RESPONSES": {
        "default": {
            "status": 200,
            "text": "Mock response from local ChatGPT clone",
            "streaming": [
                "Mock chunk 1",
                "Mock chunk 2",
                "Mock chunk 3"
            ]
        }
    }
}

@pytest.fixture
def config():
    """加载配置文件"""
    try:
        return load_config()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

@pytest.fixture
def site_config(config):
    """获取站点配置并添加mock设置"""
    try:
        # 获取第一个启用的站点配置
        enabled_sites = [site for site in config.llm_sites if site.enabled]
        if not enabled_sites:
            pytest.skip("No enabled LLM sites found in configuration")

        # 创建配置对象并直接添加mock_responses属性
        site = enabled_sites[0].model_copy(deep=True)
        site.mock_responses = TEST_CONFIG["MOCK_RESPONSES"]
        return site

    except Exception as e:
        logger.error(f"Failed to get site config: {e}")
        raise

@pytest.mark.asyncio
async def test_config_loading(config, site_config):
    """测试配置加载"""
    # 测试全局配置
    assert config.global_settings.timeout > 0
    assert config.global_settings.max_retries > 0
    
    # 测试站点配置
    assert site_config.id
    assert site_config.name
    assert site_config.url.startswith(("http://", "https://"))
    
    # 测试选择器配置
    assert "input_area" in site_config.selectors
    assert "submit_button" in site_config.selectors
    
    logger.info("Configuration loading test passed")

@pytest.mark.asyncio
async def test_automator_creation(site_config):
    """测试自动化器创建"""
    if TEST_CONFIG["SKIP_BROWSER_TESTS"] and not TEST_CONFIG["MOCK_MODE"]:
        pytest.skip("Browser tests are disabled or mock mode not enabled")
    
    automator = None
    try:
        automator = LLMWebsiteAutomator(site_config)
        assert automator is not None
        assert automator.config == site_config
        logger.info("Automator creation test passed")
    except Exception as e:
        logger.error(f"Failed to create automator: {e}")
        raise
    finally:
        if automator:
            await automator.cleanup()

@pytest.mark.asyncio
async def test_browser_initialization(site_config):
    """测试浏览器初始化(mock模式)"""
    # 替换print为日志输出（更符合pytest规范）
    logger.info("\n=== 开始浏览器初始化测试(Mock模式) ===")
    
    automator = None
    try:
        logger.info("创建自动化器实例...")
        automator = LLMWebsiteAutomator(site_config)
        logger.info("初始化mock浏览器实例...")
        await automator.initialize()
        
        # 验证mock实例池
        assert len(automator.browser_pool) == automator.config.pool_size, "实例池大小不匹配配置"
        for instance in automator.browser_pool:
            assert instance.is_mock, "实例未标记为mock模式"
            assert instance.request_count == 0, "mock实例初始请求计数不为0"
            
        logger.info("Mock browser initialization test passed")
    except Exception as e:
        logger.error(f"Failed to initialize browser: {e}")
        raise
    finally:
        if automator:
            await automator.cleanup()

@pytest.mark.asyncio
async def test_basic_functionality(site_config):
    """测试基本功能(mock模式)"""
    automator = None
    try:
        automator = LLMWebsiteAutomator(site_config)
        await automator.initialize()
        
        # 测试发送消息并获取mock响应
        test_message = TEST_CONFIG["TEST_MESSAGE"]
        response = await automator.send_message(test_message)
        
                    # 验证响应
        if isinstance(response, str):
            assert response == site_config.mock_responses["default"]["text"]
            logger.info(f"Received mock response: {response}")
        else:
            chunks = []
            async for chunk in response:
                chunks.append(chunk)
                logger.info(f"Received mock chunk: {chunk}")
            assert chunks == site_config.mock_responses["default"]["streaming"]
        
        logger.info("Mock basic functionality test passed")
    except Exception as e:
        logger.error(f"Failed to test basic functionality: {e}")
        raise
    finally:
        if automator:
            await automator.cleanup()

@pytest.mark.asyncio
async def test_streaming_response(site_config):
    """测试流式响应功能(mock模式)"""
    automator = None
    try:
        automator = LLMWebsiteAutomator(site_config)
        await automator.initialize()
        
        # 测试mock流式响应
        response = await automator.send_message(TEST_CONFIG["TEST_MESSAGE"])
        chunks = []
        async for chunk in response:
            chunks.append(chunk)
            logger.info(f"Received mock streaming chunk: {chunk}")
        
        # 验证mock流式响应
        assert chunks == site_config.mock_responses["default"]["streaming"]
        
        logger.info("Mock streaming response test passed")
    except Exception as e:
        logger.error(f"Failed to test streaming response: {e}")
        raise
    finally:
        if automator:
            await automator.cleanup()

@pytest.mark.asyncio
async def test_streaming_error_handling(site_config):
    """测试流式响应错误处理(mock模式)"""
    automator = None
    try:
        automator = LLMWebsiteAutomator(site_config)
        await automator.initialize()
        
        # 修改mock响应为错误状态
        original_mock = site_config.mock_responses["default"]
        site_config.mock_responses["default"] = {
            "is_error": True,
            "error_message": "Mock streaming error",
            "streaming": ["Error chunk 1", "Error chunk 2"]
        }
        
        # 测试错误处理
        with pytest.raises(Exception) as excinfo:
            response = await automator.send_message(TEST_CONFIG["TEST_MESSAGE"])
            if not isinstance(response, str):
                async for chunk in response:
                    logger.info(f"Received error chunk: {chunk}")
        
        # 检查异常信息
        error_msg = str(excinfo.value)
        assert any([
            "Mock streaming error" in error_msg,
            hasattr(excinfo.value, 'last_attempt') and "Mock streaming error" in str(excinfo.value.last_attempt.exception())
        ]), "未捕获到预期的错误消息"
        
        # 恢复原始mock配置
        site_config.mock_responses["default"] = original_mock
        logger.info("Mock streaming error handling test passed")
    except Exception as e:
        logger.error(f"Failed to test streaming error handling: {e}")
        raise
    finally:
        if automator:
            await automator.cleanup()

@pytest.mark.asyncio
async def test_streaming_timeout(site_config):
    """测试流式响应超时处理(mock模式)"""
    automator = None
    try:
        automator = LLMWebsiteAutomator(site_config)
        await automator.initialize()
        
        # 修改mock响应为超时状态
        original_mock = site_config.mock_responses["default"]
        site_config.mock_responses["default"] = {
            "is_timeout": True,
            "timeout_ms": 100,  # 设置很短的超时时间
            "streaming": ["Slow chunk 1", "Slow chunk 2"]
        }
        
        # 测试超时处理
        with pytest.raises(Exception) as excinfo:
            response = await automator.send_message(TEST_CONFIG["TEST_MESSAGE"])
            if not isinstance(response, str):
                async for chunk in response:
                    logger.info(f"Received chunk before timeout: {chunk}")
        
        # 检查异常信息
        error_msg = str(excinfo.value)
        assert any([
            isinstance(excinfo.value, asyncio.TimeoutError),
            "timeout" in error_msg.lower(),
            hasattr(excinfo.value, 'last_attempt') and isinstance(excinfo.value.last_attempt.exception(), asyncio.TimeoutError)
        ]), "未捕获到预期的超时异常"
        
        # 恢复原始mock配置
        site_config.mock_responses["default"] = original_mock
        logger.info("Mock streaming timeout test passed")
    except Exception as e:
        logger.error(f"Failed to test streaming timeout: {e}")
        raise
    finally:
        if automator:
            await automator.cleanup()

@pytest.mark.asyncio
async def test_error_handling(site_config):
    """测试错误处理(mock模式)"""
    try:
        automator = LLMWebsiteAutomator(site_config)
        await automator.initialize()
        
        # 修改mock响应为错误状态
        original_mock = site_config.mock_responses["default"]
        site_config.mock_responses["default"] = {
            "is_error": True,
            "error_message": "Mock error response",
            "text": "Error: Invalid selector"
        }
        
        # 测试错误处理
        with pytest.raises(Exception) as excinfo:  # 使用通用Exception
            await automator.send_message(TEST_CONFIG["TEST_MESSAGE"])
            
        # 检查异常信息
        error_msg = str(excinfo.value)
        assert any([
            "Mock error response" in error_msg,
            hasattr(excinfo.value, 'last_attempt') and "Mock error response" in str(excinfo.value.last_attempt.exception())
        ]), "未捕获到预期的错误消息"
        
    except Exception as e:  # 保留通用异常捕获作为最后防线
        logger.error(f"Failed to test error handling: {e}")
        raise

@pytest.mark.asyncio
# 测试资源监控(mock模式)
async def test_resource_monitoring(site_config):
    """测试资源监控(mock模式)"""
    automator = None
    try:
        automator = LLMWebsiteAutomator(site_config)
        await automator.initialize()
        
        # 获取mock实例
        mock_instance = automator.browser_pool[0]
        
        # 验证mock资源监控
        assert mock_instance.is_mock, "实例未标记为mock模式"
        assert mock_instance.mock_memory_usage > 0, "mock内存使用量应大于0"
        assert mock_instance.mock_memory_usage < automator.config.max_memory_per_instance_mb, "mock内存使用量超过配置限制"
        
        # 验证初始请求计数
        assert mock_instance.request_count == 0, "mock实例初始请求计数不为0"
        
        # 发送测试消息
        test_message = TEST_CONFIG["TEST_MESSAGE"]
        response = await automator.send_message(test_message)
        if not isinstance(response, str):
            async for _ in response:
                pass
        
        # 发送测试消息后验证
        assert mock_instance.request_count == 1, "发送消息后请求计数未增加"
        
        logger.info("Mock resource monitoring test passed")
    except Exception as e:
        logger.error(f"Failed to test resource monitoring: {e}")
        raise
    finally:
        if automator:
            await automator.cleanup()

def run_tests():
    """运行所有测试"""
    pytest.main([__file__, "-v"])

if __name__ == "__main__":
    run_tests()
    ```

    # test_browser_handler.py
    
```
import asyncio
import pytest
import logging
import tenacity
from pathlib import Path
from typing import Optional

# 禁用测试中的重试装饰器
original_retry = tenacity.retry

def disable_retry(*decorator_args, **decorator_kwargs):
    """
    返回一个不执行重试的装饰器函数。
    这个函数本身可以接受 tenacity.retry 装饰器工厂的参数，但会忽略它们。
    """
    def decorator(func):
        return func  # 直接返回原函数，不添加重试逻辑
    return decorator

# 在测试环境中替换retry装饰器
tenacity.retry = disable_retry

from config_manager import get_llm_site_config, load_config
from browser_handler import LLMWebsiteAutomator
from utils import setup_logging

# 设置日志
logger = setup_logging(log_level="DEBUG")

# 测试配置
TEST_CONFIG = {
    "SKIP_BROWSER_TESTS": False,  # 启用浏览器测试
    "MOCK_MODE": True,  # 启用mock模式
    "TEST_MESSAGE": "Hello, this is a test message.",
    "EXPECTED_RESPONSE_CONTAINS": ["hello", "hi", "greetings"],  # 任意一个都可以
    "MOCK_RESPONSES": {
        "default": {
            "status": 200,
            "text": "Mock response from local ChatGPT clone",
            "streaming": [
                "Mock chunk 1",
                "Mock chunk 2",
                "Mock chunk 3"
            ]
        }
    }
}

@pytest.fixture
def config():
    """加载配置文件"""
    try:
        return load_config()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

@pytest.fixture
def site_config(config):
    """获取站点配置并添加mock设置"""
    try:
        # 获取第一个启用的站点配置
        enabled_sites = [site for site in config.llm_sites if site.enabled]
        if not enabled_sites:
            pytest.skip("No enabled LLM sites found in configuration")

        # 创建配置对象并直接添加mock_responses属性
        site = enabled_sites[0].model_copy(deep=True)
        site.mock_responses = TEST_CONFIG["MOCK_RESPONSES"]
        return site

    except Exception as e:
        logger.error(f"Failed to get site config: {e}")
        raise

@pytest.mark.asyncio
async def test_config_loading(config, site_config):
    """测试配置加载"""
    # 测试全局配置
    assert config.global_settings.timeout > 0
    assert config.global_settings.max_retries > 0
    
    # 测试站点配置
    assert site_config.id
    assert site_config.name
    assert site_config.url.startswith(("http://", "https://"))
    
    # 测试选择器配置
    assert "input_area" in site_config.selectors
    assert "submit_button" in site_config.selectors
    
    logger.info("Configuration loading test passed")

@pytest.mark.asyncio
async def test_automator_creation(site_config):
    """测试自动化器创建"""
    if TEST_CONFIG["SKIP_BROWSER_TESTS"] and not TEST_CONFIG["MOCK_MODE"]:
        pytest.skip("Browser tests are disabled or mock mode not enabled")
    
    automator = None
    try:
        automator = LLMWebsiteAutomator(site_config)
        assert automator is not None
        assert automator.config == site_config
        logger.info("Automator creation test passed")
    except Exception as e:
        logger.error(f"Failed to create automator: {e}")
        raise
    finally:
        if automator:
            await automator.cleanup()

@pytest.mark.asyncio
async def test_browser_initialization(site_config):
    """测试浏览器初始化(mock模式)"""
    # 替换print为日志输出（更符合pytest规范）
    logger.info("\n=== 开始浏览器初始化测试(Mock模式) ===")
    
    automator = None
    try:
        logger.info("创建自动化器实例...")
        automator = LLMWebsiteAutomator(site_config)
        logger.info("初始化mock浏览器实例...")
        await automator.initialize()
        
        # 验证mock实例池
        assert len(automator.browser_pool) == automator.config.pool_size, "实例池大小不匹配配置"
        for instance in automator.browser_pool:
            assert instance.is_mock, "实例未标记为mock模式"
            assert instance.request_count == 0, "mock实例初始请求计数不为0"
            
        logger.info("Mock browser initialization test passed")
    except Exception as e:
        logger.error(f"Failed to initialize browser: {e}")
        raise
    finally:
        if automator:
            await automator.cleanup()

@pytest.mark.asyncio
async def test_basic_functionality(site_config):
    """测试基本功能(mock模式)"""
    automator = None
    try:
        automator = LLMWebsiteAutomator(site_config)
        await automator.initialize()
        
        # 测试发送消息并获取mock响应
        test_message = TEST_CONFIG["TEST_MESSAGE"]
        response = await automator.send_message(test_message)
        
                    # 验证响应
        if isinstance(response, str):
            assert response == site_config.mock_responses["default"]["text"]
            logger.info(f"Received mock response: {response}")
        else:
            chunks = []
            async for chunk in response:
                chunks.append(chunk)
                logger.info(f"Received mock chunk: {chunk}")
            assert chunks == site_config.mock_responses["default"]["streaming"]
        
        logger.info("Mock basic functionality test passed")
    except Exception as e:
        logger.error(f"Failed to test basic functionality: {e}")
        raise
    finally:
        if automator:
            await automator.cleanup()

@pytest.mark.asyncio
async def test_streaming_response(site_config):
    """测试流式响应功能(mock模式)"""
    automator = None
    try:
        automator = LLMWebsiteAutomator(site_config)
        await automator.initialize()
        
        # 测试mock流式响应
        response = await automator.send_message(TEST_CONFIG["TEST_MESSAGE"])
        chunks = []
        async for chunk in response:
            chunks.append(chunk)
            logger.info(f"Received mock streaming chunk: {chunk}")
        
        # 验证mock流式响应
        assert chunks == site_config.mock_responses["default"]["streaming"]
        
        logger.info("Mock streaming response test passed")
    except Exception as e:
        logger.error(f"Failed to test streaming response: {e}")
        raise
    finally:
        if automator:
            await automator.cleanup()

@pytest.mark.asyncio
async def test_streaming_error_handling(site_config):
    """测试流式响应错误处理(mock模式)"""
    automator = None
    try:
        automator = LLMWebsiteAutomator(site_config)
        await automator.initialize()
        
        # 修改mock响应为错误状态
        original_mock = site_config.mock_responses["default"]
        site_config.mock_responses["default"] = {
            "is_error": True,
            "error_message": "Mock streaming error",
            "streaming": ["Error chunk 1", "Error chunk 2"]
        }
        
        # 测试错误处理
        with pytest.raises(Exception) as excinfo:
            response = await automator.send_message(TEST_CONFIG["TEST_MESSAGE"])
            if not isinstance(response, str):
                async for chunk in response:
                    logger.info(f"Received error chunk: {chunk}")
        
        # 检查异常信息
        error_msg = str(excinfo.value)
        assert any([
            "Mock streaming error" in error_msg,
            hasattr(excinfo.value, 'last_attempt') and "Mock streaming error" in str(excinfo.value.last_attempt.exception())
        ]), "未捕获到预期的错误消息"
        
        # 恢复原始mock配置
        site_config.mock_responses["default"] = original_mock
        logger.info("Mock streaming error handling test passed")
    except Exception as e:
        logger.error(f"Failed to test streaming error handling: {e}")
        raise
    finally:
        if automator:
            await automator.cleanup()

@pytest.mark.asyncio
async def test_streaming_timeout(site_config):
    """测试流式响应超时处理(mock模式)"""
    automator = None
    try:
        automator = LLMWebsiteAutomator(site_config)
        await automator.initialize()
        
        # 修改mock响应为超时状态
        original_mock = site_config.mock_responses["default"]
        site_config.mock_responses["default"] = {
            "is_timeout": True,
            "timeout_ms": 100,  # 设置很短的超时时间
            "streaming": ["Slow chunk 1", "Slow chunk 2"]
        }
        
        # 测试超时处理
        with pytest.raises(Exception) as excinfo:
            response = await automator.send_message(TEST_CONFIG["TEST_MESSAGE"])
            if not isinstance(response, str):
                async for chunk in response:
                    logger.info(f"Received chunk before timeout: {chunk}")
        
        # 检查异常信息
        error_msg = str(excinfo.value)
        assert any([
            isinstance(excinfo.value, asyncio.TimeoutError),
            "timeout" in error_msg.lower(),
            hasattr(excinfo.value, 'last_attempt') and isinstance(excinfo.value.last_attempt.exception(), asyncio.TimeoutError)
        ]), "未捕获到预期的超时异常"
        
        # 恢复原始mock配置
        site_config.mock_responses["default"] = original_mock
        logger.info("Mock streaming timeout test passed")
    except Exception as e:
        logger.error(f"Failed to test streaming timeout: {e}")
        raise
    finally:
        if automator:
            await automator.cleanup()

@pytest.mark.asyncio
async def test_error_handling(site_config):
    """测试错误处理(mock模式)"""
    automator = None
    original_mock = site_config.mock_responses["default"]  # 获取原始mock
    try:
        automator = LLMWebsiteAutomator(site_config)
        await automator.initialize()
        
        # 修改mock响应为错误状态
        site_config.mock_responses["default"] = {
            "is_error": True,
            "error_message": "Mock error response",
            "text": "Error: Invalid selector"
        }
        
        # 测试错误处理
        with pytest.raises(Exception) as excinfo:  # 使用通用Exception
            await automator.send_message(TEST_CONFIG["TEST_MESSAGE"])
            
        # 检查异常信息
        error_msg = str(excinfo.value)
        assert any([
            "Mock error response" in error_msg,
            hasattr(excinfo.value, 'last_attempt') and "Mock error response" in str(excinfo.value.last_attempt.exception())
        ]), "未捕获到预期的错误消息"
        
        logger.info("Mock error handling test passed")
    except Exception as e:  # 保留通用异常捕获作为最后防线
        logger.error(f"Failed to test error handling: {e}")
        raise
    finally:
        # 恢复原始mock配置
        site_config.mock_responses["default"] = original_mock
        if automator:
            await automator.cleanup()

@pytest.mark.asyncio
# 测试资源监控(mock模式)
async def test_resource_monitoring(site_config):
    """测试资源监控(mock模式)"""
    automator = None
    try:
        automator = LLMWebsiteAutomator(site_config)
        await automator.initialize()
        
        # 获取mock实例
        mock_instance = automator.browser_pool[0]
        
        # 验证mock资源监控
        assert mock_instance.is_mock, "实例未标记为mock模式"
        assert mock_instance.mock_memory_usage > 0, "mock内存使用量应大于0"
        assert mock_instance.mock_memory_usage < automator.config.max_memory_per_instance_mb, "mock内存使用量超过配置限制"
        
        # 验证初始请求计数
        assert mock_instance.request_count == 0, "mock实例初始请求计数不为0"
        
        # 发送测试消息
        test_message = TEST_CONFIG["TEST_MESSAGE"]
        response = await automator.send_message(test_message)
        if not isinstance(response, str):
            async for _ in response:
                pass
        
        # 发送测试消息后验证
        assert mock_instance.request_count == 1, "发送消息后请求计数未增加"
        
        logger.info("Mock resource monitoring test passed")
    except Exception as e:
        logger.error(f"Failed to test resource monitoring: {e}")
        raise
    finally:
        if automator:
            await automator.cleanup()

def run_tests():
    """运行所有测试"""
    pytest.main([__file__, "-v"])

if __name__ == "__main__":
    run_tests()
    ```

    # test_concurrency.py
    
```
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
    ```

    # test_config.py
    
```
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional

def test_config_loading() -> bool:
    """测试配置加载"""
    try:
        print("\n=== 开始配置测试 ===")
        
        # 检查配置文件是否存在
        config_path = Path("config.yaml")
        if not config_path.exists():
            print("错误: 找不到配置文件 config.yaml")
            return False
        
        print("✓ 配置文件存在")
        
        # 尝试导入配置管理器
        try:
            print("正在导入配置管理器...")
            from config_manager import get_config, get_llm_site_config, get_enabled_llm_sites
            print("✓ 成功导入配置管理器")
        except ImportError as e:
            print(f"错误: 无法导入配置管理器 - {str(e)}")
            return False
        
        # 尝试加载配置
        try:
            print("正在加载配置...")
            config = get_config()
            print("✓ 配置加载成功")
            
            # 测试全局设置
            print("\n全局设置:")
            print(f"  超时时间: {config.global_settings.timeout}秒")
            print(f"  最大重试次数: {config.global_settings.max_retries}")
            
            # 测试 API 设置
            print("\nAPI设置:")
            print(f"  OpenAI模型: {config.api_settings.openai.model}")
            print(f"  温度: {config.api_settings.openai.temperature}")
            
            # 测试代理设置
            print("\n代理设置:")
            print(f"  代理启用状态: {config.proxy_settings.enabled}")
            
            # 测试 LLM 站点
            enabled_sites = get_enabled_llm_sites()
            print(f"\n已启用的LLM站点 ({len(enabled_sites)}):")
            
            for site in enabled_sites:
                print("\n站点信息:")
                print(f"  ID: {site.id}")
                print(f"  名称: {site.name}")
                print(f"  URL: {site.url}")
                print(f"  实例池大小: {site.pool_size}")
                print(f"  每实例最大请求数: {site.max_requests_per_instance}")
                print(f"  每实例最大内存(MB): {site.max_memory_per_instance_mb}")
                
                # 测试选择器
                print("\n  选择器配置:")
                print(f"    输入区域: {site.selectors.get('input_area')}")
                print(f"    提交按钮: {site.selectors.get('submit_button')}")
                
                # 测试响应处理
                print("\n  响应处理配置:")
                print(f"    处理类型: {site.response_handling.type}")
                print(f"    提取选择器: {site.response_handling.extraction_selector_key}")
                
                # 测试健康检查
                print("\n  健康检查配置:")
                print(f"    启用状态: {site.health_check.enabled}")
                print(f"    检查间隔: {site.health_check.interval_seconds}秒")
            
        except Exception as e:
            print(f"错误: 配置加载失败 - {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False
        
        print("\n=== 配置测试完成 ===")
        return True
        
    except Exception as e:
        print(f"错误: 测试过程中发生异常 - {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    try:
        print("开始执行测试...")
        success = test_config_loading()
        if success:
            print("\n✅ 配置测试通过")
            sys.exit(0)
        else:
            print("\n❌ 配置测试失败")
            sys.exit(1)
    except Exception as e:
        print(f"测试执行失败: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)
    ```

    # test_env.py
    
```
import sys
import os
from pathlib import Path

print("=== 环境测试 ===")
print(f"Python路径: {sys.executable}")
print(f"工作目录: {os.getcwd()}")
print(f"文件存在: {Path('config.yaml').exists()}")
print("=== 测试完成 ===")
    ```

    # test_health.py
    
```
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
    ```

    # test_performance.py
    
```
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
    ```

    # test_stage1.py
    
```
import pytest

@pytest.mark.asyncio
async def test_browser_initialization():
    pytest.skip("第一阶段测试跳过 - 浏览器初始化")

@pytest.mark.asyncio
async def test_basic_functionality():
    pytest.skip("第一阶段测试跳过 - 基础功能")

@pytest.mark.asyncio
async def test_error_handling():
    pytest.skip("第一阶段测试跳过 - 错误处理")

@pytest.mark.asyncio
async def test_resource_monitoring():
    pytest.skip("第一阶段测试跳过 - 资源监控")
    ```

    # test_streaming.py
    
```
import asyncio
import pytest
import pytest_asyncio
from contextlib import AsyncExitStack
from typing import List

pytestmark = pytest.mark.asyncio  # 标记所有测试为异步测试

# 模拟资源类，用于测试上下文管理
class MockResource:
    def __init__(self):
        self.is_active = False
    
    async def __aenter__(self):
        self.is_active = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.is_active = False

@pytest_asyncio.fixture
async def automator():
    """创建一个配置为测试模式的自动化器实例"""
    from browser_handler import LLMWebsiteAutomator, LLMSiteConfig
    
    # 创建测试配置
    config = LLMSiteConfig(
        id="test-config",
        name="Test Configuration",
        firefox_profile_dir="",
        mock_mode=True,
        mock_responses={
            'default': {
                'text': 'Mock response',
                'streaming': ['Chunk 1', 'Chunk 2', 'Chunk 3']
            }
        },
        pool_size=1,
        url="http://example.com",
        use_stealth=False,
        playwright_launch_options={},
        selectors={
            'input_area': '#input',
            'submit_button': '#submit',
            'response_area': '#response',
            'health_check_element': '#status'
        },
        max_requests_per_instance=100,
        max_memory_per_instance_mb=500,
        response_handling={
            'type': 'stream',
            'stream_end_conditions': [],
            'stream_poll_interval_ms': 100,
            'stream_error_handling': {
                'max_retries': 3,
                'retry_delay_ms': 100,
                'fallback_to_full_response': False
            },
            'extraction_selector_key': 'response_area',
            'extraction_method': 'textContent',
            'full_text_wait_strategy': []
        },
        health_check={
            'enabled': False,
            'interval_seconds': 60,
            'timeout_seconds': 10,
            'failure_threshold': 3,
            'check_element_selector_key': 'health_check_element'
        }
    )
    
    # 创建并初始化automator
    auto_instance = LLMWebsiteAutomator(config)
    await auto_instance.initialize()  # 确保浏览器池被初始化
    
    yield auto_instance
    
    # Cleanup after the test
    await auto_instance.cleanup()

async def test_streaming_response(automator):
    """测试基本的流式响应功能"""
    message = "测试消息"
    
    # 创建一个任务来处理流式响应
    chunks = []
    async def stream_callback(chunk: str) -> None:
        chunks.append(chunk)
    
    # 发送消息并等待响应
    response = await automator.send_prompt_and_get_response(message, stream_callback=stream_callback)
    
    # 如果返回的是异步生成器，消费它
    if hasattr(response, '__aiter__'):
        async for _ in response:
            pass  # 消费异步生成器
    
    # 验证响应
    assert len(chunks) > 0, "应该接收到流式响应块"
    assert isinstance(chunks[0], str), "响应块应该是字符串"

async def test_streaming_performance(automator):
    """测试流式响应的性能指标"""
    message = "测试性能"
    
    # 创建一个任务来处理流式响应
    chunks: List[str] = []
    start_time = asyncio.get_event_loop().time()
    
    async def stream_callback(chunk: str) -> None:
        chunks.append(chunk)
    
    # 发送消息并等待响应
    response = await automator.send_prompt_and_get_response(message, stream_callback=stream_callback)
    
    # 如果返回的是异步生成器，消费它
    if hasattr(response, '__aiter__'):
        async for _ in response:
            pass  # 消费异步生成器
    
    # 计算性能指标
    end_time = asyncio.get_event_loop().time()
    duration = end_time - start_time
    total_length = sum(len(chunk) for chunk in chunks)
    
    # 验证性能指标
    assert duration > 0, "处理时间应该大于0"
    assert total_length > 0, "总响应长度应该大于0"
    
    # 计算并验证吞吐量
    throughput = total_length / duration  # 字节/秒
    assert throughput > 0, "吞吐量应该大于0"

async def test_streaming_error_handling(automator):
    """测试流式响应的错误处理"""
    message = "触发错误"
    callback_entered_event = asyncio.Event()

    async def error_callback(chunk: str) -> None:
        print(f"DEBUG: error_callback entered with chunk: {chunk}")
        callback_entered_event.set()  # 标记回调函数被进入
        raise ValueError("测试错误")
    
    print("DEBUG: Before try-except block in test_streaming_error_handling")
    exception_caught = False
    try:
        print("DEBUG: Calling send_prompt_and_get_response with error_callback")
        await automator.send_prompt_and_get_response(message, stream_callback=error_callback)
        # 如果代码执行到这里，说明 send_prompt_and_get_response 没有按预期抛出异常
        print("DEBUG: send_prompt_and_get_response completed WITHOUT raising expected ValueError.")
    except ValueError as e:
        print(f"DEBUG: Caught ValueError in test: '{e}'")
        if str(e) == "测试错误":
            exception_caught = True
            print("DEBUG: Correct ValueError was caught.")
        else:
            print(f"DEBUG: Incorrect ValueError caught: {e}. Expected '测试错误'.")
    except Exception as e:
        print(f"DEBUG: Caught an unexpected exception: {type(e).__name__} - {e}")
    
    assert callback_entered_event.is_set(), "Error callback was never entered."
    assert exception_caught, "The expected ValueError('测试错误') was not caught."
    print("DEBUG: test_streaming_error_handling finished assertions.")

async def test_streaming_context_management(automator):
    """测试流式响应的上下文管理"""
    message = "测试上下文管理"
    
    # 创建一个任务来处理流式响应
    chunks = []
    async def stream_callback(chunk: str) -> None:
        chunks.append(chunk)
    
    # 使用上下文管理器
    async with AsyncExitStack() as stack:
        # 模拟资源分配
        resource = MockResource()
        await stack.enter_async_context(resource)
        
        # 在上下文中发送消息
        response = await automator.send_prompt_and_get_response(
            message, stream_callback=stream_callback
        )
        
        # 如果返回的是异步生成器，消费它
        if hasattr(response, '__aiter__'):
            async for _ in response:
                pass  # 消费异步生成器
        
        # 验证上下文管理
        assert resource.is_active, "资源应该在上下文中保持活动状态"
    
    # 验证上下文退出后资源已释放
    assert not resource.is_active, "退出上下文后资源应该被释放"
    assert len(chunks) > 0, "应该接收到流式响应块"

async def test_streaming_end_conditions(automator):
    """测试流式响应的结束条件"""
    message = "测试结束条件"
    
    # 创建一个任务来处理流式响应
    chunks = []
    is_completed = False
    
    async def stream_callback(chunk: str) -> None:
        chunks.append(chunk)
        if len(chunks) >= 3:  # 假设我们期望3个块
            nonlocal is_completed
            is_completed = True
    
    # 发送消息并等待响应
    response = await automator.send_prompt_and_get_response(message, stream_callback=stream_callback)
    
    # 如果返回的是异步生成器，消费它
    if hasattr(response, '__aiter__'):
        async for _ in response:
            pass  # 消费异步生成器
    
    # 验证结束条件
    assert is_completed, "流式响应应该正常完成"
    assert len(chunks) >= 3, "应该接收到预期数量的响应块"

async def test_streaming_cancellation(automator):
    """测试流式响应的取消操作"""
    message = "测试取消"
    chunks_received: List[str] = []
    callback_entered_event = asyncio.Event()
    callback_sleep_finished_event = asyncio.Event()  # 新增：标记回调中的sleep是否正常完成

    async def stream_callback_for_cancel(chunk: str) -> None:
        print(f"DEBUG (cancel_test): stream_callback entered with chunk: {chunk}")
        callback_entered_event.set()
        chunks_received.append(chunk)
        print(f"DEBUG (cancel_test): stream_callback appended '{chunk}', now sleeping for 0.5s")
        try:
            await asyncio.sleep(0.5)
            # 如果能执行到这里，说明 sleep 没有被 CancelledError 打断
            print(f"DEBUG (cancel_test): stream_callback sleep finished normally.")
            callback_sleep_finished_event.set()
        except asyncio.CancelledError:
            print(f"DEBUG (cancel_test): stream_callback's sleep was cancelled.")
            raise  # 必须重新抛出 CancelledError，否则它会被吞掉
    
    print("DEBUG (cancel_test): Creating task for send_prompt_and_get_response")
    task = asyncio.create_task(
        automator.send_prompt_and_get_response(message, stream_callback=stream_callback_for_cancel)
    )
    
    # 等待回调函数开始执行其内部的 sleep
    try:
        print("DEBUG (cancel_test): Waiting for callback to enter (timeout 0.3s)")
        await asyncio.wait_for(callback_entered_event.wait(), timeout=0.3)  # 给予足够时间让第一个 chunk 的回调被调用
        print("DEBUG (cancel_test): Callback confirmed entered. Task should be in callback's sleep.")
    except asyncio.TimeoutError:
        print("DEBUG (cancel_test): Timeout! Callback was not entered. This is a problem.")
        # 即使回调没进入，也尝试取消任务，看看会发生什么
    
    print("DEBUG (cancel_test): Cancelling task.")
    task.cancel()
    
    cancelled_correctly = False
    try:
        print("DEBUG (cancel_test): Awaiting the cancelled task.")
        await task
        # 如果 await task 正常返回 (没有抛 CancelledError)，说明取消失败或任务已在取消前完成
        print("DEBUG (cancel_test): Task completed without raising CancelledError. This is a failure.")
    except asyncio.CancelledError:
        print("DEBUG (cancel_test): asyncio.CancelledError was correctly caught by 'await task'.")
        cancelled_correctly = True
    except Exception as e:
        print(f"DEBUG (cancel_test): An unexpected exception was caught by 'await task': {type(e).__name__} - {e}")
    
    assert callback_entered_event.is_set(), "Cancellation test callback was never entered."
    assert cancelled_correctly, "Task was not cancelled with asyncio.CancelledError as expected."
    # 如果任务被正确取消，回调中的 sleep 不应该正常完成
    assert not callback_sleep_finished_event.is_set(), "Callback's sleep should have been interrupted by cancellation, not finished normally."
    
    # 断言接收到的块数量
    if callback_entered_event.is_set():  # 只有回调被进入了，才可能有数据
        assert len(chunks_received) == 1, f"Expected 1 chunk before cancellation. Actual: {len(chunks_received)}, Content: {chunks_received}"
    else:  # 如果回调都没进入，那就不应该有数据
        assert len(chunks_received) == 0, "Expected 0 chunks as callback was not entered."
    print("DEBUG (cancel_test): test_streaming_cancellation finished assertions.")
    ```

  # .env
  
```
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Proxy Settings (optional)
HTTP_PROXY=
HTTPS_PROXY=

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Application Settings
PORT=8000
HOST=127.0.0.1
  ```

  # .env.example
  
```
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Proxy Settings (optional)
HTTP_PROXY=
HTTPS_PROXY=

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Application Settings
PORT=8000
HOST=127.0.0.1
  ```

  # browser_automator.py
  
```
import asyncio
from typing import Optional
from core.context import ContextManager
from playwright.async_api import async_playwright, Page

class BrowserAutomator:
    def __init__(self, config: dict):
        """初始化浏览器自动化实例
        
        Args:
            config (dict): 配置字典，包含：
                - url: 目标网址
                - input_selector: 输入框CSS选择器
                - submit_selector: 提交按钮CSS选择器
                - response_selector: 响应区域CSS选择器
                - timeout: 超时时间(秒)
        """
        self.config = {
            'url': config.get('url', 'about:blank'),
            'input_selector': config.get('input_selector', 'textarea'),
            'submit_selector': config.get('submit_selector', 'button'),
            'response_selector': config.get('response_selector', '.response'),
            'timeout': config.get('timeout', 30)
        }
        self.context = ContextManager()
        self.page: Optional[Page] = None
        self.browser = None
        self.playwright = None

    async def start(self):
        """启动浏览器并打开目标页面"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch()
        self.page = await self.browser.new_page()
        await self.page.goto(self.config['url'])

    async def send_message(self, message: str, session_id: str = "default", stream: bool = False):
        """发送消息并获取响应，支持流式和非流式模式
        
        Args:
            message (str): 要发送的消息内容
            session_id (str): 会话ID，用于上下文管理
            stream (bool): 是否使用流式响应模式
            
        Returns:
            str 或 AsyncGenerator: 非流式模式返回字符串，流式模式返回异步生成器
        """
        try:
            # 获取上下文并构建完整消息
            context = self.context.get_context(session_id)
            full_message = self._build_message(message, context)
            
            # 执行消息发送
            await self._type_message(full_message)
            await self._click_submit()
            
            if stream:
                # 流式模式返回异步生成器
                return self._stream_response(session_id, message)
            else:
                # 非流式模式获取完整响应
                response = await self._get_response()
                
                # 更新上下文
                self._update_context(session_id, message, response)
                
                return response
        except Exception as e:
            await self._handle_error(e)
            raise
            
    async def _stream_response(self, session_id: str, original_message: str):
        """流式获取响应内容
        
        Args:
            session_id (str): 会话ID
            original_message (str): 原始用户消息
            
        Yields:
            str: 响应内容的增量部分
        """
        try:
            # 初始化变量
            selector = self.config['response_selector']
            last_content = ""
            full_response = ""
            start_time = time.time()
            last_update_time = start_time
            
            # 等待响应元素出现
            await self.page.wait_for_selector(
                selector,
                timeout=self.config['timeout'] * 1000
            )
            
            # 持续监控响应内容变化
            while True:
                # 获取当前内容
                current_content = await self.page.text_content(selector) or ""
                
                # 如果有新内容，生成增量部分
                if current_content != last_content:
                    # 计算增量内容
                    new_content = current_content[len(last_content):]
                    if new_content:
                        last_update_time = time.time()
                        yield new_content
                        full_response += new_content
                        last_content = current_content
                
                # 检查是否应该结束流式响应
                elapsed_since_update = time.time() - last_update_time
                total_elapsed = time.time() - start_time
                
                # 如果内容稳定超过2秒或总时间超过超时时间，结束流式响应
                if (elapsed_since_update > 2.0) or (total_elapsed > self.config['timeout']):
                    break
                    
                # 短暂等待后再次检查
                await asyncio.sleep(0.15)  # 150ms轮询间隔
            
            # 更新上下文
            self._update_context(session_id, original_message, full_response)
            
        except Exception as e:
            await self._handle_error(e)
            raise

    async def _type_message(self, message: str):
        """在输入框中输入消息"""
        await self.page.fill(self.config['input_selector'], message)

    async def _click_submit(self):
        """点击提交按钮"""
        await self.page.click(self.config['submit_selector'])

    async def _get_response(self) -> str:
        """等待并获取响应内容"""
        await self.page.wait_for_selector(
            self.config['response_selector'],
            timeout=self.config['timeout'] * 1000
        )
        return await self.page.text_content(self.config['response_selector'])

    def _build_message(self, message: str, context: list) -> str:
        """构建包含上下文的消息"""
        if not context:
            return message
            
        context_str = "\n".join(
            f"{msg['role']}: {msg['content']}" 
            for msg in context[-3:]  # 只使用最近3条上下文
        )
        return f"对话上下文:\n{context_str}\n\n用户新消息: {message}"

    def _update_context(self, session_id: str, user_msg: str, bot_response: str):
        """更新对话上下文"""
        self.context.add_message(session_id, "user", user_msg)
        self.context.add_message(session_id, "assistant", bot_response)

    async def _handle_error(self, error: Exception):
        """处理错误情况"""
        print(f"操作出错: {str(error)}")
        # 可以在这里添加错误日志记录等

    async def close(self):
        """关闭浏览器资源"""
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
  ```

  # browser_handler.py
  
```
import asyncio
import logging
import psutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from playwright.async_api import async_playwright, Browser, Page, ElementHandle
from tenacity import retry, stop_after_attempt, wait_exponential

from models import LLMSiteConfig, StreamEndCondition, WaitStrategy

logger = logging.getLogger(__name__)

class BrowserInstance:
    """管理单个浏览器实例的类"""
    def __init__(self, browser: Optional[Browser], page: Optional[Page], is_mock: bool = False):
        self.browser = browser
        self.page = page
        self.request_count = 0
        self.last_request_time = 0
        self.creation_time = time.time()
        self.process = psutil.Process() if not is_mock else None
        self.is_available = True
        self.is_mock = is_mock
        
    async def get_memory_usage(self) -> float:
        """获取当前浏览器实例的内存使用量（MB）"""
        try:
            return self.process.memory_info().rss / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0
        
    def should_recycle(self, max_requests: int, max_memory_mb: int) -> bool:
        """检查浏览器实例是否需要回收"""
        if self.request_count >= max_requests:
            return True
            
        try:
            memory_usage = self.process.memory_info().rss / (1024 * 1024)
            if memory_usage > max_memory_mb:
                return True
        except Exception as e:
            logger.warning(f"Failed to check memory usage: {e}")
            
        return False
        
    async def cleanup(self):
        """清理浏览器实例"""
        try:
            if not self.is_mock:
                if self.page:
                    await self.page.close()
                if self.browser:
                    await self.browser.close()
            self.is_available = False
        except Exception as e:
            logger.error(f"Error during browser cleanup: {e}")

class LLMWebsiteAutomator:
    """LLM网站自动化处理类"""
    def __init__(self, config: LLMSiteConfig):
        self.config = config
        self.browser_pool: List[BrowserInstance] = []
        self.lock = asyncio.Lock()
        self.health_check_task: Optional[asyncio.Task] = None
        self.last_health_check = 0
        self.consecutive_failures = 0
        self._initialized = False
        
    async def initialize(self):
        """初始化浏览器实例池"""
        try:
            for _ in range(self.config.pool_size):
                instance = await self._create_browser_instance()
                self.browser_pool.append(instance)
            
            # 启动健康检查任务
            if self.config.health_check.enabled:
                self.health_check_task = asyncio.create_task(self._health_check_loop())
                
        except Exception as e:
            logger.error(f"Failed to initialize browser pool: {e}")
            raise
            
    async def _create_browser_instance(self) -> BrowserInstance:
        """创建新的浏览器实例"""
        if hasattr(self.config, 'mock_responses'):
            # Mock模式返回虚拟实例
            mock_instance = BrowserInstance(None, None, is_mock=True)
            # 添加mock资源监控属性
            mock_instance.mock_memory_usage = 100  # 模拟内存使用
            return mock_instance
            
        try:
            playwright = await async_playwright().start()
            
            # 使用临时用户数据目录
            user_data_dir = tempfile.mkdtemp(prefix="firefox_profile_")
            
            # 获取启动选项
            launch_options = self.config.playwright_launch_options.model_dump(exclude_none=True)
            
            # 启动浏览器上下文
            browser = await playwright.firefox.launch_persistent_context(
                user_data_dir,
                **launch_options
            )
            
            # 创建新页面
            page = browser.pages[0] if browser.pages else await browser.new_page()
            
            # 设置视口大小
            await page.set_viewport_size({'width': 1920, 'height': 1080})
            
            # 导航到目标URL (添加重试逻辑)
            try:
                await page.goto(self.config.url, timeout=15000)
            except Exception as e:
                logger.warning(f"首次导航失败: {e}, 重试...")
                await page.goto(self.config.url, timeout=15000)
            
            # 配置 Stealth 模式
            if self.config.use_stealth:
                await self._setup_stealth_mode(page)
            
            # 导航到目标URL
            await page.goto(self.config.url)
            
            # 同步页面选项
            await self._sync_page_options(page)
            
            return BrowserInstance(browser, page)
            
        except Exception as e:
            logger.error(f"Failed to create browser instance: {e}")
            raise
            
    async def _setup_stealth_mode(self, page: Page):
        """设置浏览器隐身模式"""
        # 实现浏览器指纹隐藏等功能
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false,
            });
        """)
        
    async def _sync_page_options(self, page: Page):
        """同步页面选项配置"""
        for option in self.config.options_to_sync:
            try:
                selector = self.config.selectors[option.selector_key]
                
                if option.action == "select_by_value":
                    await page.select_option(selector, option.target_value_on_page)
                elif option.action == "click_if_not_active":
                    element = await page.query_selector(selector)
                    if element:
                        await element.click()
                        
            except Exception as e:
                if option.on_failure == "abort":
                    raise
                elif option.on_failure == "warn_and_skip":
                    logger.warning(f"Failed to sync option {option.id}: {e}")
                    
    async def _get_available_instance(self) -> BrowserInstance:
        """获取可用的浏览器实例"""
        async with self.lock:
            # 查找可用实例
            for instance in self.browser_pool:
                if not instance.should_recycle(
                    self.config.max_requests_per_instance,
                    self.config.max_memory_per_instance_mb
                ) and instance.is_available:
                    return instance
                    
            # 如果没有可用实例，回收并创建新实例
            old_instance = self.browser_pool.pop(0)
            await old_instance.cleanup()
            
            new_instance = await self._create_browser_instance()
            self.browser_pool.append(new_instance)
            return new_instance
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def send_message(self, message: str) -> Union[str, AsyncGenerator[str, None]]:        
        """发送消息并获取响应"""
        if hasattr(self.config, 'mock_responses'):
            # Mock模式
            instance = await self._get_available_instance()
            instance.request_count += 1
            instance.last_request_time = time.time()
            instance.is_available = False

            try:
                mock_responses = self.config.mock_responses
                default_response = mock_responses.get('default', {})
                
                # 处理mock错误
                if default_response.get('is_error', False):
                    raise Exception(default_response.get('error_message', 'Mock error'))
                    
                # 处理mock超时
                if default_response.get('is_timeout', False):
                    await asyncio.sleep(default_response['timeout_ms'] / 1000)
                    raise asyncio.TimeoutError("Mock timeout")

                if self.config.response_handling.type == "stream":
                    # 创建并返回异步生成器
                    chunks = default_response.get('streaming', ['Mock chunk'])
                    async def mock_stream_wrapper():
                        try:
                            async for chunk in self._mock_stream(chunks):
                                yield chunk
                        finally:
                            instance.is_available = True
                    return mock_stream_wrapper()
                else:
                    await asyncio.sleep(0.2)
                    result = default_response.get('text', 'Mock response')
                    instance.is_available = True
                    return result
            except Exception as e:
                instance.is_available = True
                raise

        # 非mock模式的实现
        instance = await self._get_available_instance()
        instance.request_count += 1
        instance.last_request_time = time.time()
        instance.is_available = False
        
        try:
            # 定位并填充输入框
            input_selector = self.config.selectors["input_area"]
            await instance.page.fill(input_selector, message)
            
            # 点击发送按钮
            submit_selector = self.config.selectors["submit_button"]
            await instance.page.click(submit_selector)
            
            if self.config.response_handling.type == "stream":
                async def real_stream_wrapper():
                    try:
                        async for chunk in self._handle_streaming_response(instance.page):
                            yield chunk
                    finally:
                        instance.is_available = True
                return real_stream_wrapper()
            else:
                result = await self._handle_full_response(instance.page)
                instance.is_available = True
                return result
                
        except Exception as e:
            instance.is_available = True
            logger.error(f"Error sending message: {e}")
            raise

    async def _mock_stream(self, chunks: List[str]) -> AsyncGenerator[str, None]:
        """生成mock流式响应"""
        for chunk in chunks:
            await asyncio.sleep(0.1)
            yield chunk
            
    async def _handle_streaming_response(self, page: Page) -> AsyncGenerator[str, None]:
        """处理流式响应"""
        last_text = ""
        stabilization_start = None
        start_time = time.time()
        bytes_yielded = 0
        retry_count = 0
        
        try:
            while True:
                try:
                    # 检查所有结束条件
                    end_conditions = sorted(
                        self.config.response_handling.stream_end_conditions,
                        key=lambda x: x.priority
                    )
                    
                    for condition in end_conditions:
                        if await self._check_stream_end_condition(page, condition, last_text):
                            logger.info(
                                f"Streaming response completed after {time.time()-start_time:.2f}s, "
                                f"{bytes_yielded} bytes transferred"
                            )
                            return
                            
                    # 获取当前响应文本
                    selector = self.config.selectors[self.config.response_handling.extraction_selector_key]
                    element = await page.query_selector(selector)
                    if not element:
                        continue
                        
                    current_text = await element.get_property(
                        self.config.response_handling.stream_text_property
                    )
                    current_text = await current_text.json_value()
                    
                    # 如果有新内容，yield差异部分
                    if current_text != last_text:
                        new_content = current_text[len(last_text):]
                        bytes_yielded += len(new_content.encode('utf-8'))
                        yield new_content
                        last_text = current_text
                        stabilization_start = None
                        retry_count = 0  # 成功获取内容后重置重试计数
                    elif stabilization_start is None:
                        stabilization_start = time.time()
                        
                    await asyncio.sleep(self.config.response_handling.stream_poll_interval_ms / 1000)
                    
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Streaming error (attempt {retry_count}): {e}")
                    
                    # 检查是否超过最大重试次数
                    if retry_count >= self.config.response_handling.stream_error_handling.max_retries:
                        if self.config.response_handling.stream_error_handling.fallback_to_full_response:
                            logger.warning("Falling back to full response due to streaming errors")
                            full_response = await self._handle_full_response(page)
                            yield full_response[len(last_text):]
                        raise
                        
                    # 等待重试延迟
                    await asyncio.sleep(
                        self.config.response_handling.stream_error_handling.retry_delay_ms / 1000
                    )
                    
        finally:
            # 记录性能指标
            duration = time.time() - start_time
            logger.info(
                f"Streaming session stats - Duration: {duration:.2f}s, "
                f"Bytes: {bytes_yielded}, Throughput: {bytes_yielded/duration:.2f} B/s"
            )
            
            # 确保实例在流式响应结束后被标记为可用
            instance = next((i for i in self.browser_pool if i.page == page), None)
            if instance:
                instance.is_available = True
                
    async def _check_stream_end_condition(
        self,
        page: Page,
        condition: StreamEndCondition,
        current_text: str
    ) -> bool:
        """检查流式响应是否结束"""
        try:
            if condition.type == "element_disappears":
                selector = self.config.selectors[condition.selector_key]
                element = await page.query_selector(selector)
                return element is None
                
            elif condition.type == "text_stabilized":
                if not current_text:
                    return False
                if not hasattr(self, '_text_stable_since'):
                    self._text_stable_since = time.time()
                elif time.time() - self._text_stable_since >= condition.stabilization_time_ms / 1000:
                    return True
                return False
                
            elif condition.type == "timeout":
                if not hasattr(self, '_stream_start_time'):
                    self._stream_start_time = time.time()
                return time.time() - self._stream_start_time >= condition.timeout_seconds
                
        except Exception as e:
            logger.error(f"Error checking stream end condition: {e}")
            return False
            
    async def _handle_full_response(self, page: Page) -> str:
        """处理完整响应"""
        # 等待响应完成
        for strategy in self.config.response_handling.full_text_wait_strategy:
            if strategy.type == "element_disappears":
                selector = self.config.selectors[strategy.selector_key]
                await page.wait_for_selector(selector, state="hidden")
            elif strategy.type == "element_contains_text_or_is_not_empty":
                selector = self.config.selectors[strategy.selector_key]
                await page.wait_for_selector(selector)
                
        # 提取响应文本
        selector = self.config.selectors[self.config.response_handling.extraction_selector_key]
        element = await page.query_selector(selector)
        if not element:
            raise ValueError(f"Response element not found: {selector}")
            
        text = await element.get_property(self.config.response_handling.extraction_method)
        return await text.json_value()
        
    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check.interval_seconds)
                await self._perform_health_check()
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                
    async def _perform_health_check(self):
        """执行健康检查"""
        try:
            instance = await self._get_available_instance()
            selector = self.config.selectors[self.config.health_check.check_element_selector_key]
            
            # 设置超时
            async with asyncio.timeout(self.config.health_check.timeout_seconds):
                element = await instance.page.wait_for_selector(selector)
                if element:
                    self.consecutive_failures = 0
                    self.last_health_check = time.time()
                    return
                    
            self.consecutive_failures += 1
            
            # 如果连续失败次数超过阈值，重新初始化实例
            if self.consecutive_failures >= self.config.health_check.failure_threshold:
                logger.warning("Health check failed, reinitializing browser pool")
                await self.cleanup()
                await self.initialize()
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.consecutive_failures += 1
            
    async def cleanup(self):
        """清理所有浏览器实例"""
        if self.health_check_task:
            self.health_check_task.cancel()
            
        for instance in self.browser_pool:
            await instance.cleanup()
        self.browser_pool.clear()

    # 新增方法
    async def send_prompt_and_get_response(self, prompt: str, stream_callback=None) -> Union[str, AsyncGenerator[str, None]]:
        """发送提示并获取响应，支持流式回调"""
        start_time = time.time()
        
        if getattr(self.config, 'mock_mode', False):
            # Mock模式返回预定义响应
            mock_responses = getattr(self.config, 'mock_responses', {})
            default_response = mock_responses.get('default', {}).get('text', 'Mock response')
            
            # 模拟流式响应延迟
            if mock_responses.get('default', {}).get('streaming') and stream_callback:
                # 如果提供了流式回调，模拟流式响应
                chunks = mock_responses.get('default', {}).get('streaming', ['Mock chunk'])
                for chunk in chunks:
                    await asyncio.sleep(0.1)  # 模拟网络延迟
                    await stream_callback(chunk)
                return ''.join(chunks)
            else:
                await asyncio.sleep(0.5)  # 模拟网络延迟
                return default_response
            
        try:
            # 记录开始时间
            logger.info(f"Starting prompt processing: {prompt[:50]}...")
            
            # 发送消息并获取响应
            response = await self.send_message(prompt)
            
            # 如果提供了流式回调，直接返回异步生成器
            if stream_callback and hasattr(response, '__aiter__'):
                async def stream_wrapper():
                    full_response = ""
                    chunk_count = 0
                    total_bytes = 0
                    
                    try:
                        async for chunk in response:
                            chunk_count += 1
                            total_bytes += len(chunk.encode('utf-8'))
                            full_response += chunk
                            await stream_callback(chunk)
                            yield chunk
                            
                        # 记录流式响应性能指标
                        duration = time.time() - start_time
                        logger.info(
                            f"Completed streaming response in {duration:.2f}s, "
                            f"{chunk_count} chunks, {total_bytes} bytes, "
                            f"avg {total_bytes/duration:.2f} bytes/s"
                        )
                    except Exception as e:
                        logger.error(f"Error in stream processing: {e}")
                        raise
                        
                return stream_wrapper()
            
            # 如果是字符串响应或没有回调，直接返回完整响应
            if isinstance(response, str):
                logger.info(f"Completed non-streaming response in {time.time() - start_time:.2f}s")
                return response
            else:
                # 收集所有内容
                full_response = ""
                async for chunk in response:
                    full_response += chunk
                
                logger.info(f"Collected full streaming response in {time.time() - start_time:.2f}s")
                return full_response
        except Exception as e:
            logger.error(f"Error getting response for prompt: {e}")
            raise

    async def is_healthy(self) -> bool:
        """检查当前automator是否健康"""
        try:
            # 使用健康检查逻辑
            selector = self.config.selectors[self.config.health_check.check_element_selector_key]
            instance = await self._get_available_instance()
            
            async with asyncio.timeout(self.config.health_check.timeout_seconds):
                element = await instance.page.wait_for_selector(selector)
                return element is not None
                
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
  ```

  # browser_handler.py.bak
  
```
import asyncio
import logging
import psutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from playwright.async_api import async_playwright, Browser, Page, ElementHandle
from tenacity import retry, stop_after_attempt, wait_exponential

from models import LLMSiteConfig, StreamEndCondition, WaitStrategy

logger = logging.getLogger(__name__)

class BrowserInstance:
    """管理单个浏览器实例的类"""
    def __init__(self, browser: Browser, page: Page):
        self.browser = browser
        self.page = page
        self.request_count = 0
        self.last_request_time = 0
        self.creation_time = time.time()
        self.process = psutil.Process()
        
    async def get_memory_usage(self) -> float:
        """获取当前浏览器实例的内存使用量（MB）"""
        try:
            return self.process.memory_info().rss / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0
        
    def should_recycle(self, max_requests: int, max_memory_mb: int) -> bool:
        """检查浏览器实例是否需要回收"""
        if self.request_count >= max_requests:
            return True
            
        try:
            memory_usage = self.process.memory_info().rss / (1024 * 1024)
            if memory_usage > max_memory_mb:
                return True
        except Exception as e:
            logger.warning(f"Failed to check memory usage: {e}")
            
        return False
        
    async def cleanup(self):
        """清理浏览器实例"""
        try:
            await self.page.close()
            await self.browser.close()
        except Exception as e:
            logger.error(f"Error during browser cleanup: {e}")

class LLMWebsiteAutomator:
    """LLM网站自动化处理类"""
    def __init__(self, config: LLMSiteConfig):
        self.config = config
        self.browser_pool: List[BrowserInstance] = []
        self.lock = asyncio.Lock()
        self.health_check_task: Optional[asyncio.Task] = None
        self.last_health_check = 0
        self.consecutive_failures = 0
        
    async def initialize(self):
        """初始化浏览器实例池"""
        try:
            for _ in range(self.config.pool_size):
                instance = await self._create_browser_instance()
                self.browser_pool.append(instance)
            
            # 启动健康检查任务
            if self.config.health_check.enabled:
                self.health_check_task = asyncio.create_task(self._health_check_loop())
                
        except Exception as e:
            logger.error(f"Failed to initialize browser pool: {e}")
            raise
            
    async def _create_browser_instance(self) -> BrowserInstance:
        """创建新的浏览器实例"""
        try:
            playwright = await async_playwright().start()
            
            # 使用临时用户数据目录
            user_data_dir = tempfile.mkdtemp(prefix="firefox_profile_")
            
            # 获取启动选项
            launch_options = self.config.playwright_launch_options.model_dump(exclude_none=True)
            
            # 启动浏览器上下文
            browser = await playwright.firefox.launch_persistent_context(
                user_data_dir,
                **launch_options
            )
            
            # 创建新页面
            page = browser.pages[0] if browser.pages else await browser.new_page()
            
            # 设置视口大小
            await page.set_viewport_size({'width': 1920, 'height': 1080})
            
            # 导航到目标URL (添加重试逻辑)
            try:
                await page.goto(self.config.url, timeout=15000)
            except Exception as e:
                logger.warning(f"首次导航失败: {e}, 重试...")
                await page.goto(self.config.url, timeout=15000)
            
            # 配置 Stealth 模式
            if self.config.use_stealth:
                await self._setup_stealth_mode(page)
            
            # 导航到目标URL
            await page.goto(self.config.url)
            
            # 同步页面选项
            await self._sync_page_options(page)
            
            return BrowserInstance(browser, page)
            
        except Exception as e:
            logger.error(f"Failed to create browser instance: {e}")
            raise
            
    async def _setup_stealth_mode(self, page: Page):
        """设置浏览器隐身模式"""
        # 实现浏览器指纹隐藏等功能
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false,
            });
        """)
        
    async def _sync_page_options(self, page: Page):
        """同步页面选项配置"""
        for option in self.config.options_to_sync:
            try:
                selector = self.config.selectors[option.selector_key]
                
                if option.action == "select_by_value":
                    await page.select_option(selector, option.target_value_on_page)
                elif option.action == "click_if_not_active":
                    element = await page.query_selector(selector)
                    if element:
                        await element.click()
                        
            except Exception as e:
                if option.on_failure == "abort":
                    raise
                elif option.on_failure == "warn_and_skip":
                    logger.warning(f"Failed to sync option {option.id}: {e}")
                    
    async def _get_available_instance(self) -> BrowserInstance:
        """获取可用的浏览器实例"""
        async with self.lock:
            # 查找可用实例
            for instance in self.browser_pool:
                if not instance.should_recycle(
                    self.config.max_requests_per_instance,
                    self.config.max_memory_per_instance_mb
                ):
                    return instance
                    
            # 如果没有可用实例，回收并创建新实例
            old_instance = self.browser_pool.pop(0)
            await old_instance.cleanup()
            
            new_instance = await self._create_browser_instance()
            self.browser_pool.append(new_instance)
            return new_instance
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def send_message(self, message: str) -> Union[str, AsyncGenerator[str, None]]:
        """发送消息并获取响应"""
        instance = await self._get_available_instance()
        instance.request_count += 1
        instance.last_request_time = time.time()
        
        try:
            # 定位并填充输入框
            input_selector = self.config.selectors["input_area"]
            await instance.page.fill(input_selector, message)
            
            # 点击发送按钮
            submit_selector = self.config.selectors["submit_button"]
            await instance.page.click(submit_selector)
            
            if self.config.response_handling.type == "stream":
                return self._handle_streaming_response(instance.page)
            else:
                return await self._handle_full_response(instance.page)
                
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise
            
    async def _handle_streaming_response(self, page: Page) -> AsyncGenerator[str, None]:
        """处理流式响应"""
        last_text = ""
        stabilization_start = None
        
        while True:
            try:
                # 检查所有结束条件
                end_conditions = sorted(
                    self.config.response_handling.stream_end_conditions,
                    key=lambda x: x.priority
                )
                
                for condition in end_conditions:
                    if await self._check_stream_end_condition(page, condition, last_text):
                        return
                        
                # 获取当前响应文本
                selector = self.config.selectors[self.config.response_handling.extraction_selector_key]
                element = await page.query_selector(selector)
                if not element:
                    continue
                    
                current_text = await element.get_property(
                    self.config.response_handling.stream_text_property
                )
                current_text = await current_text.json_value()
                
                # 如果有新内容，yield差异部分
                if current_text != last_text:
                    yield current_text[len(last_text):]
                    last_text = current_text
                    stabilization_start = None
                elif stabilization_start is None:
                    stabilization_start = time.time()
                    
                await asyncio.sleep(self.config.response_handling.stream_poll_interval_ms / 1000)
                
            except Exception as e:
                logger.error(f"Error in streaming response: {e}")
                break
                
    async def _check_stream_end_condition(
        self,
        page: Page,
        condition: StreamEndCondition,
        current_text: str
    ) -> bool:
        """检查流式响应是否结束"""
        try:
            if condition.type == "element_disappears":
                selector = self.config.selectors[condition.selector_key]
                element = await page.query_selector(selector)
                return element is None
                
            elif condition.type == "text_stabilized":
                if not current_text:
                    return False
                if not hasattr(self, '_text_stable_since'):
                    self._text_stable_since = time.time()
                elif time.time() - self._text_stable_since >= condition.stabilization_time_ms / 1000:
                    return True
                return False
                
            elif condition.type == "timeout":
                if not hasattr(self, '_stream_start_time'):
                    self._stream_start_time = time.time()
                return time.time() - self._stream_start_time >= condition.timeout_seconds
                
        except Exception as e:
            logger.error(f"Error checking stream end condition: {e}")
            return False
            
    async def _handle_full_response(self, page: Page) -> str:
        """处理完整响应"""
        # 等待响应完成
        for strategy in self.config.response_handling.full_text_wait_strategy:
            if strategy.type == "element_disappears":
                selector = self.config.selectors[strategy.selector_key]
                await page.wait_for_selector(selector, state="hidden")
            elif strategy.type == "element_contains_text_or_is_not_empty":
                selector = self.config.selectors[strategy.selector_key]
                await page.wait_for_selector(selector)
                
        # 提取响应文本
        selector = self.config.selectors[self.config.response_handling.extraction_selector_key]
        element = await page.query_selector(selector)
        if not element:
            raise ValueError(f"Response element not found: {selector}")
            
        text = await element.get_property(self.config.response_handling.extraction_method)
        return await text.json_value()
        
    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check.interval_seconds)
                await self._perform_health_check()
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                
    async def _perform_health_check(self):
        """执行健康检查"""
        try:
            instance = await self._get_available_instance()
            selector = self.config.selectors[self.config.health_check.check_element_selector_key]
            
            # 设置超时
            async with asyncio.timeout(self.config.health_check.timeout_seconds):
                element = await instance.page.wait_for_selector(selector)
                if element:
                    self.consecutive_failures = 0
                    self.last_health_check = time.time()
                    return
                    
            self.consecutive_failures += 1
            
            # 如果连续失败次数超过阈值，重新初始化实例
            if self.consecutive_failures >= self.config.health_check.failure_threshold:
                logger.warning("Health check failed, reinitializing browser pool")
                await self.cleanup()
                await self.initialize()
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.consecutive_failures += 1
            
    async def cleanup(self):
        """清理所有浏览器实例"""
        if self.health_check_task:
            self.health_check_task.cancel()
            
        for instance in self.browser_pool:
            await instance.cleanup()
        self.browser_pool.clear()
  ```

  # config.yaml
  
```
# Global settings for all sites
global_settings:
  timeout: 30  # Default timeout in seconds
  max_retries: 3  # Default number of retries
  retry_delay: 5  # Delay between retries in seconds
  log_level: "INFO"
  request_delay: 1.0  # Delay between requests in seconds
  concurrent_requests: 5  # Maximum number of concurrent requests
  user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# API configurations
api_settings:
  openai:
    model: "gpt-3.5-turbo"
    temperature: 0.7
    max_tokens: 2000
    top_p: 1.0
    frequency_penalty: 0.0
    presence_penalty: 0.0
  anthropic:
    model: "claude-2"
    temperature: 0.7
    max_tokens: 2000

# Proxy settings
proxy_settings:
  enabled: false
  http_proxy: ""
  https_proxy: ""
  bypass_list: []

# LLM sites configuration
llm_sites:
  - id: "local-chatgpt-clone"
    name: "Local ChatGPT Clone Interface"
    enabled: true
    url: "http://localhost:3000/chat"

    # Firefox Profile 配置
    firefox_profile_dir: "local_chatgpt_clone_profile"

    # Playwright 浏览器启动选项
    playwright_launch_options:
      headless: false  # 调试时设为 false，便于观察
      viewport:
        width: 1920
        height: 1080
    use_stealth: true

    # 浏览器实例池大小
    pool_size: 1  # 测试时使用单个实例

    # 实例资源限制
    max_requests_per_instance: 150
    max_memory_per_instance_mb: 1536

    # CSS 选择器
    selectors:
      input_area: "textarea#prompt-textarea"
      submit_button: "button[data-testid='send-message-button']"
      response_container: "div.chat-messages-container"
      last_response_message: "div.chat-messages-container > div.message-bubble[data-role='assistant']:last-child"
      streaming_response_target: "div.chat-messages-container > div.message-bubble[data-role='assistant']:last-child > div.content"
      thinking_indicator: "div.spinner-overlay.active"
      model_selector_dropdown: "select#model-selector"
      theme_toggle_dark: "button#theme-toggle[aria-label='Switch to dark theme']"

    # 备用选择器
    backup_selectors:
      input_area:
        - "textarea[name='user_prompt']"
        - "div.input-wrapper > textarea"
      submit_button:
        - "button.submit-chat"
        - "button[type='submit']"

    # 页面选项同步配置
    options_to_sync:
      - id: "model_selection_on_page"
        description: "Ensure the correct model is selected on the page"
        selector_key: "model_selector_dropdown"
        action: "select_by_value"
        target_value_on_page: "gpt-3.5-turbo"
        on_failure: "warn_and_skip"

      - id: "dark_theme_activation"
        description: "Ensure dark theme is active"
        selector_key: "theme_toggle_dark"
        action: "click_if_not_active"
        on_failure: "warn_and_skip"

    # 响应处理配置
    response_handling:
      type: "stream"
      stream_poll_interval_ms: 150
      stream_text_property: "textContent"
      stream_buffer_size: 1024  # 流式响应缓冲区大小(字节)
      stream_timeout_ms: 30000  # 流式响应超时时间(毫秒)
      stream_error_handling:
        max_retries: 2  # 流式响应错误重试次数
        retry_delay_ms: 500  # 重试间隔(毫秒)
        fallback_to_full_response: true  # 流式失败时是否回退到完整响应
      
      # 流式响应结束条件
      stream_end_conditions:
        - type: "element_disappears"
          selector_key: "thinking_indicator"
          priority: 0
        - type: "text_stabilized"
          stabilization_time_ms: 2500
          priority: 1
        - type: "timeout"
          timeout_seconds: 180
          priority: 2

      # 完整文本等待策略
      full_text_wait_strategy:
        - type: "element_disappears"
          selector_key: "thinking_indicator"
        - type: "element_contains_text_or_is_not_empty"
          selector_key: "last_response_message"
      
      extraction_selector_key: "last_response_message"
      extraction_method: "innerText"

    # 站点健康检查配置
    health_check:
      enabled: true
      interval_seconds: 300
      timeout_seconds: 45
      failure_threshold: 3
      check_element_selector_key: "input_area"
  ```

  # config_manager.py
  
```
import os
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from models import AppConfig, LLMSiteConfig, GlobalSettings
from utils import setup_logging

# Initialize logger
logger = setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))

# Global config instance
_config: Optional[AppConfig] = None

def load_config(config_path: Union[str, Path] = "config.yaml") -> AppConfig:
    """
    Load configuration from YAML file and validate using Pydantic models.
    
    Args:
        config_path (Union[str, Path]): Path to the configuration file
        
    Returns:
        AppConfig: Validated configuration object
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        ValueError: If the configuration is invalid
    """
    global _config
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        error_msg = f"Configuration file not found: {config_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        
        # Validate configuration using Pydantic model
        _config = AppConfig(**config_data)
        logger.info("Configuration loaded and validated successfully")
        
        # Log some basic configuration info
        logger.debug(f"Global timeout: {_config.global_settings.timeout}s")
        logger.debug(f"Configured LLM sites: {[site.id for site in _config.llm_sites]}")
        
        return _config
    
    except yaml.YAMLError as e:
        error_msg = f"Error parsing YAML configuration: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    except Exception as e:
        error_msg = f"Error loading configuration: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def get_config() -> AppConfig:
    """
    Get the current configuration. Loads the default configuration if not already loaded.
    
    Returns:
        AppConfig: The current configuration
    """
    global _config
    
    if _config is None:
        _config = load_config()
    
    return _config

def get_llm_site_config(site_id: str) -> LLMSiteConfig:
    """
    Get configuration for a specific LLM site.
    
    Args:
        site_id (str): ID of the LLM site to get configuration for
        
    Returns:
        LLMSiteConfig: Configuration for the specified LLM site
        
    Raises:
        KeyError: If the site doesn't exist in the configuration
    """
    config = get_config()
    
    for site in config.llm_sites:
        if site.id == site_id:
            if not site.enabled:
                logger.warning(f"LLM site '{site_id}' is disabled in configuration")
            return site
    
    error_msg = f"LLM site '{site_id}' not found in configuration"
    logger.error(error_msg)
    raise KeyError(error_msg)

def get_enabled_llm_sites() -> List[LLMSiteConfig]:
    """
    Get a list of all enabled LLM sites.
    
    Returns:
        List[LLMSiteConfig]: List of enabled LLM site configurations
    """
    config = get_config()
    return [site for site in config.llm_sites if site.enabled]

def get_global_settings() -> GlobalSettings:
    """
    Get global settings from configuration.
    
    Returns:
        GlobalSettings: Global settings object
    """
    config = get_config()
    return config.global_settings

def reload_config(config_path: Union[str, Path] = "config.yaml") -> AppConfig:
    """
    Force reload of configuration from file.
    
    Args:
        config_path (Union[str, Path]): Path to the configuration file
        
    Returns:
        AppConfig: Reloaded configuration object
    """
    global _config
    _config = None
    return load_config(config_path)

# Example usage
if __name__ == "__main__":
    try:
        # Load configuration
        config = get_config()
        print(f"Loaded configuration with {len(config.llm_sites)} LLM sites")
        
        # Get enabled sites
        enabled_sites = get_enabled_llm_sites()
        print(f"Found {len(enabled_sites)} enabled LLM sites")
        
        # Display site information
        for site in enabled_sites:
            print(f"Site '{site.id}': {site.name} ({site.url})")
        
    except Exception as e:
        print(f"Error: {e}")
  ```

  # context_manager.py
  
```
from typing import Dict, List
from datetime import datetime
from pydantic import BaseModel

class DialogContext(BaseModel):
    history: List[Dict[str, str]] = []
    variables: Dict[str, str] = {}
    
class ContextManager:
    def __init__(self, max_history=5):
        self.contexts: Dict[str, DialogContext] = {}
        self.max_history = max_history

    async def update_context(self, session_id: str, role: str, message: str):
        if session_id not in self.contexts:
            self.contexts[session_id] = DialogContext()
        
        self.contexts[session_id].history.append({
            "role": role,
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # 保持历史记录长度
        if len(self.contexts[session_id].history) > self.max_history:
            self.contexts[session_id].history.pop(0)

    async def get_context(self, session_id: str) -> List[Dict]:
        return self.contexts.get(session_id, DialogContext()).history
  ```

  # dev1.md
  
```
**开发文档：LLM 网页自动化 OpenAI API 适配器 (Windows 本地版)**

**版本：** 1.2


目录：

1. 项目概述与目标
2. 核心原则
3. 技术栈 (Windows 环境)
4. 项目文件结构
5. 核心模块设计
    * 5.1 配置管理 (config_manager.py & config.yaml)
    * 5.2 浏览器自动化处理器 (browser_handler.py)
    * 5.3 API 服务 (main.py)
    * 5.4 数据模型 (models.py)
    * 5.5 工具与日志 (utils.py)
6. 分阶段实施计划 (Windows 环境优化)
    * 阶段 0: 环境搭建与基础框架 (Windows 重点)
    * 阶段 1: 单一网站核心自动化 (非流式) 与基础测试
    * 阶段 2: FastAPI 接口封装 (非流式) 与初步并发控制
    * 阶段 3: 实现流式响应与性能考量
    * 阶段 4: 浏览器实例池与资源监控
    * 阶段 5: 鲁棒性、高级功能与可维护性提升
7. 规范与最佳实践
    * 7.1 日志规范与管理
    * 7.2 错误处理与快照
    * 7.3 安全注意事项 (Windows 本地)
    * 7.4 测试策略与节奏
    * 7.5 依赖管理
8. 部署与运行 (Windows 本地)
## 1. 项目概述与目标

本项目旨在通过 Firefox 浏览器自动化技术，在 **Windows 本地环境**下模拟用户访问各种 LLM 网站的操作，并将这些操作封装成符合 OpenAI Chat Completions API 规范 (/v1/chat/completions) 的本地服务。

**主要目标：**

*   **统一接口：** 为不同的 LLM 网站提供统一的、类 OpenAI 的 API 调用方式。
*   **配置驱动：** 允许用户通过配置文件灵活定义目标网站、CSS 选择器、交互逻辑等。
*   **Windows 本地运行：** 所有操作在用户 Windows 本地环境中执行，确保数据隐私和对本地部署模型的支持。
*   **鲁棒性与可维护性：** 构建稳定、易于调试和扩展的系统。
*   **流式响应支持：** 尽可能支持 LLM 的流式输出，提升用户体验。

## 2. 核心原则

*   **配置驱动：** 所有网站特定信息通过 `config.yaml` 文件管理。
*   **模块化：** 清晰分离浏览器自动化、API 接口、配置管理和工具函数。
*   **日志先行：** 详尽的日志记录，便于调试、监控和问题追溯。
*   **状态同步：** 在与网页交互前，检查并（如果可能）自动调整网页上的配置项以匹配 API 请求。
*   **鲁棒设计：** 包含错误处理、重试机制、失败快照等，提升系统稳定性。
*   **迭代测试：** 每个阶段都伴随相应的测试，确保代码质量。

## 3. 技术栈 (Windows 环境)

*   **Python 3.8+** (确保在 Windows 上正确安装并配置 PATH)
*   **Playwright (with Firefox driver)**
    *   **playwright-stealth**
*   **FastAPI**
*   **Uvicorn**
*   **Pydantic**
*   **PyYAML**
*   **python-dotenv**
*   **psutil**: 用于系统资源监控。
*   **Tenacity**: 实现重试逻辑。

## 4. 项目文件结构

```
llm_api_adapter/
├── main.py                 # FastAPI 应用主入口
├── browser_handler.py      # 浏览器自动化核心逻辑 (LLMWebsiteAutomator 类)
├── config_manager.py       # 加载和管理 config.yaml 的逻辑
├── models.py               # Pydantic 模型 (OpenAI API 及内部数据结构)
├── utils.py                # 通用工具函数 (日志配置, 通知等)
├── config.yaml             # LLM 网站详细配置
├── .env                    # 环境变量 (CONFIG_FILE_PATH, LOG_LEVEL等)
├── .env.example            # .env 文件示例
├── requirements.txt        # 项目依赖
├── logs/                   # 日志文件存放目录 (通过 .gitignore 排除)
├── error_snapshots/        # 错误发生时的快照 (截图, DOM)
└── profiles/               # Firefox Profile 存放目录 (Windows 绝对路径或相对于项目根目录的路径)
    └── site_A_profile/
└── tests/                  # 单元测试和集成测试
    ├── conftest.py
    ├── unit/
    └── integration/
```

## 5. 核心模块设计

### 5.1 配置管理 (`config_manager.py` & `config.yaml`)

*   **`config.yaml`**: 核心配置文件，定义全局设置和每个 LLM 站点的详细参数。
    *   **结构示例 (重点部分)：**
        ```yaml
# config.yaml

# 全局设置，适用于所有站点，除非被站点特定配置覆盖
global_settings:
  # Playwright 相关默认超时 (毫秒)
  default_page_load_timeout_ms: 60000   # 页面加载超时
  default_action_timeout_ms: 30000      # 点击、填充等操作的超时
  default_navigation_timeout_ms: 60000 # 导航操作超时
  
  # Firefox Profile 相关
  default_firefox_profile_base_path: "./profiles" # Profile的相对路径基准 (相对于项目根目录)
                                                # 也可使用Windows绝对路径，例如: "C:/Users/YourUser/AppData/Roaming/Mozilla/Firefox/ProfilesAdapter"

  # 日志与快照
  log_level: "INFO"  # 应用日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL) - 可被.env覆盖
  error_snapshot_dir: "./error_snapshots" # 错误快照保存目录 (相对路径或绝对路径)
  
  # 资源管理与健康检查
  max_stale_instance_lifetime_seconds: 300 # 热更新后，旧配置实例的最大存活时间（秒）
  default_max_requests_per_instance: 100   # 每个浏览器实例处理多少请求后重启
  default_max_memory_per_instance_mb: 1024 # 单个Firefox实例允许的最大内存占用 (MB)，用于健康检查
  default_instance_health_check_interval_seconds: 60 # 后台健康检查空闲实例的频率（秒）

# LLM 站点配置列表
llm_sites:
  - id: "my-local-chatgpt-clone" # API 调用时使用的 model 名称 (必须唯一)
    name: "My Local ChatGPT Clone Interface" # 人类可读名称
    enabled: true # 是否启用此站点配置
    url: "http://localhost:3000/chat/custom_model_A" # LLM 网站的完整 URL

    # Firefox Profile 配置
    firefox_profile_dir: "local_chatgpt_clone_profile" # Profile目录名 (相对于 global_settings.default_firefox_profile_base_path)
                                                      # 或 Windows 绝对路径，例如: "D:/MyFirefoxProfiles/chatgpt_clone"

    # Playwright 浏览器启动选项
    playwright_launch_options:
      headless: false # true 为无头模式, false 为有头模式 (调试时建议false)
      # user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0" # 可选，自定义UA
      # viewport: { width: 1920, height: 1080 } # 可选，设置浏览器视口大小
      # slow_mo: 50 # 可选，减慢 Playwright 操作速度 (毫秒)，用于调试
    use_stealth: true # 是否启用 playwright-stealth 规避检测

    # 浏览器实例池大小 (针对此模型)
    pool_size: 2 # 此模型允许同时运行的浏览器实例数量

    # 实例资源限制 (覆盖全局设置)
    max_requests_per_instance: 150
    max_memory_per_instance_mb: 1536

    # CSS 选择器 (必须准确)
    selectors:
      input_area: "textarea#prompt-input-field"
      submit_button: "button[data-testid='send-message-button']"
      response_container: "div.chat-messages-container" # 包含所有消息的父容器
      last_response_message: "div.chat-messages-container > div.message-bubble[data-role='assistant']:last-child" # 精确获取最后一条助手消息
      streaming_response_target: "div.chat-messages-container > div.message-bubble[data-role='assistant']:last-child > div.content" # 流式文本更新的目标元素
      thinking_indicator: "div.spinner-overlay.active" # "思考中"指示器
      # 用于页面选项同步的选择器
      model_selector_dropdown: "select#llm-model-selector"
      theme_toggle_dark: "button#theme-toggle[aria-label='Switch to dark theme']"
      # ... 其他特定于此站点的元素选择器 ...
    
    # 备用选择器 (当主选择器失败时，按顺序尝试)
    backup_selectors:
      input_area: 
        - "textarea[name='user_prompt']"
        - "div.input-wrapper > textarea"
      submit_button:
        - "button.submit-chat"

    # 页面选项同步配置 (在发送 prompt 前执行)
    options_to_sync:
      - id: "model_selection_on_page" # 内部操作标识
        description: "Ensure the 'SuperChat-v3' model is selected on the page."
        selector_key: "model_selector_dropdown" # 引用上面 selectors 中的键名
        action: "select_by_value" # 操作类型: 'select_by_value', 'select_by_label', 'ensure_checked', 'ensure_unchecked', 'click'
        target_value_on_page: "superchat-v3-turbo" # 对于 select_by_value, 这是 option 的 value 属性
                                                    # 对于 ensure_checked/unchecked, 此字段通常不用, 检查元素checked状态
        on_failure: "abort" # 同步失败时的行为: "skip" (跳过), "warn_and_skip" (警告并跳过), "abort" (中断API请求)
      
      - id: "dark_theme_activation"
        description: "Ensure dark theme is active."
        selector_key: "theme_toggle_dark" # 假设这是一个需要点击才能切换到暗色模式的按钮 (如果它当前不是暗色)
        action: "click_if_not_active" # 假设这是一个自定义的或需要特定逻辑判断的action
                                      # 或者更通用的: 'click' (总是点击), 'ensure_attribute_value'
        # target_attribute: "aria-pressed" # 假设用此属性判断是否已激活
        # target_attribute_value: "true"
        # 如果不用自定义action，可能需要多个步骤或更复杂的选择器逻辑
        on_failure: "warn_and_skip"

    # 响应处理配置
    response_handling:
      type: "stream" # 响应类型: "stream" (流式) 或 "full_text" (等待完整文本)
      stream_poll_interval_ms: 150 # 流式响应时 DOM 轮询的间隔时间（毫秒）
      stream_text_property: "textContent" # 从流式目标元素提取文本时使用的属性: "textContent", "innerText", "innerHTML"
      
      # 流式响应结束条件 (满足任一即可，按 priority 从低到高评估，0为最高)
      stream_end_conditions:
        - type: "element_disappears" # 类型: "element_disappears", "element_appears", "text_stabilized", "timeout"
          selector_key: "thinking_indicator" # 引用 selectors 中的键名
          priority: 0 # 最高优先级
        - type: "text_stabilized" # 文本在 stabilization_time_ms 内无变化
          stabilization_time_ms: 2500
          priority: 1
        - type: "element_appears" # 例如“停止生成”按钮出现
          selector_key: "stop_generation_button" # 假设 selectors 中定义了此键
          priority: 0 # 与 thinking_indicator 消失同级
        - type: "timeout" # 流式响应的全局超时
          timeout_seconds: 180 # (秒)
          priority: 2 # 最低优先级，作为最终保障
      
      # 如果 response_handling.type 为 "full_text"，则使用此策略等待响应完成
      full_text_wait_strategy: # 按顺序检查，满足一个即可 (或者可以设计成 all_must_be_true)
        - type: "element_disappears"
          selector_key: "thinking_indicator"
        - type: "element_contains_text_or_is_not_empty" # 确保响应元素不再是初始空状态或加载状态
          selector_key: "last_response_message"
      
      # 从哪个元素提取最终的完整响应文本 (流式和非流式都可能用到)
      extraction_selector_key: "last_response_message"
      # 提取文本时使用的方法
      extraction_method: "innerText" # "innerText", "textContent", "innerHTML"

    # 站点健康检查配置 (用于后台定期检查和 /health API)
    health_check:
      enabled: true # 是否对此站点进行后台健康检查
      interval_seconds: 300 # 后台定期检查的频率（秒）
      timeout_seconds: 45   # 单次健康检查操作的超时（秒）
      failure_threshold: 3  # 连续失败多少次后认为此站点实例不健康
      check_element_selector_key: "input_area" # 健康检查时，验证此元素是否存在且可见

  # --- 可以添加更多站点配置 ---
  - id: "another-ollama-frontend"
    name: "Another Ollama WebUI"
    enabled: false # 此站点当前禁用
    url: "http://localhost:8080"
    firefox_profile_dir: "ollama_frontend_profile"
    playwright_launch_options:
      headless: true
    use_stealth: false
    pool_size: 1
    selectors:
      input_area: "textarea[placeholder*='Send a message']"
      submit_button: "button:has-text('Send')"
      response_container: "main" # 简化示例
      last_response_message: "article:last-of-type div[data-message-author-role='assistant']"
      thinking_indicator: "button[aria-label='Stop generating']" # 等待它出现然后消失，或者它本身就是思考的标志
    # ... 其他配置参考上面的例子 ...
    response_handling:
      type: "full_text"
      stream_poll_interval_ms: 200 # 即使是full_text，也可以保留，只是不用于流式回调
      full_text_wait_strategy:
        - type: "element_attribute_equals"
          selector_key: "thinking_indicator" # 假设这个按钮在生成时有特定状态
          attribute_name: "disabled" # 例如，生成时按钮是 disabled
          attribute_value: "false" # 等待它变为 non-disabled (即 false)
      extraction_selector_key: "last_response_message"
      extraction_method: "textContent"
    health_check:
      enabled: true
      check_element_selector_key: "input_area"
        ```
    *   **`global_settings.default_firefox_profile_base_path`**: 使用相对于项目根目录的路径 (e.g., `"./profiles"`) 或 Windows 绝对路径。
    *   **配置热更新策略**:
        1.  当 `/reload_config` API 端点被调用时，`config_manager` 加载并验证新的 `config.yaml`。
        2.  `main.py` 模块在确认新配置有效后，将当前活动的浏览器实例池标记为“待弃用”状态。
        3.  所有新的 API 请求将从基于新配置动态创建的新实例池中获取浏览器实例。
        4.  待弃用池中的实例在完成其当前正在处理的请求并被归还后，将被立即关闭，并不再加入任何池中。
        5.  设置一个固定的“最大孤立实例运行时长”（例如，在 `global_settings` 中配置 `max_stale_instance_lifetime_seconds: 300`），如果旧配置下的实例在此时间内未完成任务，将被强制关闭以回收资源。

*   **`config_manager.py`**:
    *   `load_config(config_path: str, force_reload: bool = False) -> AppConfig`: 解析 YAML，使用 Pydantic 模型 (`AppConfig` 包含 `GlobalSettings` 和 `List[LLMSiteConfig]`) 进行校验和结构化。支持缓存和强制重载。(实现时需支持热重载通知机制，以便 `main.py` 响应)
    *   `get_config() -> AppConfig`: 获取已加载的配置。
    *   `get_site_config(model_id: str) -> Optional[LLMSiteConfig]`: 根据模型ID获取特定站点配置。

### 5.2 浏览器自动化处理器 (`browser_handler.py`)

*   **`LLMWebsiteAutomator` 类**: 封装单个 LLM 网站的所有 Playwright 交互。
    *   `__init__(self, site_config: LLMSiteConfig, global_settings: GlobalSettings)`: 初始化，存储配置。
    *   `_launch_browser_if_needed(self)`:
        *   惰性启动或连接到 Firefox `PersistentContext` (使用 `user_data_dir`)。
        *   应用 `playwright-stealth`。
        *   设置视口、User-Agent 等 (根据配置)。
        *   导航到目标 URL。
    *   `_get_selector(self, selector_key: str) -> str`: 辅助函数，获取主选择器，如果失败则尝试备用选择器。
    *   `_perform_action(self, action: Callable, selector_key: str, *args, **kwargs)`: 封装 Playwright 操作（点击、填充等），包含超时、重试（使用 Tenacity）、失败快照逻辑。
    *   `_ensure_page_options(self)`: 根据 `site_config.options_to_sync` 检查并同步页面上的选项（模型选择、模式开关等）。处理同步失败的情况 (skip/abort)。
    *   `send_prompt_and_get_response(self, prompt_text: str, stream_callback: Optional[Callable[[str], None]] = None) -> Optional[str]`:
        *   核心方法，处理发送提示到获取响应的整个流程。
        *   调用 `_launch_browser_if_needed`, `_ensure_page_options`。
        *   使用 `_perform_action` 发送提示。
        *   根据 `response_handling.type` (stream/full_text) 调用不同的等待和提取逻辑：
            *   **流式 (`stream_callback` 提供时)**: 监控 `streaming_response_target` 的文本变化（使用 `stream_text_property`），通过 `stream_callback` 回传增量文本。根据 `stream_end_conditions` (带优先级) 判断结束。
            *   **完整文本**: 等待 `full_text_wait_strategy` 条件满足，然后从 `extraction_selector_key` 提取文本。
    *   **流式响应结束条件优先级 (`stream_end_conditions`)**:
        *   在 `send_prompt_and_get_response` 方法的流式处理部分，轮询检查结束条件时，严格按照 `config.yaml` 中为每个条件定义的 `priority` 整数值进行（例如，0 代表最高优先级，数值越大优先级越低）。
        *   在每一轮检查中，从最高优先级的条件开始评估。一旦任何一个条件满足，即判定流已结束，并停止对其他（包括同级和更低优先级）条件的检查。
    *   **健康检查 (`is_healthy`)**:
        *   检查 `self.page.is_closed()` 返回 `False`。
        *   检查 `config.yaml` 中 `health_check.check_element_selector_key` 指定的元素是否存在且可见。
        *   执行一次 `self.page.title()` 调用并验证其是否返回非空字符串，以初步判断页面 JavaScript 环境是否崩溃。
        *   记录完整的健康检查步骤及结果到 DEBUG 日志。
    *   `close(self)`: 安全关闭浏览器和 Playwright 上下文。

### 5.3 API 服务 (`main.py`)

*   **FastAPI 应用**:
    *   **实例管理与并发控制**:
        *   **阶段 2 (初步并发控制)**: 为每个模型ID（`site.id`）维护一个独立的 `LLMWebsiteAutomator` 实例。使用与该实例绑定的 `asyncio.Lock()`来确保任何时刻只有一个 API 请求能够操作此浏览器实例。后续对同一模型的请求将异步等待锁的释放。
        *   **阶段 4 (浏览器实例池)**: 系统将升级为为每个模型ID维护一个 `asyncio.Queue` 作为 `LLMWebsiteAutomator` 实例池。池的大小在 `config.yaml` 中针对每个站点进行配置 (e.g., `pool_size: 3`)。API 请求时从对应模型的池中异步获取实例，使用完毕后异步归还。
    *   `active_automators: Dict[str, Set[LLMWebsiteAutomator]] = {}` (跟踪当前正在使用的实例，用于优雅关闭)。
    *   `@app.on_event("startup")`:
        *   初始化日志、加载配置。
        *   为每个启用的站点创建 `LLMWebsiteAutomator` 实例或实例池。
        *   启动后台健康检查任务。
    *   `@app.on_event("shutdown")`: 优雅关闭所有浏览器实例。
    *   `@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)`:
        *   从池中获取 `LLMWebsiteAutomator` 实例。
        *   根据 `request.stream` 调用 `automator.send_prompt_and_get_response` (传递 `stream_callback` 或不传)。
        *   如果是流式，返回 `StreamingResponse`。
        *   将实例归还到池中。
        *   处理 API 参数映射。
        *   处理临时选择器覆盖。
    *   **流式响应性能**:
        *   在 `config.yaml` 的站点配置中，增加 `response_handling.stream_poll_interval_ms` (e.g., `150`)，用于定义流式响应时 DOM 轮询的间隔时间（毫秒）。
    *   `@app.get("/health")`: 返回整体服务状态，可综合各 Automator 的健康状况。
    *   `@app.post("/reload_config")` (管理员接口，需鉴权): 触发配置热重载。

### 5.4 数据模型 (`models.py`)

*   **OpenAI API 模型**: `ChatMessage`, `ChatCompletionRequest` (带字段校验，如 `temperature` 范围), `ChatCompletionChoice`, `Usage`, `ChatCompletionResponse`。
*   **配置模型**: `LLMSiteConfig`, `GlobalSettings`, `AppConfig` (严格对应 `config.yaml` 结构，用于 Pydantic 校验)。
*   **内部辅助模型**: (如有需要)。

### 5.5 工具与日志 (`utils.py`)

*   `setup_logging(log_level: str, log_dir: str, global_config: GlobalSettings)`: 配置日志格式、级别、文件轮转、控制台输出。
*   `save_error_snapshot(page: Page, site_id: str, snapshot_dir: str)`: 保存截图和 DOM 内容到 `error_snapshots/` 目录。
*   `notify_on_critical_error(message: str, config: NotificationConfig)`: 发送告警 (e.g., Slack, Email)。

## 6. 分阶段实施计划 (Windows 环境优化)

### 阶段 0: 环境搭建与基础框架 (Windows 重点)
*   **任务：**
    1.  环境设置。
    2.  安装核心依赖 (`requirements.txt`)，包括 `psutil`, `tenacity`。
    3.  创建项目结构，`.env.example`, 初始 `config.yaml` (带 `global_settings` 和一个站点模板)。
    4.  `utils.py`: `setup_logging` 实现。
    5.  `models.py`: 定义所有 Pydantic 模型，包括 OpenAI API 模型（带字段校验）和配置模型。
    6.  `config_manager.py`: `load_config`, `get_config`, `get_site_config` 初步实现。
*   **产出：** 可加载配置的基础框架，日志系统可用。

### 阶段 1: 单一网站核心自动化 (非流式) 与基础测试
*   **任务：**
    1.  完善 `config.yaml` 中一个站点的详细配置 (selectors, profile, options_to_sync, response_handling (full_text), health_check)。
    2.  `browser_handler.py` (`LLMWebsiteAutomator`):
        *   `__init__`, `_launch_browser_if_needed` (集成 `playwright-stealth`, 启动选项)。
        *   `_get_selector` (支持备用选择器)。
        *   `_perform_action` (封装 Playwright 调用，带超时、重试、失败快照)。
        *   `_ensure_page_options` 实现。
        *   `send_prompt_and_get_response` (非流式版本，即 `stream_callback=None`)。
        *   `is_healthy` 初步实现，包含基础的页面元素检查和标题检查。
        *   `close` 方法。
    3.  **基础资源监控实施**: 在 `LLMWebsiteAutomator` 的 `send_prompt_and_get_response` 方法执行完毕后（无论成功或失败，通过 `finally` 块确保），使用 `psutil` 获取当前 Python 进程的内存使用情况，并记录到 DEBUG 日志。
    4.  手动准备 Firefox Profile 并配置。
    5.  **同步测试**:
        *   为 `config_manager.py` 的核心功能编写单元测试。
        *   为 `LLMWebsiteAutomator` 的 `_ensure_page_options` 和非流式 `send_prompt_and_get_response` 方法编写集成测试（使用本地 mock HTML 页面）。
*   **产出：** 能够通过代码自动化一个网站的完整交互（登录复用、选项同步、发送提示、获取完整响应）。

### 阶段 2: FastAPI 接口封装 (非流式) 与初步并发控制
*   **任务：**
    1.  `main.py`:
        *   FastAPI 应用，`/v1/chat/completions` 端点 (非流式)。
        *   **并发控制**: 实现基于每个模型单实例 + `asyncio.Lock` 的并发访问控制。
        *   Startup/shutdown 事件处理 (加载配置、创建/关闭 automators)。
        *   `/health` 端点 (调用 `automator.is_healthy`)。
    2.  异步执行 `automator` 的阻塞方法 (`run_in_executor`)。
    3.  **同步测试**: 编写 API 端点（非流式，单模型串行访问）的集成测试。
*   **产出：** 可通过 OpenAI 兼容 API 调用单个网站的非流式聊天功能。

### 阶段 3: 实现流式响应与性能考量
*   **任务：**
    1.  `config.yaml`: 完善站点的 `response_handling` (stream 类型, `stream_text_property`, `stream_end_conditions` 带优先级和超时, `stream_poll_interval_ms`)。
    2.  `browser_handler.py` (`LLMWebsiteAutomator`):
        *   增强 `send_prompt_and_get_response` 以支持 `stream_callback`。
        *   实现流式文本监控、增量提取、结束条件判断逻辑，确保严格按照 `priority` 处理 `stream_end_conditions`。
    3.  `main.py`:
        *   修改 `/v1/chat/completions` 端点，当 `request.stream is True` 时返回 `StreamingResponse`。
    4.  **性能日志**: 记录流式轮询次数、每次轮询耗时、以及从发送提示到流结束的总耗时。
    5.  **同步测试**: 增加流式 API 的集成测试用例，覆盖不同的结束条件。
*   **产出：** API 支持流式响应。

### 阶段 4: 并发处理与浏览器实例管理
*   **任务：**
    1.  `main.py`:
        *   将 `automators` 从单实例字典改为 `automator_pools: Dict[str, asyncio.Queue[LLMWebsiteAutomator]]`。
        *   实现从池中获取/归还 Automator 实例的逻辑。
        *   配置池的大小 (`pool_size` 来自 `config.yaml`)。
    2.  `browser_handler.py`:
        *   `LLMWebsiteAutomator` 实例在处理完 `config.yaml` 中定义的 `max_requests_per_instance` 次请求后，在归还到池时，池将其关闭并异步创建一个新实例补充。
    3.  **资源监控增强与干预**:
        *   `main.py` 中启动一个后台 `asyncio` 任务，定期（例如，每 `default_instance_health_check_interval_seconds`）遍历所有池中的空闲实例，调用其 `is_healthy()` 方法。
        *   如果实例不健康，或通过 `psutil` 检测到其关联的 Firefox 进程（如果能够通过 PID 追踪，这部分较复杂，初期可能依赖主进程或整体资源判断）内存占用超过 `config.yaml` 中定义的 `max_memory_per_instance_mb`，则从池中移除该实例，执行 `close()`，并异步创建新实例补充。
*   **产出：** 提升并发处理能力，更有效地管理浏览器资源。

### 阶段 5: 鲁棒性、高级功能与可维护性提升
*   **任务：**
    1.  **配置热更新**: `main.py` 实现 `/reload_config` API 端点和相应逻辑。
    2.  **后台健康检查与实例重启**: `main.py` 中实现定期检查，并能重启不健康的 Automator 实例（从池中移除，关闭，创建新的补上）。
    3.  **监控与告警**:
        *   集成 Prometheus 监控指标 (请求延迟、实例状态等)。
        *   关键错误时通过 `utils.py` 中的通知函数发送告警。
    4.  **测试覆盖**: 编写更全面的单元测试和集成测试 (`pytest-playwright`)。
    5.  API参数映射/动态选择器注入: 根据需求实现。
*   **产出：** 更稳定、可监控、易于维护的服务。

## 7. 规范与最佳实践

### 7.1 日志规范与管理
*   **格式：** `%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s`
*   **级别：** DEBUG, INFO, WARNING, ERROR, CRITICAL。由 `config.yaml` 或 `.env` 控制。
*   **输出：** 控制台 + 日志文件 (按大小/日期轮转)。
*   **内容：** API请求详情、配置加载、浏览器关键操作、选项同步状态、响应提取、错误堆栈、性能指标（耗时）。
*   **日志清理**: `utils.py` 中的 `setup_logging` 函数配置 `logging.handlers.RotatingFileHandler` 时，必须设置 `backupCount` 参数（例如，`backupCount=10`），以自动限制保留的日志文件数量。

### 7.2 错误处理与快照
*   在 `_perform_action` 和其他关键浏览器交互点进行 `try-except`。
*   捕获 `playwright.sync_api.Error` (如 `TimeoutError`) 等。
*   失败时：
    1.  记录详细错误日志。
    2.  调用 `utils.save_error_snapshot` 保存当前页面截图和 DOM 到 `error_snapshots/`。
    3.  根据策略重试或向上抛出异常。
*   API层面返回标准HTTP错误码和OpenAI格式的错误JSON。

### 7.3 安全注意事项 (Windows 本地)
*   **Firefox Profile (`profiles/`)**: 包含登录凭证，目录权限应设为严格。在文档中强调。
*   **`.env` 文件**: 存储敏感信息（API Key等），不应提交到版本库。
*   **API 访问控制**:
    *   服务默认监听本地回环地址 (`127.0.0.1`)。
    *   如需局域网访问，在 Windows 防火墙中为应用或指定端口配置入站规则。
    *   在 `main.py` 中为所有 `/v1/*` 路径（或至少是修改配置的路径如 `/reload_config`）实现一个基于 HTTP `Authorization` Header 的简单 API Key 认证机制。API Key 在 `.env` 文件中配置 (e.g., `API_KEY="your_secret_key"`)。

### 7.4 测试策略与节奏
*   **单元测试 (`pytest`)**:
    *   `config_manager.py`: 测试配置加载、解析、校验（有效/无效配置）。
    *   `models.py`: Pydantic模型校验逻辑。
    *   `utils.py`: 工具函数。
*   **集成测试 (`pytest-playwright`)**:
    *   `browser_handler.py`: 针对一个或多个mock的本地HTML页面（模拟LLM网站结构）或一个稳定公开的测试网站，测试完整的交互流程（选项同步、提示发送、响应提取 - 包括流式和非流式）。
    *   `main.py`: 测试API端点的请求响应，模拟不同场景（有效请求、无效模型、流式/非流式、认证）。
*   **同步测试**: 在每个功能模块或重要特性开发完成后，立即编写并执行相关的单元测试和集成测试。测试应覆盖正常流程、边界条件和预期的错误场景。

### 7.5 依赖管理
*   使用 `requirements.txt` (或 `pyproject.toml` 若使用 Poetry/PDM)。
*   锁定主要依赖版本，以保证环境一致性。

## 8. 部署与运行 (Windows 本地)

*   **本地运行**: 主要场景。确保用户系统已安装 Firefox。

*   **运行步骤**:
    1.  确保 Python 和 pip 已安装并配置到系统 PATH。
    2.  克隆项目仓库。
    3.  创建并激活虚拟环境 (推荐):
        ```bash
        python -m venv venv
        .\venv\Scripts\activate  # Windows
        ```
    4.  安装依赖: `pip install -r requirements.txt`
    5.  安装 Playwright 浏览器驱动: `playwright install firefox`
    6.  复制 `.env.example` 为 `.env` 并根据需要修改其中的配置，例如 `API_KEY`。
    7.  手动为需要登录的站点准备 Firefox Profile。在 `config.yaml` 的站点配置中，`firefox_profile_dir` 字段指定 Profile 目录的路径（使用 Windows 绝对路径，例如 `"C:/path/to/your/profiles/site_A_profile"`，或相对于项目根目录的路径，例如 `"profiles/site_A_profile"`）。确保该目录存在并且包含有效的 Firefox Profile。
    8.  根据 `config.yaml` 配置各 LLM 站点信息。
    9.  运行服务: `uvicorn main:app --host 127.0.0.1 --port 8000` (或配置文件中指定的 host 和 port)。

---
  ```

  # main.py
  
```
import os
import asyncio
import time
from typing import Dict, Set, Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv
from pydantic import BaseModel

from config_manager import get_config, get_llm_site_config, get_enabled_llm_sites
from browser_handler import LLMWebsiteAutomator
from utils import setup_logging
from models import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStreamResponse

# Load environment variables from .env file
load_dotenv()

# Setup logging
logger = setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file="logs/wrapper_api.log"
)

app = FastAPI(
    title="LLM API Wrapper",
    description="Wrapper API for LLM websites with OpenAI API compatibility",
    version="1.0"
)

# Global state
active_automators: Dict[str, LLMWebsiteAutomator] = {}
automator_locks: Dict[str, asyncio.Lock] = {}

class HealthResponse(BaseModel):
    status: str
    details: Dict[str, str]

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting LLM API Wrapper application")
    
    # Load configuration
    config = get_config()
    logger.info("Configuration loaded successfully")
    
    # Initialize automators for enabled sites
    enabled_sites = get_enabled_llm_sites()
    for site in enabled_sites:
        logger.info(f"Initializing automator for site: {site.id}")
        automator = LLMWebsiteAutomator(site)
        active_automators[site.id] = automator
        automator_locks[site.id] = asyncio.Lock()
    
    logger.info(f"Initialized {len(active_automators)} automators")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down LLM API Wrapper application")
    
    # Close all automators
    for automator in active_automators.values():
        await automator.close()
    
    logger.info("All automators closed successfully")

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI compatible chat completions endpoint"""
    # Get model ID from request
    model_id = request.model
    if model_id not in active_automators:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' not found or disabled"
        )
    
    # Get automator and lock for this model
    automator = active_automators[model_id]
    lock = automator_locks[model_id]
    
    # Process messages to construct prompt
    prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
    
    # Acquire lock to ensure single access to this automator
    async with lock:
        try:
            if request.stream:
                # Streaming response
                async def generate_stream() -> AsyncGenerator[str, None]:
                    start_time = time.time()
                    completion_text = ""
                    
                    try:
                        async for chunk in automator.send_prompt_and_get_response(prompt, stream=True):
                            completion_text += chunk
                            
                            # Format as OpenAI streaming response
                            response_chunk = ChatCompletionStreamResponse(
                                choices=[{
                                    "delta": {
                                        "content": chunk,
                                        "role": "assistant"
                                    },
                                    "finish_reason": None
                                }],
                                model=model_id
                            )
                            yield f"data: {response_chunk.json()}\n\n"
                            
                        # Final completion message
                        final_response = ChatCompletionStreamResponse(
                            choices=[{
                                "delta": {},
                                "finish_reason": "stop"
                            }],
                            model=model_id
                        )
                        yield f"data: {final_response.json()}\n\n"
                        
                    finally:
                        # Log performance metrics
                        duration = time.time() - start_time
                        logger.info(
                            f"Streaming request completed - Model: {model_id}, "
                            f"Duration: {duration:.2f}s, "
                            f"Completion length: {len(completion_text)} chars"
                        )
                
                return StreamingResponse(
                    generate_stream(),
                    media_type="text/event-stream"
                )
            else:
                # Non-streaming response
                start_time = time.time()
                response_text = await automator.send_prompt_and_get_response(prompt)
                
                # Log performance metrics
                duration = time.time() - start_time
                logger.info(
                    f"Request completed - Model: {model_id}, "
                    f"Duration: {duration:.2f}s, "
                    f"Completion length: {len(response_text)} chars"
                )
                
                # Construct OpenAI compatible response
                return ChatCompletionResponse(
                    choices=[{
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": "stop"
                    }],
                    model=model_id,
                    usage={
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(response_text.split()),
                        "total_tokens": len(prompt.split()) + len(response_text.split())
                    }
                )
            
        except Exception as e:
            logger.error(f"Error processing request for model {model_id}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing request: {str(e)}"
            )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    status = "healthy"
    details = {}
    
    # Check health of each automator
    for model_id, automator in active_automators.items():
        try:
            is_healthy = await automator.is_healthy()
            details[model_id] = "healthy" if is_healthy else "unhealthy"
            if not is_healthy:
                status = "degraded"
        except Exception as e:
            details[model_id] = f"error: {str(e)}"
            status = "unhealthy"
    
    return HealthResponse(status=status, details=details)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
  ```

  # models.py
  
```
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator, model_validator, HttpUrl

# Enums
class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AuthType(str, Enum):
    NONE = "none"
    BASIC = "basic"
    BEARER = "bearer"

class HttpMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"

# OpenAI API Models
class OpenAIMessage(BaseModel):
    role: str = Field(..., description="The role of the message author (system, user, assistant)")
    content: str = Field(..., description="The content of the message")
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed_roles = ['system', 'user', 'assistant', 'function']
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}")
        return v

class OpenAIFunctionParameter(BaseModel):
    type: str
    description: Optional[str] = None
    enum: Optional[List[str]] = None
    
class OpenAIFunctionParameters(BaseModel):
    type: str = "object"
    properties: Dict[str, OpenAIFunctionParameter]
    required: Optional[List[str]] = None

class OpenAIFunction(BaseModel):
    name: str
    description: str
    parameters: OpenAIFunctionParameters

class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    functions: Optional[List[OpenAIFunction]] = None
    function_call: Optional[Union[str, Dict[str, str]]] = None
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 0 or v > 2):
            raise ValueError("Temperature must be between 0 and 2")
        return v
    
    @field_validator('top_p')
    @classmethod
    def validate_top_p(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Top_p must be between 0 and 1")
        return v
    
    @field_validator('presence_penalty', 'frequency_penalty')
    @classmethod
    def validate_penalty(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < -2 or v > 2):
            raise ValueError("Penalty values must be between -2 and 2")
        return v

class OpenAIFunctionCall(BaseModel):
    name: str
    arguments: str

class OpenAIChatChoice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: str
    function_call: Optional[OpenAIFunctionCall] = None

class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class OpenAIChatStreamDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class OpenAIChatStreamChoice(BaseModel):
    index: int
    delta: OpenAIChatStreamDelta
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[OpenAIChatStreamChoice]

class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChatChoice]
    usage: OpenAIUsage

# Configuration Models
class ProxySettings(BaseModel):
    enabled: bool = False
    http_proxy: Optional[str] = None
    https_proxy: Optional[str] = None
    bypass_list: List[str] = Field(default_factory=list)

class APISettings(BaseModel):
    model: str
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

class APIConfigurations(BaseModel):
    openai: APISettings
    anthropic: Optional[APISettings] = None

class EndpointConfig(BaseModel):
    path: str
    method: HttpMethod
    rate_limit: Optional[int] = None
    timeout: Optional[int] = None

class AuthConfig(BaseModel):
    type: AuthType = AuthType.NONE
    token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    
    @model_validator(mode='after')
    def check_auth_credentials(self):
        auth_type = self.type
        if auth_type == AuthType.BASIC:
            if not self.username or not self.password:
                raise ValueError("Username and password are required for basic auth")
        elif auth_type == AuthType.BEARER:
            if not self.token:
                raise ValueError("Token is required for bearer auth")
        return self

class RetryConfig(BaseModel):
    max_retries: int = 3
    retry_delay: int = 5

class SiteConfig(BaseModel):
    enabled: bool = True
    base_url: str
    rate_limit: Optional[int] = None
    timeout: Optional[int] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    endpoints: List[EndpointConfig] = Field(default_factory=list)
    
    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        # Simple URL validation
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Base URL must start with http:// or https://")
        return v

class GlobalSettings(BaseModel):
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 5
    log_level: LogLevel = LogLevel.INFO
    request_delay: float = 1.0
    concurrent_requests: int = 5
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# LLM Site Configuration Models
class ViewportConfig(BaseModel):
    width: int = 1920
    height: int = 1080

class PlaywrightLaunchOptions(BaseModel):
    headless: bool = True
    viewport: Optional[ViewportConfig] = None
    user_agent: Optional[str] = None
    slow_mo: Optional[int] = None

class StreamEndCondition(BaseModel):
    type: str
    selector_key: Optional[str] = None
    stabilization_time_ms: Optional[int] = None
    timeout_seconds: Optional[int] = None
    priority: int = 999  # 默认最低优先级

class WaitStrategy(BaseModel):
    type: str
    selector_key: Optional[str] = None

class ResponseHandling(BaseModel):
    type: str  # "stream" or "full_text"
    stream_poll_interval_ms: int = 150
    stream_text_property: str = "textContent"
    stream_end_conditions: List[StreamEndCondition] = Field(default_factory=list)
    full_text_wait_strategy: List[WaitStrategy] = Field(default_factory=list)
    extraction_selector_key: str
    extraction_method: str = "innerText"

class OptionSync(BaseModel):
    id: str
    description: str
    selector_key: str
    action: str
    target_value_on_page: Optional[str] = None
    on_failure: str = "abort"  # "abort", "skip", "warn_and_skip"

class HealthCheck(BaseModel):
    enabled: bool = True
    interval_seconds: int = 300
    timeout_seconds: int = 45
    failure_threshold: int = 3
    check_element_selector_key: str

class LLMSiteConfig(BaseModel):
    id: str
    name: str
    enabled: bool = True
    url: str
    firefox_profile_dir: str
    playwright_launch_options: PlaywrightLaunchOptions = Field(default_factory=PlaywrightLaunchOptions)
    use_stealth: bool = True
    pool_size: int = 1
    max_requests_per_instance: int = 100
    max_memory_per_instance_mb: int = 1024
    selectors: Dict[str, str]
    backup_selectors: Optional[Dict[str, List[str]]] = None
    options_to_sync: List[OptionSync] = Field(default_factory=list)
    response_handling: ResponseHandling
    health_check: HealthCheck
    mock_responses: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Mock responses for testing purposes"
    )

class AppConfig(BaseModel):
    global_settings: GlobalSettings
    api_settings: APIConfigurations
    proxy_settings: ProxySettings = Field(default_factory=ProxySettings)
    llm_sites: List[LLMSiteConfig] = Field(default_factory=list)

# API 别名
ChatCompletionRequest = OpenAIChatCompletionRequest
ChatCompletionResponse = OpenAIChatCompletionResponse
  ```

  # pytest.ini
  
```
[pytest]
asyncio_default_fixture_loop_scope = function
  ```

  # requirements.txt
  
```
playwright>=1.42.0
pytest-asyncio>=0.23.0
tenacity>=8.2.3
pytest>=8.0.0
  ```

  # utils.py
  
```
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (Optional[str]): Path to log file. If None, logs only to console
        max_bytes (int): Maximum size of log file before rotation
        backup_count (int): Number of backup files to keep
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get logger
    logger = logging.getLogger("wrapper_api")
    
    # Convert string log level to logging constant
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is specified)
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Log initial message
    logger.info(f"Logging setup completed. Level: {logging.getLevelName(log_level)}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    return logger

def get_env_log_level() -> str:
    """
    Get log level from environment variable or return default.
    
    Returns:
        str: Log level string
    """
    return os.getenv("LOG_LEVEL", "INFO").upper()

# Example usage
if __name__ == "__main__":
    # Setup logging with both console and file output
    logger = setup_logging(
        log_level=get_env_log_level(),
        log_file="logs/wrapper_api.log"
    )
    
    # Test logging
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
  ```

