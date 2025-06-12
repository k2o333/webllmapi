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

    # cleanup.py
    
```
import os
import shutil
import psutil
import asyncio
from pathlib import Path

async def cleanup_test_environment():
    """清理测试环境，删除所有测试相关的临时文件和进程"""
    print("开始清理测试环境...")
    
    # 获取项目根目录
    root_dir = Path(__file__).parent.parent
    
    # 清理 __pycache__ 目录
    def clean_pycache(directory):
        for root, dirs, files in os.walk(directory):
            # 删除 __pycache__ 目录
            if '__pycache__' in dirs:
                pycache_path = os.path.join(root, '__pycache__')
                print(f"删除 {pycache_path}")
                shutil.rmtree(pycache_path, ignore_errors=True)
            # 删除 .pyc 文件
            for file in files:
                if file.endswith('.pyc'):
                    file_path = os.path.join(root, file)
                    print(f"删除 {file_path}")
                    os.remove(file_path)
    
    # 清理测试缓存目录
    cache_dirs = [
        '.pytest_cache',
        '.mypy_cache',
        '.coverage'
    ]
    
    for cache_dir in cache_dirs:
        cache_path = root_dir / cache_dir
        if cache_path.exists():
            print(f"删除缓存目录: {cache_path}")
            shutil.rmtree(cache_path, ignore_errors=True)
    
    # 清理 Python 缓存文件
    clean_pycache(root_dir)
    
    # 清理日志文件
    logs_dir = root_dir / 'logs'
    if logs_dir.exists():
        for log_file in logs_dir.glob('*.log'):
            print(f"删除日志文件: {log_file}")
            log_file.unlink()
    
    # 终止可能残留的浏览器进程（根据需求只关闭Firefox）
    browser_processes = ['firefox', 'playwright']
    for proc in psutil.process_iter(['name']):
        try:
            for browser in browser_processes:
                if browser in proc.info['name'].lower():
                    print(f"终止进程: {proc.info['name']} (PID: {proc.pid})")
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    print("测试环境清理完成")

def run_cleanup():
    """运行清理程序"""
    asyncio.run(cleanup_test_environment())

if __name__ == "__main__":
    run_cleanup()
    ```

    # run_stage4_tests.py
    
```
import os
import sys
import time
import pytest
import asyncio
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入清理模块
from cleanup import run_cleanup

def run_tests():
    """运行阶段4测试并记录结果"""
    # 记录开始时间
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("="*80)
    print(f"开始阶段4测试 - {start_datetime}")
    print("="*80)
    
    # 首先清理环境
    print("\n[1/3] 清理测试环境...")
    run_cleanup()
    
    # 运行测试
    print("\n[2/3] 运行阶段4测试...")
    test_file = Path(__file__).parent / "test_stage4.py"
    result = pytest.main(["-v", str(test_file)])
    
    # 计算测试时间
    end_time = time.time()
    duration = end_time - start_time
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 准备测试结果摘要
    print("\n[3/3] 生成测试报告...")
    
    # 获取测试结果文件路径
    results_file = Path(__file__).parent / "test_results.txt"
    
    # 添加本次测试的结果到测试结果文件
    with open(results_file, "a") as f:
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write(f"测试执行摘要 - {end_datetime}\n")
        f.write("="*80 + "\n")
        f.write(f"开始时间: {start_datetime}\n")
        f.write(f"结束时间: {end_datetime}\n")
        f.write(f"测试持续时间: {duration:.2f} 秒\n")
        f.write(f"测试结果代码: {result}\n")
        f.write(f"测试状态: {'成功' if result == 0 else '失败'}\n")
        f.write("="*80 + "\n")
    
    # 打印测试摘要
    print("\n" + "="*80)
    print(f"测试执行摘要")
    print("="*80)
    print(f"开始时间: {start_datetime}")
    print(f"结束时间: {end_datetime}")
    print(f"测试持续时间: {duration:.2f} 秒")
    print(f"测试状态: {'成功' if result == 0 else '失败'}")
    print("="*80)
    print(f"\n测试结果已添加到: {results_file}")
    
    return result

if __name__ == "__main__":
    sys.exit(run_tests())
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

    # test_stage4.py
    
```
import pytest
import asyncio
import psutil
from browser_handler import BrowserInstance, LLMWebsiteAutomator
from models import LLMSiteConfig, StreamEndCondition, WaitStrategy, ResponseHandling, HealthCheck

@pytest.fixture
def mock_config():
    """创建用于测试的模拟配置"""
    return LLMSiteConfig(
        id="test_site",
        name="Test Site",
        url="https://test.com",
        firefox_profile_dir="./test_profile",
        mock_mode=True,
        max_requests_per_instance=5,
        max_memory_per_instance_mb=500,
        selectors={
            "input_area": "#input",
            "submit_button": "#submit",
            "response_area": "#response",
            "health_check_element": "#health"
        },
        response_handling=ResponseHandling(
            type="streaming",
            extraction_selector_key="response_area",
            stream_end_conditions=[
                StreamEndCondition(
                    type="element_present",
                    selector_key="response_area",
                    timeout_seconds=30
                )
            ],
            full_text_wait_strategy=[
                WaitStrategy(
                    type="element_visible",
                    selector_key="response_area"
                )
            ]
        ),
        health_check=HealthCheck(
            enabled=True,
            check_element_selector_key="health_check_element"
        )
    )

@pytest.mark.asyncio
async def test_browser_instance_memory_monitoring():
    """测试浏览器实例的内存监控功能"""
    # 创建一个模拟的浏览器实例
    browser_instance = BrowserInstance(None, None, psutil.Process().pid, is_mock=False)
    
    # 测试内存使用量获取
    memory_usage = await browser_instance.get_memory_usage()
    assert isinstance(memory_usage, float)
    assert memory_usage > 0

    # 清理
    await browser_instance.cleanup()

@pytest.mark.asyncio
async def test_browser_instance_request_counting():
    """测试浏览器实例的请求计数功能"""
    # 创建一个模拟的浏览器实例
    browser_instance = BrowserInstance(None, None, None, is_mock=True)
    
    # 初始请求计数应为0
    assert browser_instance.request_count == 0
    
    # 模拟增加请求计数
    browser_instance.request_count += 1
    assert browser_instance.request_count == 1

    # 清理
    await browser_instance.cleanup()

@pytest.mark.asyncio
async def test_browser_instance_recycling(mock_config):
    """测试浏览器实例的回收机制"""
    automator = LLMWebsiteAutomator(mock_config)
    await automator.initialize()
    
    # 确保初始化完成
    assert await automator.is_healthy()
    
    # 发送多个请求以触发回收条件
    for i in range(mock_config.max_requests_per_instance):
        response = await automator.send_message("test message")
        assert response is not None
    
    # 验证实例是否应该被回收
    assert automator.should_recycle_based_on_metrics() == True
    
    # 清理
    await automator.cleanup()

@pytest.mark.asyncio
async def test_memory_usage_monitoring(mock_config):
    """测试内存使用监控功能"""
    automator = LLMWebsiteAutomator(mock_config)
    await automator.initialize()
    
    # 确保初始化完成
    assert await automator.is_healthy()
    
    # 确保mock实例的内存使用值是浮点数
    if automator.managed_browser_instance and automator.managed_browser_instance.is_mock:
        automator.managed_browser_instance.mock_memory_usage = 50.0
    
    # 测试内存使用量获取
    memory_usage = await automator.get_memory_usage()
    assert isinstance(memory_usage, float)
    
    # 清理
    await automator.cleanup()

@pytest.mark.asyncio
async def test_request_count_tracking(mock_config):
    """测试请求数量跟踪功能"""
    automator = LLMWebsiteAutomator(mock_config)
    await automator.initialize()
    
    # 确保初始化完成
    assert await automator.is_healthy()
    
    # 初始请求计数应为0
    assert automator.get_request_count() == 0
    
    # 发送一条消息并检查请求计数
    response = await automator.send_message("test message")
    assert response is not None
    assert automator.get_request_count() == 1
    
    # 清理
    await automator.cleanup()

@pytest.mark.asyncio
async def test_instance_health_check(mock_config):
    """测试实例健康检查功能"""
    automator = LLMWebsiteAutomator(mock_config)
    await automator.initialize()
    
    # 测试健康检查
    is_healthy = await automator.is_healthy()
    assert is_healthy == True
    
    # 清理
    await automator.cleanup()

@pytest.mark.asyncio
async def test_cleanup_and_resource_release(mock_config):
    """测试清理和资源释放功能"""
    automator = LLMWebsiteAutomator(mock_config)
    await automator.initialize()
    
    # 执行清理
    await automator.cleanup()
    
    # 验证清理后的状态
    assert automator.managed_browser_instance is None
    assert automator._playwright_instance is None

if __name__ == "__main__":
    pytest.main(["-v", "test_stage4.py"])
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
API_KEY="sk-1234" # Key for accessing sensitive endpoints like /reload_config

# Proxy Settings (optional)
HTTP_PROXY=
HTTPS_PROXY=

# Logging
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Application Settings
PORT=8000
HOST=0.0.0.0
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

  # .gitignore
  
```
# Python项目通用.gitignore

# 虚拟环境
venv/
env/
__pycache__/
*.pyc
*.pyo
*.pyd

# 日志文件
logs/
*.log

# 浏览器缓存
local_chatgpt_clone_profile/

# 测试相关
pytest.ini
__pycache__
test_*.py
*.pytest_cache*

# IDE配置
.vscode/
.idea/

# 系统文件
.DS_Store
Thumbs.db

# 配置文件（如果存在本地敏感配置）
config_local.yaml

# 临时文件
*.tmp
*.swp

# 打包文件
*.zip
*.tar.gz

# 浏览器缓存目录
browser_cache/

.convert/

.env
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
# browser_handler.py

import asyncio
import logging
import psutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from playwright.async_api import async_playwright, Browser, Page, ElementHandle, BrowserContext
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

from models import LLMSiteConfig, StreamEndCondition, WaitStrategy, ModelVariantConfig, ModelSelectionStep

logger = logging.getLogger("wrapper_api.browser_handler") # 【次要修改】统一使用分层 logger

class BrowserInstance:
    """管理单个浏览器实例的类"""
    def __init__(self, browser_context: Optional[BrowserContext], page: Optional[Page], browser_pid: Optional[int] = None, is_mock: bool = False):
        self.browser_context = browser_context
        self.page = page
        self.browser_pid = browser_pid
        self.request_count = 0
        self.last_request_time = 0
        self.creation_time = time.time()
        self.process = psutil.Process(browser_pid) if browser_pid and psutil.pid_exists(browser_pid) else None
        self.is_available = True
        self.is_mock = is_mock
        self.firefox_profile_dir_path: Optional[Path] = None

    async def get_memory_usage(self) -> float:
        if self.is_mock:
            return getattr(self, 'mock_memory_usage', 50.0)
        if self.process:
            try:
                return self.process.memory_info().rss / (1024 * 1024)
            except psutil.NoSuchProcess:
                logger.warning(f"Browser process with PID {self.browser_pid} not found for memory usage check.")
                self.process = None
                return 0.0
            except Exception as e:
                logger.warning(f"Failed to get memory usage for PID {self.browser_pid}: {e}")
                return 0.0
        return 0.0

    async def should_recycle(self, max_requests: int, max_memory_mb: int) -> bool:
        if self.is_mock:
            return self.request_count >= max_requests if max_requests > 0 else False

        if max_requests > 0 and self.request_count >= max_requests:
            logger.info(f"Instance (PID: {self.browser_pid}) should recycle: request count {self.request_count} >= {max_requests}")
            return True

        memory_usage = await self.get_memory_usage()
        if max_memory_mb > 0 and memory_usage > 0:
            if memory_usage > max_memory_mb:
                logger.info(f"Instance (PID: {self.browser_pid}) should recycle: memory usage {memory_usage:.2f}MB > {max_memory_mb}MB")
                return True
        elif max_memory_mb > 0 and self.process is None:
            logger.warning(f"Instance (PID: {self.browser_pid}) cannot check memory for recycle, process not available. Skipping memory check.")

        return False

    async def cleanup(self):
        logger.info(f"Cleaning up BrowserInstance (PID: {self.browser_pid}, Mock: {self.is_mock})")
        try:
            if not self.is_mock:
                if self.page and not self.page.is_closed():
                    await self.page.close()
                if self.browser_context:
                    await self.browser_context.close()
            self.is_available = False
        except Exception as e:
            logger.error(f"Error during BrowserInstance cleanup (PID: {self.browser_pid}): {e}")


class LLMWebsiteAutomator:
    def __init__(self, config: LLMSiteConfig):
        self.config = config
        self.managed_browser_instance: Optional[BrowserInstance] = None
        self._playwright_instance = None
        self._initialized_event = asyncio.Event()

    async def initialize(self):
        if self.managed_browser_instance:
            logger.warning(f"Automator for {self.config.id} already initialized.")
            await self.cleanup()

        logger.info(f"Initializing LLMWebsiteAutomator for site: {self.config.id} (Mock: {self.config.mock_mode})")
        try:
            self.managed_browser_instance = await self._create_single_browser_instance()
            self._initialized_event.set()
            logger.info(f"LLMWebsiteAutomator for {self.config.id} initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LLMWebsiteAutomator for {self.config.id}: {e}", exc_info=True)
            self._initialized_event.clear()
            raise

    async def _create_single_browser_instance(self) -> BrowserInstance:
        if self.config.mock_mode:
            logger.info(f"Creating MOCK browser instance for {self.config.id}")
            mock_bi = BrowserInstance(browser_context=None, page=None, browser_pid=None, is_mock=True)
            mock_bi.mock_memory_usage = 50
            return mock_bi

        logger.info(f"Creating REAL browser instance for {self.config.id}")
        playwright_obj = None
        browser_context_obj = None

        try:
            logger.info(f"[{self.config.id}] Starting Playwright...")
            playwright_obj = await async_playwright().start()
            self._playwright_instance = playwright_obj
            logger.info(f"[{self.config.id}] Playwright started successfully.")

            profile_path_str = self.config.firefox_profile_dir
            profile_path = Path(profile_path_str)
            if not profile_path.is_absolute():
                profile_path = Path.cwd() / profile_path_str
            
            if not profile_path.exists():
                logger.warning(f"[{self.config.id}] Profile directory not found at {profile_path}. Creating it.")
                profile_path.mkdir(parents=True, exist_ok=True)
            else:
                logger.info(f"[{self.config.id}] Using existing profile directory at {profile_path}")

            user_data_dir = str(profile_path)
            
            launch_options = self.config.playwright_launch_options.model_dump(exclude_none=True)
            if 'viewport' in launch_options and launch_options['viewport'] is None:
                del launch_options['viewport']

            logger.info(f"[{self.config.id}] Launching Firefox persistent context with options: {launch_options} and user_data_dir: {user_data_dir}")
            browser_context_obj = await playwright_obj.firefox.launch_persistent_context(
                user_data_dir,
                **launch_options
            )
            logger.info(f"[{self.config.id}] Firefox persistent context launched.")
            
            browser_pid = None
            page = browser_context_obj.pages[0] if browser_context_obj.pages else await browser_context_obj.new_page()
            
            if self.config.playwright_launch_options.viewport:
                 await page.set_viewport_size(self.config.playwright_launch_options.viewport.model_dump())

            if self.config.use_stealth:
                logger.info(f"[{self.config.id}] Setting up stealth mode.")
                await self._setup_stealth_mode(page)

            logger.info(f"Navigating to URL: {self.config.url} for {self.config.id}")
            try:
                await page.goto(self.config.url, timeout=60000, wait_until="domcontentloaded")
            except Exception as e:
                logger.warning(f"Initial navigation to {self.config.url} failed: {e}. Retrying...")
                await asyncio.sleep(2)
                await page.goto(self.config.url, timeout=60000, wait_until="domcontentloaded")
            logger.info(f"[{self.config.id}] Successfully navigated to {self.config.url}")

            await self._sync_page_options(page)
            
            bi = BrowserInstance(browser_context_obj, page, browser_pid, is_mock=False)
            bi.firefox_profile_dir_path = profile_path
            return bi
        except Exception as e:
            logger.error(f"Failed to create REAL browser instance for {self.config.id}: {e}", exc_info=True)
            if browser_context_obj: await browser_context_obj.close()
            if playwright_obj and self._playwright_instance:
                 await self._playwright_instance.stop()
                 self._playwright_instance = None
            raise

    async def _setup_stealth_mode(self, page: Page):
        await page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

    async def _sync_page_options(self, page: Page):
        if not self.config.options_to_sync:
            return
        logger.info(f"Syncing general page options for {self.config.id}...")
        for option_sync_config in self.config.options_to_sync:
            selector = self.config.selectors.get(option_sync_config.selector_key)
            if not selector:
                logger.warning(f"Selector key '{option_sync_config.selector_key}' not found for option sync '{option_sync_config.id}'. Skipping.")
                continue
            try:
                if option_sync_config.action == "select_by_value":
                    await page.select_option(selector, value=option_sync_config.target_value_on_page, timeout=10000)
                elif option_sync_config.action == "click_if_not_active":
                    element = await page.query_selector(selector)
                    if element: await element.click(timeout=10000)
                logger.info(f"Successfully synced general option: {option_sync_config.id}")
            except Exception as e:
                logger.error(f"Failed to sync general option '{option_sync_config.id}' with selector '{selector}': {e}")
                if option_sync_config.on_failure == "abort": raise
                elif option_sync_config.on_failure == "warn_and_skip": logger.warning(f"Skipping option sync for '{option_sync_config.id}'.")

    async def _execute_model_selection_flow(self, target_model_variant_id: str, page: Page):
        if not self.config.model_variants:
            logger.debug(f"[{self.config.id}] No model_variants configured. Skipping model selection.")
            return

        selected_variant: Optional[ModelVariantConfig] = None
        for variant in self.config.model_variants:
            if variant.id == target_model_variant_id:
                selected_variant = variant
                break
        
        if not selected_variant:
            logger.warning(f"[{self.config.id}] Model variant '{target_model_variant_id}' not found in configuration. Skipping specific model selection.")
            return

        logger.info(f"[{self.config.id}] Executing model selection flow for variant '{target_model_variant_id}' ({selected_variant.name_on_page or ''}).")
        
        for step_idx, step in enumerate(selected_variant.selection_flow):
            logger.debug(f"[{self.config.id}] Model selection step {step_idx + 1}/{len(selected_variant.selection_flow)}: {step.action} on '{step.selector_key}' (Desc: {step.description or 'N/A'})")
            selector_str = self.config.selectors.get(step.selector_key)
            if not selector_str:
                logger.error(f"[{self.config.id}] Selector_key '{step.selector_key}' not found in main selectors for model selection step. Aborting flow.")
                raise RuntimeError(f"Missing selector_key '{step.selector_key}' for model selection on {self.config.id}")

            try:
                element_to_interact = await page.query_selector(selector_str)
                if not element_to_interact:
                    logger.debug(f"[{self.config.id}] Element '{step.selector_key}' not immediately found, waiting...")
                    try:
                        await page.wait_for_selector(selector_str, state="visible", timeout=15000) 
                        element_to_interact = await page.query_selector(selector_str)
                    except Exception as wait_e:
                         logger.warning(f"[{self.config.id}] Timeout waiting for element '{step.selector_key}': {wait_e}")
                         element_to_interact = None

                if not element_to_interact:
                    logger.error(f"[{self.config.id}] Element for selector '{step.selector_key}' (Selector: {selector_str}) not found/visible for model selection step.")
                    raise RuntimeError(f"Element not found for model selection step '{step.selector_key}' on {self.config.id}")

                if step.action == "click":
                    await element_to_interact.click(timeout=10000)
                    logger.debug(f"[{self.config.id}] Clicked element for '{step.selector_key}'.")
                else:
                    logger.warning(f"[{self.config.id}] Unsupported action '{step.action}' in model selection flow. Skipping step.")

                if step.wait_after_ms:
                    logger.debug(f"[{self.config.id}] Waiting for {step.wait_after_ms}ms after action on '{step.selector_key}'.")
                    await asyncio.sleep(step.wait_after_ms / 1000.0)
            
            except Exception as e:
                logger.error(f"[{self.config.id}] Error during model selection step for '{step.selector_key}' (Selector: {selector_str}): {e}", exc_info=True)
                raise RuntimeError(f"Failed model selection step for {self.config.id}/{target_model_variant_id}: {e}") from e
        
        logger.info(f"[{self.config.id}] Model selection flow for variant '{target_model_variant_id}' completed.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def send_message(self, message: str, target_model_variant_id: Optional[str] = None) -> Union[str, AsyncGenerator[str, None]]:
        if not self.managed_browser_instance:
            raise RuntimeError(f"Automator for {self.config.id} is not properly initialized or browser instance is missing.")
        
        if self.managed_browser_instance.is_mock:
            self.managed_browser_instance.request_count += 1
            self.managed_browser_instance.last_request_time = time.time()
            mock_responses_config = self.config.mock_responses or {}
            default_response = mock_responses_config.get('default', {})
            if default_response.get('is_error', False): raise Exception(default_response.get('error_message', 'Mock error'))
            if default_response.get('is_timeout', False):
                await asyncio.sleep(default_response.get('timeout_ms', 100) / 1000)
                raise asyncio.TimeoutError("Mock timeout")
            if self.config.response_handling.type == "stream":
                chunks = default_response.get('streaming', ['Mock chunk 1\n', 'Mock chunk 2\n', 'Mock chunk 3\n'])
                async def mock_stream_wrapper():
                    for chunk in chunks:
                        await asyncio.sleep(0.05)
                        yield chunk
                return mock_stream_wrapper()
            else:
                await asyncio.sleep(0.05)
                return default_response.get('text', 'Mock full response')

        page = self.managed_browser_instance.page
        self.managed_browser_instance.request_count += 1
        self.managed_browser_instance.last_request_time = time.time()

        logger.debug(f"[{self.config.id}] ENTERING send_message method.")

        try:
            logger.debug(f"[{self.config.id}] Step 1: Syncing page options...")
            await self._sync_page_options(page)
            logger.debug(f"[{self.config.id}] Step 1: Page options sync complete.")

            if target_model_variant_id:
                logger.debug(f"[{self.config.id}] Step 2: Executing model selection for '{target_model_variant_id}'...")
                await self._execute_model_selection_flow(target_model_variant_id, page)
                logger.debug(f"[{self.config.id}] Step 2: Model selection complete.")

            input_selector = self.config.selectors["input_area"]
            submit_selector = self.config.selectors["submit_button"]

            logger.debug(f"[{self.config.id}] Step 3: Attempting to fill input area with selector: {input_selector}")
            await page.fill(input_selector, message, timeout=20000)
            logger.debug(f"[{self.config.id}] Step 3: Successfully filled input area.")

            logger.debug(f"[{self.config.id}] Step 4: Attempting to find submit button with selector: {submit_selector}")
            submit_button_element = await page.query_selector(submit_selector)
            if not submit_button_element:
                logger.error(f"[{self.config.id}] FAILED to find submit button element. It's None.")
                raise RuntimeError(f"Submit button '{submit_selector}' not found on {self.config.id}")
            logger.debug(f"[{self.config.id}] Step 4: Found submit button element. Is enabled? {await submit_button_element.is_enabled()}")

            if await submit_button_element.is_disabled():
                logger.warning(f"[{self.config.id}] Submit button is disabled. Waiting briefly...")
                await asyncio.sleep(0.5)
                if await submit_button_element.is_disabled():
                    logger.error(f"[{self.config.id}] Submit button remained disabled.")
                    raise RuntimeError(f"Submit button '{submit_selector}' remained disabled on {self.config.id}")

            logger.debug(f"[{self.config.id}] Step 5: Attempting to click submit button.")
            await submit_button_element.click(timeout=20000)
            logger.debug(f"[{self.config.id}] Step 5: Clicked submit button successfully.")

            if self.config.response_handling.type == "stream":
                logger.info(f"[{self.config.id}] Starting to handle streaming response...")
                return self._handle_streaming_response(page)
            else:
                logger.info(f"[{self.config.id}] Starting to handle full response...")
                return await self._handle_full_response(page)
        except Exception as e:
            logger.error(f"[{self.config.id}] EXCEPTION in send_message: {e}", exc_info=True)
            error_snapshot_dir = Path("error_snapshots")
            error_snapshot_dir.mkdir(exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            try:
                if page and not page.is_closed():
                    screenshot_path = error_snapshot_dir / f"{self.config.id}_{timestamp}_error.png"
                    await page.screenshot(path=screenshot_path)
                    logger.info(f"Saved error screenshot to {screenshot_path}")
            except Exception as se:
                logger.error(f"Failed to save error screenshot: {se}")
            raise RetryError(f"Final attempt failed for send_message on {self.config.id}: {e}")

    async def _handle_streaming_response(self, page: Page) -> AsyncGenerator[str, None]:
        last_text = ""
        bytes_yielded = 0
        response_area_selector = self.config.selectors[self.config.response_handling.extraction_selector_key]
        text_property = self.config.response_handling.stream_text_property
        
        # 【核心修改】1. 初始化流状态，增加 'thinking_indicator_seen'
        stream_state = {
            '_stream_start_time': time.time(),
            'thinking_indicator_seen': False
        }
        logger.debug(f"[{self.config.id}] Streaming from selector '{response_area_selector}', property '{text_property}'")

        # 【核心修改】2. 增加一个小的初始延迟，让页面有时间加载“思考中”指示器
        await asyncio.sleep(0.5)

        loop_count = 0
        while True:
            try:
                loop_count += 1

                # 【核心修改】3. 在检查结束条件前，先检查“思考中”指示器是否已出现
                # 这段代码会找到 'element_disappears' 类型的条件，并检查其选择器对应的元素是否可见
                # 如果可见，它会将 'thinking_indicator_seen' 状态翻转为 True
                thinking_indicator_condition = next((c for c in self.config.response_handling.stream_end_conditions if c.type == "element_disappears"), None)
                if thinking_indicator_condition and not stream_state['thinking_indicator_seen']:
                    thinking_selector = self.config.selectors.get(thinking_indicator_condition.selector_key)
                    if thinking_selector:
                        # 使用短超时检查，因为它在循环中
                        is_visible = await page.is_visible(thinking_selector, timeout=50)
                        if is_visible:
                            stream_state['thinking_indicator_seen'] = True
                            logger.info(f"[{self.config.id}] 'thinking_indicator' 已出现。现在开始监控其消失。")

                # 检查所有结束条件
                sorted_end_conditions = sorted(
                    self.config.response_handling.stream_end_conditions,
                    key=lambda c: c.priority
                )
                should_end_stream = False
                for condition in sorted_end_conditions:
                    if await self._check_stream_end_condition(page, condition, last_text, stream_state):
                        should_end_stream = True
                        break
                if should_end_stream:
                    logger.info(f"[{self.config.id}] Stream loop ended by a condition on iteration {loop_count}.")
                    break

                # 每次循环都尝试获取元素
                element = await page.query_selector(response_area_selector)

                logger.debug(f"[{self.config.id}] Stream loop #{loop_count}: Element found? {'Yes' if element else 'No'}")

                current_text = ""
                if element:
                    if text_property == "textContent": current_text = await element.text_content() or ""
                    elif text_property == "innerText": current_text = await element.inner_text() or ""
                    elif text_property == "innerHTML": current_text = await element.inner_html() or ""
                    else:
                        logger.warning(f"Unsupported stream_text_property: {text_property}. Defaulting to text_content.")
                        current_text = await element.text_content() or ""
                    
                    logger.debug(f"[{self.config.id}] Stream loop #{loop_count}: Extracted text (len {len(current_text)}): '{current_text[:150]}...'")
                
                if current_text != last_text:
                    new_content = current_text[len(last_text):]
                    if new_content:
                        logger.info(f"[{self.config.id}] >>> Yielding new chunk (len {len(new_content)}): '{new_content[:100]}...'")
                        bytes_yielded += len(new_content.encode('utf-8'))
                        yield new_content
                        last_text = current_text
                        stream_state['_text_stable_since'] = None
                elif '_text_stable_since' not in stream_state or stream_state['_text_stable_since'] is None:
                     stream_state['_text_stable_since'] = time.time()

                await asyncio.sleep(self.config.response_handling.stream_poll_interval_ms / 1000)
            except Exception as e:
                logger.error(f"Error during streaming for {self.config.id}: {e}", exc_info=True)
                break
        
        duration = time.time() - stream_state['_stream_start_time']
        logger.info(
            f"[{self.config.id}] Streaming response finished. Duration: {duration:.2f}s, "
            f"Bytes: {bytes_yielded}, Throughput: {bytes_yielded/duration if duration > 0 else 0:.2f} B/s"
        )

    async def _check_stream_end_condition(
        self, page: Page, condition: StreamEndCondition, current_text: str, stream_state: Dict[str, Any]
    ) -> bool:
        result = False
        try:
            if condition.type == "element_disappears":
                # 【核心修改】4. 增加状态检查：只有在指示器出现过之后，才检查它是否消失
                if not stream_state.get('thinking_indicator_seen', False):
                    return False  # 如果指示器从未出现过，此条件不能触发，防止提前退出

                if not condition.selector_key: return False
                selector = self.config.selectors.get(condition.selector_key)
                if not selector: return False
                
                # 现在检查元素是否已隐藏或不存在
                result = await page.is_hidden(selector, timeout=100)

            elif condition.type == "element_appears":
                if not condition.selector_key: return False
                selector = self.config.selectors.get(condition.selector_key)
                if not selector: return False
                result = await page.is_visible(selector, timeout=100)
            elif condition.type == "text_stabilized":
                if not condition.stabilization_time_ms: return False
                stable_since = stream_state.get('_text_stable_since')
                if stable_since and (time.time() - stable_since) * 1000 >= condition.stabilization_time_ms:
                    logger.debug(f"[{self.config.id}] End condition check: Text stabilized for {condition.stabilization_time_ms} ms.")
                    result = True
            elif condition.type == "timeout":
                if not condition.timeout_seconds: return False
                start_time = stream_state.get('_stream_start_time', time.time())
                if (time.time() - start_time) >= condition.timeout_seconds:
                    logger.debug(f"[{self.config.id}] End condition check: Stream timeout of {condition.timeout_seconds}s reached.")
                    result = True
            else:
                logger.warning(f"Unknown stream end condition type: {condition.type}")
        except Exception as e:
            logger.error(f"Error checking stream end condition {condition.type} for {self.config.id}: {e}")
            result = False

        if result:
            logger.info(f"[{self.config.id}] >>> Stream END CONDITION MET: {condition.model_dump_json()}")
        return result

    async def _handle_full_response(self, page: Page) -> str:
        logger.info(f"[{self.config.id}] Waiting for full response elements...")
        extraction_selector = self.config.selectors[self.config.response_handling.extraction_selector_key]
        extraction_method = self.config.response_handling.extraction_method

        if self.config.response_handling.full_text_wait_strategy:
            for strategy_idx, strategy in enumerate(self.config.response_handling.full_text_wait_strategy):
                try:
                    strategy_selector_key = strategy.selector_key
                    if not strategy_selector_key:
                        logger.warning(f"[{self.config.id}] Wait strategy {strategy_idx} missing selector_key: {strategy.type}. Skipping.")
                        continue
                    
                    strat_selector = self.config.selectors.get(strategy_selector_key)
                    if not strat_selector:
                        logger.warning(f"[{self.config.id}] Selector for wait strategy key '{strategy_selector_key}' not found. Skipping.")
                        continue

                    logger.debug(f"[{self.config.id}] Applying wait strategy: {strategy.type} with selector {strat_selector}")
                    wait_timeout = 30000 
                    if strategy.type == "element_disappears":
                        await page.wait_for_selector(strat_selector, state="hidden", timeout=wait_timeout)
                    elif strategy.type == "element_appears":
                         await page.wait_for_selector(strat_selector, state="visible", timeout=wait_timeout)
                    elif strategy.type == "element_contains_text_or_is_not_empty":
                         await page.wait_for_function(f"""
                            () => {{
                                const el = document.querySelector('{strat_selector.replace("'", "\\'")}');
                                return el && (el.textContent || el.innerText || '').trim() !== '';
                            }}
                         """, timeout=wait_timeout)
                    elif strategy.type == "element_attribute_equals":
                        attribute_name = getattr(strategy, 'attribute_name', None)
                        attribute_value = getattr(strategy, 'attribute_value', None)
                        if attribute_name is None or attribute_value is None:
                            logger.warning(f"Wait strategy 'element_attribute_equals' missing attribute_name or attribute_value. Skipping.")
                            continue
                        await page.wait_for_function(f"""
                            () => {{
                                const el = document.querySelector('{strat_selector.replace("'", "\\'")}');
                                return el && el.getAttribute('{attribute_name}') === '{attribute_value}';
                            }}
                        """, timeout=wait_timeout)
                    logger.debug(f"[{self.config.id}] Wait strategy satisfied: {strategy.type}")
                except Exception as e:
                    logger.warning(f"[{self.config.id}] Timeout or error waiting for strategy {strategy.type} (selector: {strat_selector}): {e}")
        
        logger.info(f"[{self.config.id}] Extracting text from {extraction_selector} using {extraction_method}")
        element = await page.query_selector(extraction_selector)
        if not element:
            logger.warning(f"[{self.config.id}] Response element '{extraction_selector}' not found immediately. Waiting briefly...")
            await asyncio.sleep(1)
            element = await page.query_selector(extraction_selector)
            if not element:
                logger.error(f"[{self.config.id}] Response element not found: {extraction_selector}")
                raise ValueError(f"Response element not found: {extraction_selector} on site {self.config.id}")

        text_content = ""
        if extraction_method == "textContent": text_content = await element.text_content() or ""
        elif extraction_method == "innerText": text_content = await element.inner_text() or ""
        elif extraction_method == "innerHTML": text_content = await element.inner_html() or ""
        else:
            logger.warning(f"Unsupported extraction_method: {extraction_method}. Defaulting to text_content.")
            text_content = await element.text_content() or ""
        
        logger.info(f"[{self.config.id}] Full response extracted. Length: {len(text_content)}")
        return text_content.strip()

    async def send_prompt_and_get_response(self, prompt: str, stream_callback=None, target_model_variant_id: Optional[str] = None) -> Union[str, AsyncGenerator[str, None]]:
        await self._initialized_event.wait()
        start_time = time.time()
        logger.info(f"[{self.config.id}] Processing prompt (len {len(prompt)}): {prompt[:50]}... (Stream: {bool(stream_callback)}, Variant: {target_model_variant_id or 'Default'})")

        try:
            response_or_generator = await self.send_message(prompt, target_model_variant_id=target_model_variant_id)
            if stream_callback and hasattr(response_or_generator, '__aiter__'):
                logger.info(f"[{self.config.id}] Returning stream_wrapper for callback processing.")
                async def stream_wrapper():
                    full_response_text = ""
                    chunk_count = 0
                    total_bytes = 0
                    try:
                        async for chunk in response_or_generator:
                            await stream_callback(chunk)
                            full_response_text += chunk
                            chunk_count += 1
                            total_bytes += len(chunk.encode('utf-8'))
                            yield chunk
                        duration = time.time() - start_time
                        logger.info(
                            f"[{self.config.id}] Streaming with callback completed. Duration: {duration:.2f}s, "
                            f"Chunks: {chunk_count}, Bytes: {total_bytes}, "
                            f"Avg Throughput: {total_bytes/duration if duration > 0 else 0:.2f} B/s"
                        )
                    except Exception as e:
                        logger.error(f"[{self.config.id}] Error in stream_wrapper: {e}", exc_info=True)
                        raise
                return stream_wrapper()
            elif isinstance(response_or_generator, str):
                duration = time.time() - start_time
                logger.info(f"[{self.config.id}] Non-streaming response received. Duration: {duration:.2f}s, Length: {len(response_or_generator)}")
                return response_or_generator
            elif hasattr(response_or_generator, '__aiter__'):
                logger.info(f"[{self.config.id}] Collecting non-callback stream response...")
                full_response_text_list = []
                async for chunk in response_or_generator: full_response_text_list.append(chunk)
                full_response_text = "".join(full_response_text_list)
                duration = time.time() - start_time
                logger.info(f"[{self.config.id}] Collected non-callback stream. Duration: {duration:.2f}s, Length: {len(full_response_text)}")
                return full_response_text
            else:
                logger.error(f"[{self.config.id}] Unexpected response type from send_message: {type(response_or_generator)}")
                raise TypeError("Unexpected response type from send_message")
        except RetryError as e:
            logger.error(f"[{self.config.id}] All retry attempts failed for prompt: {prompt[:50]}. Error: {e.last_attempt.exception()}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"[{self.config.id}] Unhandled error processing prompt: {prompt[:50]}. Error: {e}", exc_info=True)
            raise

    async def is_healthy(self) -> bool:
        await self._initialized_event.wait()
        if not self.managed_browser_instance or self.managed_browser_instance.is_mock:
            logger.debug(f"[{self.config.id}] Health check: Instance is mock or not present. Assuming healthy for mock.")
            return True
        if not self.managed_browser_instance.page or self.managed_browser_instance.page.is_closed():
            logger.warning(f"[{self.config.id}] Health check: Page is None or closed.")
            return False

        page = self.managed_browser_instance.page
        hc_config = self.config.health_check
        try:
            title = await page.title()
            if not title:
                logger.warning(f"[{self.config.id}] Health check: Page title is empty.")
                return False
            logger.debug(f"[{self.config.id}] Health check: Page title: {title}")

            hc_element_selector_key = hc_config.check_element_selector_key
            hc_selector = self.config.selectors.get(hc_element_selector_key)
            if not hc_selector:
                logger.warning(f"Health check: HC selector key '{hc_element_selector_key}' not in selectors.")
                return False

            logger.debug(f"[{self.config.id}] Health check: Waiting for element '{hc_selector}' (timeout: {hc_config.timeout_seconds}s)")
            await page.wait_for_selector(hc_selector, state="visible", timeout=hc_config.timeout_seconds * 1000)
            
            logger.info(f"[{self.config.id}] Health check: PASSED.")
            return True
        except Exception as e:
            logger.warning(f"[{self.config.id}] Health check: FAILED. Error: {e}")
            return False

    async def cleanup(self):
        logger.info(f"Cleaning up LLMWebsiteAutomator for {self.config.id}...")
        if self.managed_browser_instance:
            await self.managed_browser_instance.cleanup()
            self.managed_browser_instance = None
        if self._playwright_instance:
            try:
                await self._playwright_instance.stop()
                logger.info(f"Playwright instance stopped for {self.config.id}.")
            except Exception as e: logger.error(f"Error stopping Playwright for {self.config.id}: {e}")
            finally: self._playwright_instance = None
        self._initialized_event.clear()

    def get_request_count(self) -> int:
        return self.managed_browser_instance.request_count if self.managed_browser_instance else 0

    async def get_memory_usage(self) -> float:
        return await self.managed_browser_instance.get_memory_usage() if self.managed_browser_instance else 0.0

    async def should_recycle_based_on_metrics(self) -> bool:
        if not self.managed_browser_instance: return False
        return await self.managed_browser_instance.should_recycle(
            self.config.max_requests_per_instance,
            self.config.max_memory_per_instance_mb
        )
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

  # call_api.py
  
```
import requests
import json
import argparse
import os
from typing import Optional

def call_chat_completion_api(
    model_id: str,
    prompt: str,
    api_key: str,
    stream: bool = False,
    api_base_url: str = "http://localhost:8000"
):
    """
    调用本地的 LLM 网页自动化 API。

    Args:
        model_id (str): 配置中 LLM 站点的 ID (例如 "wenxiaobai")，
                        或站点 ID/模型变体 ID (例如 "wenxiaobai/deepseek-v3")。
        prompt (str): 要发送给 LLM 的用户提示。
        api_key (str): 访问 API 的密钥 (与 .env 中的 API_KEY 对应)。
        stream (bool): 如果为 True，则启用流式响应。
        api_base_url (str): 本地 API 服务的基地址。
    """
    url = f"{api_base_url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": api_key
    }

    messages = [{"role": "user", "content": prompt}]

    data = {
        "model": model_id,
        "messages": messages,
        "stream": stream
    }

    print(f"\n--- 调用 API ---")
    print(f"URL: {url}")
    print(f"模型: {model_id} (流式: {stream})")
    print(f"提示: {prompt[:50]}...")
    print(f"----------------\n")

    try:
        if stream:
            # 流式请求
            full_response_content = ""
            with requests.post(url, headers=headers, json=data, stream=True) as response:
                response.raise_for_status() # 检查 HTTP 错误，如果状态码不是 2xx 则抛出异常

                print("流式响应:")
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        
                        if decoded_line.startswith("data:"):
                            json_data_str = decoded_line[len("data:"):].strip()
                            
                            if json_data_str == "[DONE]":
                                break # 流结束标记
                            
                            try:
                                chunk = json.loads(json_data_str)
                                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if content:
                                    print(content, end='', flush=True) # 实时打印内容
                                    full_response_content += content
                            except json.JSONDecodeError:
                                print(f"\n[API 响应解析错误] 无法解析 JSON 行: {json_data_str}")
                                # 可以根据需要记录更详细的错误
                print("\n\n流式响应结束。")
                print(f"总计接收字符: {len(full_response_content)}")

        else:
            # 非流式请求
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status() # 检查 HTTP 错误

            print("非流式响应:")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))

    except requests.exceptions.HTTPError as e:
        print(f"\n[HTTP 错误] 请求失败: {e}")
        if e.response is not None:
            print(f"服务器响应: {e.response.text}")
    except requests.exceptions.ConnectionError as e:
        print(f"\n[连接错误] 无法连接到服务器。请确保您的 FastAPI 服务已运行在 {api_base_url}。错误: {e}")
    except requests.exceptions.Timeout as e:
        print(f"\n[超时错误] 请求超时。错误: {e}")
    except requests.exceptions.RequestException as e:
        print(f"\n[未知请求错误] 发生未知请求错误: {e}")
    except Exception as e:
        print(f"\n[意外错误] 发生意外错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="调用本地 LLM 网页自动化 API")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="LLM 站点 ID 或 '站点ID/模型变体ID' (例如: 'wenxiaobai' 或 'wenxiaobai/deepseek-v3')")
    parser.add_argument("--prompt", "-p", type=str, required=True,
                        help="发送给 LLM 的用户提示")
    parser.add_argument("--stream", "-s", action="store_true",
                        help="启用流式响应 (默认为非流式)")
    parser.add_argument("--api-key", "-k", type=str,
                        help="API 密钥 (如果未提供，将尝试从环境变量 API_KEY 获取)")
    parser.add_argument("--url", "-u", type=str, default="http://localhost:8000",
                        help="本地 API 服务的基地址 (默认为 http://localhost:8000)")

    args = parser.parse_args()

    # 如果命令行未提供 API 密钥，则尝试从环境变量获取
    api_key_to_use = args.api_key
    if not api_key_to_use:
        api_key_to_use = os.getenv("API_KEY")
        if not api_key_to_use:
            print("错误: 未提供 API 密钥。请通过 --api-key 参数提供，或在 .env 文件中设置 API_KEY 环境变量。", file=os.sys.stderr)
            os.sys.exit(1)

    call_chat_completion_api(
        model_id=args.model,
        prompt=args.prompt,
        api_key=api_key_to_use,
        stream=args.stream,
        api_base_url=args.url
    )
  ```

  # config.yaml
  
```
# Global settings for all sites
global_settings:
  timeout: 60
  max_retries: 3
  retry_delay: 5
  log_level: "DEBUG"
  request_delay: 0.5 # Default delay after actions like clicking 'new chat' (in seconds)
  concurrent_requests: 5
  user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
  idle_instance_check_interval_seconds: 60
  max_stale_instance_lifetime_seconds: 300

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
  - id: "wenxiaobai"
    name: "闻小AI (wenxiaobai.com)"
    enabled: true
    url: "https://www.wenxiaobai.com/"
    firefox_profile_dir: "profiles/wenxiaobai_profile"
    playwright_launch_options:
      headless: false
      viewport:
        width: 1920
        height: 1080
    use_stealth: true
    pool_size: 1
    max_requests_per_instance: 100
    max_memory_per_instance_mb: 1024

    selectors:
      new_chat_button: "xpath=//div[contains(@class, 'NewChat_left') and .//span[normalize-space()='新对话']]"
      input_area: "xpath=//textarea[@placeholder='给 小白 发送消息']"
      submit_button: "xpath=//div[@class='MsgInput_icon_container__4burH']//*[name()='svg']"
      response_markdown_body_last_assistant: "xpath=(//div[contains(@class, 'markdown-body') and @data-sentry-component='Markdown' and @data-sentry-source-file='index.tsx'])[1]"
      thinking_indicator: 'xpath=//*[@id="chat_turn_container"]//div[contains(@class, "loading-title") and normalize-space(.)="思考中..."]'
      model_selection_popup_trigger: "xpath=//div[contains(@class, 'PopupItem_btn_act__') and contains(@data-key, 'deepseek') and .//div[contains(@class, 'PopupItem_btn_msg__')]]"
      wenxiaobai_deepseek_v3_choice: "xpath=//div[contains(@class, 'PopupItem_option_item__') and .//div[contains(@class, 'PopupItem_option_text__') and text()='日常问答（V3）']]"
      wenxiaobai_deepseek_r1_choice: "xpath=//div[contains(@class, 'PopupItem_option_item__') and .//div[contains(@class, 'PopupItem_option_text__') and text()='深度思考（R1）']]"
      wenxiaobai_qwen3_choice: "xpath=//div[contains(@class, 'PopupItem_option_item__') and .//div[contains(@class, 'PopupItem_option_text__') and text()='千问 3']]"
      response_container_general: "xpath=(//div[contains(@class, 'Answser_answer_body_content')])[1]"
      last_response_options_trigger: "xpath=//*[@id='chat_turn_container']//div[starts-with(@class, 'TurnCard_right_opts') and contains(@class, 'TurnCard_more_opts_show')]"
      health_check_element: "xpath=//textarea[@placeholder='给 小白 发送消息']"

    model_variants:
      - id: "deepseek-v3"
        name_on_page: "日常问答（V3）"
        selection_flow:
          - selector_key: "model_selection_popup_trigger"
            description: "Click to open model selection popup"
            wait_after_ms: 500
          - selector_key: "wenxiaobai_deepseek_v3_choice"
            description: "Select DeepSeek V3 model"
            wait_after_ms: 200
      - id: "deepseek-r1"
        name_on_page: "深度思考（R1）"
        selection_flow:
          - selector_key: "model_selection_popup_trigger"
            wait_after_ms: 500
          - selector_key: "wenxiaobai_deepseek_r1_choice"
            wait_after_ms: 200
      - id: "qwen3"
        name_on_page: "千问 3"
        selection_flow:
          - selector_key: "model_selection_popup_trigger"
            wait_after_ms: 500
          - selector_key: "wenxiaobai_qwen3_choice"
            wait_after_ms: 200

    options_to_sync: []

    response_handling:
      type: "stream"
      stream_poll_interval_ms: 200
      stream_text_property: "textContent"
      extraction_selector_key: "response_markdown_body_last_assistant"
      extraction_method: "innerText"

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

      full_text_wait_strategy:
        - type: "element_disappears"
          selector_key: "thinking_indicator"
        - type: "element_contains_text_or_is_not_empty"
          selector_key: "response_markdown_body_last_assistant"

    health_check:
      enabled: true
      interval_seconds: 300
      timeout_seconds: 60
      failure_threshold: 3
      check_element_selector_key: "health_check_element"
  ```

  # config1.yaml
  
```
# Global settings for all sites
global_settings:
  timeout: 60  # Default timeout in seconds
  max_retries: 3  # Default number of retries
  retry_delay: 5  # Delay between retries in seconds
  log_level: "debug"
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
  - id: "wenxiaobai.com"
    name: "wenxiaobai.com"
    enabled: true
    url: "https://www.wenxiaobai.com/"

    # Firefox Profile 配置
    firefox_profile_dir: "wenxiaobai.com_profile"

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
      input_area: "//textarea[@placeholder='给 小白 发送消息']"
      submit_button: "//div[@class='MsgInput_icon_container__4burH']//*[name()='svg']"
      response_container: "(//div[contains(@class, "Answser_answer_body_content")])[1]"
      last_response_message: "//*[@id="chat_turn_container"]//div[starts-with(@class, 'TurnCard_right_opts') and contains(@class, 'TurnCard_more_opts_show')]"
      streaming_response_target: "(//div[@id='chat_turn_container']//div[contains(@class, 'markdown-body') and @data-sentry-component='Markdown'])[1]"
      thinking_indicator: 'xpath=//*[@id="chat_turn_container"]//div[contains(@class, "loading-title") and normalize-space(.)="思考中..."]'
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

  # config2.yaml
  
```
# Global settings for all sites
global_settings:
  timeout: 60
  max_retries: 3
  retry_delay: 5
  log_level: "DEBUG" 
  request_delay: 0.5 # Default delay after actions like clicking 'new chat' (in seconds)
  concurrent_requests: 5
  user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
  idle_instance_check_interval_seconds: 60
  max_stale_instance_lifetime_seconds: 300

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
  - id: "wenxiaobai" 
    name: "闻小AI (wenxiaobai.com)"
    enabled: true
    url: "https://www.wenxiaobai.com/"
    firefox_profile_dir: "profiles/wenxiaobai_profile"
    # You can add site-specific request_delay if needed, e.g.:
    # request_delay: 0.8 
    playwright_launch_options:
      headless: false 
      viewport:
        width: 1920
        height: 1080
    use_stealth: true
    pool_size: 1
    max_requests_per_instance: 100
    max_memory_per_instance_mb: 1024

    selectors:
      new_chat_button: "xpath=//div[contains(@class, 'NewChat_left') and .//span[normalize-space()='新对话']]"
      input_area: "xpath=//textarea[@placeholder='给 小白 发送消息']"
      submit_button: "xpath=//div[@class='MsgInput_icon_container__4burH']//*[name()='svg']"
      response_markdown_body_last_assistant: "xpath=(//div[contains(@class, 'markdown-body') and @data-sentry-component='Markdown' and @data-sentry-source-file='index.tsx'])[1]"
      thinking_indicator: "xpath=(//*[@id="chat_turn_container"]//div[contains(@class, "loading-title") and normalize-space(.)="思考中..."])[1]"
      model_selection_popup_trigger: "xpath=//div[contains(@class, 'PopupItem_btn_act__') and contains(@data-key, 'deepseek') and .//div[contains(@class, 'PopupItem_btn_msg__')]]"
      wenxiaobai_deepseek_v3_choice: "xpath=//div[contains(@class, 'PopupItem_option_item__') and .//div[contains(@class, 'PopupItem_option_text__') and text()='日常问答（V3）']]"
      wenxiaobai_deepseek_r1_choice: "xpath=//div[contains(@class, 'PopupItem_option_item__') and .//div[contains(@class, 'PopupItem_option_text__') and text()='深度思考（R1）']]"
      wenxiaobai_qwen3_choice:       "xpath=//div[contains(@class, 'PopupItem_option_item__') and .//div[contains(@class, 'PopupItem_option_text__') and text()='千问 3']]"
      response_container_general: "xpath=(//div[contains(@class, 'Answser_answer_body_content')])[1]" 
      last_response_options_trigger: "xpath=(//*[@id='chat_turn_container']//div[starts-with(@class, 'TurnCard_right_opts') and contains(@class, 'TurnCard_more_opts_show')])[1]"
      health_check_element: "xpath=//textarea[@placeholder='给 小白 发送消息']"

    model_variants:
      - id: "deepseek-v3"
        name_on_page: "日常问答（V3）"
        selection_flow:
          - selector_key: "model_selection_popup_trigger"
            description: "Click to open model selection popup"
            wait_after_ms: 500
          - selector_key: "wenxiaobai_deepseek_v3_choice"
            description: "Select DeepSeek V3 model"
            wait_after_ms: 200
      - id: "deepseek-r1"
        name_on_page: "深度思考（R1）"
        selection_flow:
          - selector_key: "model_selection_popup_trigger"
            wait_after_ms: 500
          - selector_key: "wenxiaobai_deepseek_r1_choice"
            wait_after_ms: 200
      - id: "qwen3"
        name_on_page: "千问 3"
        selection_flow:
          - selector_key: "model_selection_popup_trigger"
            wait_after_ms: 500
          - selector_key: "wenxiaobai_qwen3_choice"
            wait_after_ms: 200
            
    options_to_sync: []

    response_handling:
      type: "stream"
      stream_poll_interval_ms: 200
      stream_text_property: "textContent" 
      extraction_selector_key: "response_markdown_body_last_assistant"
      extraction_method: "innerText" 
      
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
      
      full_text_wait_strategy:
        - type: "element_disappears"
          selector_key: "thinking_indicator"
        - type: "element_contains_text_or_is_not_empty"
          selector_key: "response_markdown_body_last_assistant"

    health_check:
      enabled: true
      interval_seconds: 300
      timeout_seconds: 60
      failure_threshold: 3
      check_element_selector_key: "health_check_element"
  ```

  # config_manager.py
  
```
import os
import yaml
import logging  # <--- 修改：导入 logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from models import AppConfig, LLMSiteConfig, GlobalSettings
# from utils import setup_logging  # <--- 修改：不再从 utils 导入 setup_logging

# Initialize logger
# <--- 关键修改：不再调用 setup_logging()，只获取一个 logger 实例。
# 这个 logger 会自动继承 main.py 中配置的根 logger 的设置。
logger = logging.getLogger("wrapper_api.config_manager")

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
    # This part only runs when you execute `python config_manager.py` directly
    # It needs its own logging setup for testing
    from utils import setup_logging
    setup_logging()
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

  # dev2.md
  
```
**开发文档：LLM 网页自动化 OpenAI API 适配器 (Windows 本地版)**

**版本：** 1.2


目录：

1.  项目概述与目标
2.  核心原则
3.  技术栈 (Windows 环境)
4.  项目文件结构
5.  核心模块设计
    *   5.1 配置管理 (config\_manager.py & config.yaml)
    *   5.2 浏览器自动化处理器 (browser\_handler.py)
    *   5.3 API 服务 (main.py)
    *   5.4 数据模型 (models.py)
    *   5.5 工具与日志 (utils.py)
6.  分阶段实施计划 (Windows 环境优化)
    *   阶段 0: 环境搭建与基础框架 (Windows 重点)
    *   阶段 1: 单一网站核心自动化 (非流式) 与基础测试
    *   阶段 2: FastAPI 接口封装 (非流式) 与初步并发控制
    *   阶段 3: 实现流式响应与性能考量
    *   阶段 4: 浏览器实例池与资源监控
    *   阶段 5: 鲁棒性、高级功能与可维护性提升
7.  规范与最佳实践
    *   7.1 日志规范与管理
    *   7.2 错误处理与快照
    *   7.3 安全注意事项 (Windows 本地)
    *   7.4 测试策略与节奏
    *   7.5 依赖管理
8.  部署与运行 (Windows 本地)

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
        *   根据 `response_handling.type` (stream/full\_text) 调用不同的等待和提取逻辑：
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
    1.  完善 `config.yaml` 中一个站点的详细配置 (selectors, profile, options\_to\_sync, response\_handling (full\_text), health\_check)。
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
# main.py
import os
import asyncio
import time
import uuid
import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Optional, AsyncGenerator, List, Any

from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from config_manager import get_config, LLMSiteConfig, reload_config as actual_reload_config, AppConfig
from browser_handler import LLMWebsiteAutomator
from utils import setup_logging, notify_on_critical_error
from models import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    ChatCompletionStreamResponse,
    OpenAIMessage,
    OpenAIChatChoice,
    OpenAIUsage,
    OpenAIChatStreamChoice,
    OpenAIChatStreamDelta,
    GlobalSettings,
    BaseModel
)

# --- 应用状态类 ---
class AppState:
    def __init__(self):
        self.automator_pools: Dict[str, asyncio.Queue[LLMWebsiteAutomator]] = {}
        self.site_configs: Dict[str, LLMSiteConfig] = {}
        self.global_settings: Optional[GlobalSettings] = None
        self.idle_monitor_task: Optional[asyncio.Task] = None
        self.stale_automators: Dict[LLMWebsiteAutomator, float] = {}

# --- FastAPI 生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 初始化日志和应用状态
    load_dotenv()
    setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"), log_file="logs/wrapper_api.log")
    logger = logging.getLogger("wrapper_api")
    app.state.logger = logger
    app.state.core = AppState()

    logger.info("执行启动逻辑...")
    try:
        initial_config = get_config()
        logger.info(f"配置加载成功。发现 {len(initial_config.llm_sites)} 个站点配置。")
        enabled_sites = [s for s in initial_config.llm_sites if s.enabled]
        logger.info(f"发现 {len(enabled_sites)} 个启用的站点。")
        if not enabled_sites:
            logger.warning("配置文件中没有启用的站点，将不会初始化任何浏览器实例。")
        
        await _initialize_pools(app, initial_config)
        logger.info(f"池初始化完成。创建了 {len(app.state.core.automator_pools)} 个自动化器池。")
    except Exception as e:
        logger.critical(f"启动时初始化池失败: {e}", exc_info=True)
        raise
    
    gs = app.state.core.global_settings
    if gs and gs.idle_instance_check_interval_seconds > 0:
        app.state.core.idle_monitor_task = asyncio.create_task(monitor_idle_instances_periodically(app))
    
    yield  # 应用开始处理请求
    
    # --- 关闭逻辑 ---
    logger.info("执行关闭逻辑...")
    core_state = app.state.core
    if core_state.idle_monitor_task and not core_state.idle_monitor_task.done():
        core_state.idle_monitor_task.cancel()
        try:
            await core_state.idle_monitor_task
        except asyncio.CancelledError:
            logger.info("空闲监控任务已取消。")

    for automator in list(core_state.stale_automators.keys()):
        await _cleanup_stale_automator(automator, "shutdown")
    core_state.stale_automators.clear()
    
    for model_id, pool in core_state.automator_pools.items():
        while not pool.empty():
            try:
                await pool.get_nowait().cleanup()
            except (asyncio.QueueEmpty, Exception) as e:
                logger.error(f"关闭时从 {model_id} 清理自动化器时出错: {e}")
                break
    core_state.automator_pools.clear()
    core_state.site_configs.clear()
    logger.info("所有自动化器已清理。")

# --- FastAPI 应用实例 ---
app = FastAPI(
    title="LLM API 包装器",
    description="具有模型选择、实例池和热重载的包装器 API。",
    version="1.3",
    lifespan=lifespan
)

# --- Helper Functions (需要 app.state) ---
async def _create_new_automator_instance(site_config: LLMSiteConfig) -> LLMWebsiteAutomator:
    logger = app.state.logger
    logger.info(f"正在为池创建新的自动化器实例: {site_config.id}")
    automator = LLMWebsiteAutomator(site_config)
    try:
        await automator.initialize()
        logger.info(f"成功为 {site_config.id} 初始化了新的自动化器")
        return automator
    except Exception as e:
        logger.error(f"在创建期间为 {site_config.id} 初始化新自动化器失败: {e}", exc_info=True)
        raise

async def _cleanup_stale_automator(automator: LLMWebsiteAutomator, reason: str):
    logger = app.state.logger
    core_state = app.state.core
    if automator in core_state.stale_automators:
        del core_state.stale_automators[automator]
    try:
        model_id_label = automator.config.id if automator.config else "unknown"
        logger.info(f"正在清理过时/失败的 {model_id_label} 自动化器，原因: {reason}")
        await automator.cleanup()
    except Exception as e:
        logger.error(f"为 {automator.config.id if automator.config else 'unknown'} 清理自动化器时出错: {e}")

async def monitor_idle_instances_periodically(app: FastAPI):
    logger = app.state.logger
    core_state = app.state.core
    if not core_state.global_settings:
        logger.error("无法启动空闲实例监控器：全局设置未加载。")
        return

    logger.info("启动空闲实例监控任务...")
    await asyncio.sleep(15)
    while True:
        try:
            gs = core_state.global_settings
            check_interval = gs.idle_instance_check_interval_seconds
            max_stale_life = gs.max_stale_instance_lifetime_seconds
            logger.debug(f"空闲/过时监控器正在运行。下一次检查在 {check_interval} 秒后。")
            await asyncio.sleep(check_interval)

            for automator, stale_since in list(core_state.stale_automators.items()):
                if (time.time() - stale_since) > max_stale_life:
                    logger.warning(f"{automator.config.id} 的过时自动化器已超过最大生命周期。强制清理。")
                    await _cleanup_stale_automator(automator, "max_stale_lifetime_exceeded")

            for model_id, pool in list(core_state.automator_pools.items()):
                site_config = core_state.site_configs.get(model_id)
                if not site_config or pool.empty(): continue
                
                idle_instance = await pool.get()
                recycled = False
                try:
                    is_healthy = await idle_instance.is_healthy()
                    should_recycle = await idle_instance.should_recycle_based_on_metrics()
                    if not is_healthy or should_recycle:
                        recycled = True
                        reason = ("unhealthy" if not is_healthy else "") + ("_metrics" if should_recycle else "")
                        await _cleanup_stale_automator(idle_instance, reason.strip('_'))
                        new_instance = await _create_new_automator_instance(site_config)
                        await pool.put(new_instance)
                except Exception as e:
                    recycled = True
                    logger.error(f"检查 {model_id} 的空闲实例时出错: {e}。正在回收。")
                    await _cleanup_stale_automator(idle_instance, "check_error")
                    try:
                        await pool.put(await _create_new_automator_instance(site_config))
                    except Exception as create_e:
                        logger.error(f"为 {model_id} 创建替换实例失败: {create_e}")
                finally:
                    if not recycled:
                        await pool.put(idle_instance)
        except asyncio.CancelledError:
            logger.info("空闲/过时实例监控任务已取消。")
            break
        except Exception as e:
            logger.error(f"空闲/过时监控器发生意外错误: {e}", exc_info=True)
            await asyncio.sleep(60)

async def _initialize_pools(app: FastAPI, config_to_load: AppConfig):
    logger = app.state.logger
    core_state = app.state.core
    logger.info("开始初始化或更新浏览器实例池...")
    core_state.global_settings = config_to_load.global_settings
    new_configs = {site.id: site for site in config_to_load.llm_sites if site.enabled}
    
    # ... (rest of the _initialize_pools logic, slightly adapted for app.state) ...
    # This logic is complex but less likely to be the core issue.
    # The main change is using core_state.automator_pools etc.
    core_state.site_configs = new_configs
    for site_id, site_cfg in new_configs.items():
        if site_id not in core_state.automator_pools:
            core_state.automator_pools[site_id] = asyncio.Queue(maxsize=site_cfg.pool_size)
        
        pool = core_state.automator_pools[site_id]
        while pool.qsize() < site_cfg.pool_size:
            try:
                new_instance = await _create_new_automator_instance(site_cfg)
                await pool.put(new_instance)
            except Exception as e:
                logger.error(f"向池 {site_id} 添加新实例失败: {e}")
                break # Stop if one fails
    logger.info("浏览器实例池初始化/更新完成。")


# --- API 端点 ---
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(key: str = Security(api_key_header)):
    expected_key = os.getenv("API_KEY")
    if not expected_key:
        # For local development, allow access if no key is set in .env
        return "development_key"
    if key == expected_key:
        return key
    raise HTTPException(status_code=403, detail="无效的 API 密钥")

@app.post("/v1/chat/completions")
async def chat_completions_endpoint(request: OpenAIChatCompletionRequest, req: Request):
    logger = req.app.state.logger
    core_state = req.app.state.core
    
    site_id, variant_id = (request.model.split("/", 1) + [None])[:2]
    logger.info(f"收到对站点 '{site_id}'，变体 '{variant_id or '默认'}' 的请求")

    site_config = core_state.site_configs.get(site_id)
    if not site_config:
        raise HTTPException(status_code=404, detail=f"模型 '{site_id}' 未找到或未启用。")
    pool = core_state.automator_pools.get(site_id)
    if not pool:
        raise HTTPException(status_code=500, detail=f"内部错误：模型 '{site_id}' 的池不可用。")

    automator = None
    try:
        timeout = core_state.global_settings.timeout if core_state.global_settings else 60
        automator = await asyncio.wait_for(pool.get(), timeout=timeout)
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])

        if request.stream:
            return StreamingResponse(
                _stream_generator(automator, prompt, variant_id, request.model, logger),
                media_type="text/event-stream"
            )
        else:
            resp_text = await automator.send_prompt_and_get_response(prompt, None, variant_id)
            return OpenAIChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4()}", object="chat.completion", created=int(time.time()), model=request.model,
                choices=[OpenAIChatChoice(index=0, message=OpenAIMessage(role="assistant", content=resp_text), finish_reason="stop")],
                usage=OpenAIUsage(prompt_tokens=len(prompt.split()), completion_tokens=len(resp_text.split()), total_tokens=len(prompt.split())+len(resp_text.split()))
            )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="服务暂时不可用，请稍后重试。")
    finally:
        if automator:
            await pool.put(automator)

async def _stream_generator(automator, prompt, variant_id, model_name, logger):
    response_id = f"chatcmpl-{uuid.uuid4()}"
    try:
        async for chunk in automator.send_prompt_and_get_response(prompt, None, variant_id):
            delta = OpenAIChatStreamDelta(content=chunk)
            choice = OpenAIChatStreamChoice(index=0, delta=delta, finish_reason=None)
            resp = ChatCompletionStreamResponse(id=response_id, created=int(time.time()), model=model_name, choices=[choice])
            yield f"data: {resp.model_dump_json(exclude_none=True)}\n\n"
        
        final_choice = OpenAIChatStreamChoice(index=0, delta=OpenAIChatStreamDelta(), finish_reason="stop")
        final_resp = ChatCompletionStreamResponse(id=response_id, created=int(time.time()), model=model_name, choices=[final_choice])
        yield f"data: {final_resp.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"流处理错误: {e}", exc_info=True)
        yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'stream_error'}})}\n\n"
        yield "data: [DONE]\n\n"
  ```

  # models.py
  
```
# models.py
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator, model_validator, HttpUrl, ConfigDict

# 枚举类型
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

# OpenAI API 模型
class OpenAIMessage(BaseModel):
    role: str = Field(..., description="消息作者的角色 (system, user, assistant)")
    content: str = Field(..., description="消息内容")

    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed_roles = ['system', 'user', 'assistant', 'function']
        if v not in allowed_roles:
            raise ValueError(f"角色必须是 {allowed_roles} 中的一个")
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
            raise ValueError("温度必须在 0 到 2 之间")
        return v

    @field_validator('top_p')
    @classmethod
    def validate_top_p(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Top_p 必须在 0 到 1 之间")
        return v

    @field_validator('presence_penalty', 'frequency_penalty')
    @classmethod
    def validate_penalty(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < -2 or v > 2):
            raise ValueError("惩罚值必须在 -2 到 2 之间")
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

# 配置模型
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
                raise ValueError("基本认证需要用户名和密码")
        elif auth_type == AuthType.BEARER:
            if not self.token:
                raise ValueError("Bearer 认证需要令牌")
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
        if not v.startswith(('http://', 'https://')):
            raise ValueError("基础 URL 必须以 http:// 或 https:// 开头")
        return v

class GlobalSettings(BaseModel):
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 5
    log_level: LogLevel = LogLevel.INFO
    request_delay: float = 1.0
    concurrent_requests: int = 5
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    idle_instance_check_interval_seconds: int = Field(default=60, description="检查池中空闲浏览器实例的间隔时间（秒）。")
    max_stale_instance_lifetime_seconds: int = Field(default=300, description="配置重载后，标记为过时的实例的最大生命周期（秒）。")

# LLM 站点配置模型
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
    priority: int = 999

class WaitStrategy(BaseModel):
    type: str
    selector_key: Optional[str] = None
    attribute_name: Optional[str] = None
    attribute_value: Optional[str] = None

class ResponseHandling(BaseModel):
    type: str
    stream_poll_interval_ms: int = 150
    stream_text_property: str = "textContent"
    stream_end_conditions: List[StreamEndCondition] = Field(default_factory=list)
    full_text_wait_strategy: List[WaitStrategy] = Field(default_factory=list)
    extraction_selector_key: str
    extraction_method: str = "innerText"
    stream_error_handling: Optional[Dict[str, Any]] = None

class OptionSync(BaseModel):
    id: str
    description: str
    selector_key: str
    action: str
    target_value_on_page: Optional[str] = None
    on_failure: str = "abort"

class HealthCheck(BaseModel):
    enabled: bool = True
    interval_seconds: int = 300
    timeout_seconds: int = 45
    failure_threshold: int = 3
    check_element_selector_key: str

# 模型选择流程的新模型
class ModelSelectionStep(BaseModel):
    action: str = Field(default="click", description="要执行的操作，例如 'click'")
    selector_key: str = Field(..., description="主 'selectors' 字典中目标元素对应的键")
    description: Optional[str] = None
    wait_after_ms: Optional[int] = Field(default=None, description="此操作后可选的延迟（毫秒）")

class ModelVariantConfig(BaseModel):
    id: str = Field(..., description="此模型变体的唯一 ID，用于 API 请求（例如 'deepseek-v3'）。这是 site_id/ 后面的部分")
    name_on_page: Optional[str] = Field(default=None, description="页面上显示的人类可读名称，用于日志/参考。")
    selection_flow: List[ModelSelectionStep] = Field(..., description="选择此模型变体的一系列操作。")

class LLMSiteConfig(BaseModel):
    id: str
    name: str
    enabled: bool = True
    url: str
    firefox_profile_dir: str
    playwright_launch_options: PlaywrightLaunchOptions = Field(default_factory=PlaywrightLaunchOptions)
    use_stealth: bool = True
    pool_size: int = Field(default=1, ge=1, description="此站点池的浏览器实例数。")
    max_requests_per_instance: int = 100
    max_memory_per_instance_mb: int = 1024
    selectors: Dict[str, str]
    backup_selectors: Optional[Dict[str, List[str]]] = None
    options_to_sync: List[OptionSync] = Field(default_factory=list)
    
    model_variants: Optional[List[ModelVariantConfig]] = Field(default=None, description="此站点上不同模型变体的配置以及如何选择它们。")
    
    response_handling: ResponseHandling
    health_check: HealthCheck
    mock_mode: bool = False
    mock_responses: Optional[Dict[str, Any]] = Field(
        default=None,
        description="用于测试目的的模拟响应"
    )

    # 修正点：添加 model_config 以解决 Pydantic 警告
    model_config = ConfigDict(
        protected_namespaces=()
    )

class AppConfig(BaseModel):
    global_settings: GlobalSettings
    api_settings: APIConfigurations
    proxy_settings: ProxySettings = Field(default_factory=ProxySettings)
    llm_sites: List[LLMSiteConfig] = Field(default_factory=list)

class NotificationConfig(BaseModel):
    log_only: bool = True

# API 别名
ChatCompletionRequest = OpenAIChatCompletionRequest
  ```

  # pw.py
  
```
import asyncio
import time
from pathlib import Path
from playwright.async_api import async_playwright, Page, BrowserContext, ElementHandle
import logging
from collections import Counter, defaultdict

# --- 配置区 ---
TARGET_URL = "https://www.wenxiaobai.com/"  # 修改为你的 LLM 网页 URL
MONITOR_DURATION_SECONDS = 100  # 手动提交后，监控页面变化的时长
POLL_INTERVAL_SECONDS = 0.2   # 检查 MutationObserver 记录的频率
MIN_TEXT_CHANGE_LENGTH = 3    # 文本节点内容变化被认为是“有效”的最小长度
MAX_XPATH_DEPTH = 10          # 向上追溯父节点以寻找共同容器的最大深度
MIN_OCCURRENCES_FOR_CANDIDATE = 3 # 一个 XPath 需要至少变化这么多次才被认为是候选容器

LOG_FILE_PATH = "logs/dynamic_stream_detector.log"

# --- 日志设置 ---
detector_logger = logging.getLogger("DynamicStreamDetector")
detector_logger.setLevel(logging.INFO) # INFO 或 DEBUG
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setFormatter(formatter)
detector_logger.addHandler(ch)

Path("logs").mkdir(exist_ok=True)
fh = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
fh.setFormatter(formatter)
detector_logger.addHandler(fh)

ENABLE_PLAYWRIGHT_TRACING = True
TRACE_DIR = Path("traces/dynamic_stream_detection")

# --- JavaScript Code to Inject ---
JS_MUTATION_OBSERVER_SCRIPT = """
async (options) => {
    if (window.llmStreamChanges) { // Clear previous if any
        window.llmStreamChanges.observer.disconnect();
        window.llmStreamChanges.records = [];
    }

    window.llmStreamChanges = {
        records: [],
        observer: null,
        options: options || {} // minTextChangeLength
    };

    const getElementXPath = (element) => {
        if (!element || !element.parentNode) return null; // Element might be detached
        if (element.id !== '') return `id("${element.id}")`;
        if (element === document.body) return '/html/body'; // More stable than just 'body'

        let ix = 0;
        const siblings = element.parentNode.childNodes;
        for (let i = 0; i < siblings.length; i++) {
            const sibling = siblings[i];
            if (sibling === element) {
                const parentPath = getElementXPath(element.parentNode);
                if (!parentPath) return null; // If parent has no path, this one won't either
                return parentPath + '/' + element.tagName.toLowerCase() + '[' + (ix + 1) + ']';
            }
            if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
                ix++;
            }
        }
        return null; // Should not happen if element is in DOM
    };

    const observer = new MutationObserver((mutationsList) => {
        for (const mutation of mutationsList) {
            let targetElement = null;
            let changeType = mutation.type;
            let textSample = "";

            if (mutation.type === 'childList') {
                if (mutation.addedNodes.length > 0) {
                    targetElement = mutation.addedNodes[0].parentElement || mutation.target; // Prefer parent if adding nodes
                    // Try to get text from the added node or its children
                    const addedNode = mutation.addedNodes[0];
                    if (addedNode.textContent && addedNode.textContent.trim().length > 0) {
                        textSample = addedNode.textContent.trim().substring(0, 50);
                    } else if (addedNode.nodeType === Node.ELEMENT_NODE && addedNode.innerText) {
                         textSample = addedNode.innerText.trim().substring(0,50);
                    }
                } else if (mutation.removedNodes.length > 0) {
                    // Less likely for streaming output, but good to note
                    targetElement = mutation.target;
                }
            } else if (mutation.type === 'characterData') {
                targetElement = mutation.target.parentElement; // The text node's parent
                if (mutation.target.textContent &&
                    mutation.target.textContent.trim().length >= (window.llmStreamChanges.options.minTextChangeLength || 1)) {
                    textSample = mutation.target.textContent.trim().substring(0, 50);
                } else {
                    continue; // Ignore small character data changes
                }
            }

            if (targetElement) {
                const xpath = getElementXPath(targetElement);
                if (xpath) { // Only record if xpath could be generated
                    window.llmStreamChanges.records.push({
                        xpath: xpath,
                        type: changeType,
                        timestamp: Date.now(),
                        textSample: textSample.replace(/\\n/g, ' '),
                        targetTagName: targetElement.tagName.toLowerCase(),
                        // attributes: Array.from(targetElement.attributes).map(attr => ({name: attr.name, value: attr.value}))
                    });
                }
            }
        }
    });

    observer.observe(document.documentElement, {
        childList: true,
        subtree: true,
        characterData: true,
        characterDataOldValue: false // Don't need old value for this
    });
    window.llmStreamChanges.observer = observer;
    return true; // Signal observer started
}
"""

JS_GET_CHANGES_SCRIPT = """
() => {
    if (window.llmStreamChanges && window.llmStreamChanges.records) {
        const records = window.llmStreamChanges.records;
        window.llmStreamChanges.records = []; // Clear after fetching
        return records;
    }
    return [];
}
"""

JS_STOP_OBSERVER_SCRIPT = """
() => {
    if (window.llmStreamChanges && window.llmStreamChanges.observer) {
        window.llmStreamChanges.observer.disconnect();
        delete window.llmStreamChanges;
        return true;
    }
    return false;
}
"""

async def analyze_changes(all_changes: list):
    detector_logger.info(f"\n--- 分析 {len(all_changes)} 条 DOM 变化记录 ---")
    if not all_changes:
        detector_logger.info("没有检测到 DOM 变化。")
        return

    xpath_counts = Counter()
    xpath_texts = defaultdict(list)
    xpath_types = defaultdict(Counter)
    xpath_parents = defaultdict(Counter)

    for change in all_changes:
        xpath = change['xpath']
        xpath_counts[xpath] += 1
        if change['textSample']:
            xpath_texts[xpath].append(change['textSample'])
        xpath_types[xpath][change['type']] += 1

        # Analyze parent XPaths
        current_xpath_parts = xpath.split('/')
        for i in range(1, min(len(current_xpath_parts) -1, MAX_XPATH_DEPTH + 1)): # -1 to avoid full document path
            parent_xpath = "/".join(current_xpath_parts[:-i])
            if parent_xpath: # Ensure not empty
                 xpath_parents[parent_xpath][xpath] +=1 # Count how many changed children a parent has


    detector_logger.info("\n--- XPath 变化频率统计 (按变化次数排序) ---")
    # Sort by count, then by XPath string length (shorter preferred for same count)
    sorted_xpaths = sorted(xpath_counts.items(), key=lambda item: (item[1], -len(item[0])), reverse=True)

    candidate_containers = []

    for xpath, count in sorted_xpaths:
        if count < MIN_OCCURRENCES_FOR_CANDIDATE: # Filter out infrequent changes
            continue

        types_str = ", ".join([f"{t}:{c}" for t,c in xpath_types[xpath].items()])
        detector_logger.info(f"XPath: {xpath}")
        detector_logger.info(f"  发生变化次数: {count}")
        detector_logger.info(f"  变化类型: {types_str}")
        text_samples = xpath_texts.get(xpath, [])
        if text_samples:
            unique_samples = list(set(text_samples)) # Show unique samples
            detector_logger.info(f"  关联文本片段 (最多显示3个不同样本): {unique_samples[:3]}")
        candidate_containers.append(xpath) # Add to raw candidates

    detector_logger.info(f"\n--- 初步候选容器 XPath (出现次数 >= {MIN_OCCURRENCES_FOR_CANDIDATE}): ---")
    if candidate_containers:
        for i, xpath_cand in enumerate(candidate_containers[:10]): # Show top 10
            detector_logger.info(f"  {i+1}. {xpath_cand} (变化次数: {xpath_counts[xpath_cand]})")
    else:
        detector_logger.info("  未找到足够频繁变化的 XPath 作为初步候选。")


    detector_logger.info("\n--- 潜在父容器分析 (基于子元素变化频率) ---")
    # Sort parent candidates by the number of distinct children that changed under them,
    # and then by the total number of changes under them.
    sorted_parent_candidates = sorted(
        xpath_parents.items(),
        key=lambda item: (len(item[1]), sum(item[1].values())), # (num_distinct_changed_children, total_child_changes)
        reverse=True
    )

    final_suggested_containers = []
    if sorted_parent_candidates:
        detector_logger.info("最有可能是流容器的父 XPath (按其下不同变化子元素的数量和总变化量排序):")
        for parent_xpath, children_counts in sorted_parent_candidates[:5]: # Top 5 parent candidates
            num_distinct_children = len(children_counts)
            total_child_changes = sum(children_counts.values())
            detector_logger.info(f"  父 XPath: {parent_xpath}")
            detector_logger.info(f"    其下有 {num_distinct_children} 个不同子路径发生变化, 总计 {total_child_changes} 次子变化。")
            if num_distinct_children > 1 or total_child_changes >= MIN_OCCURRENCES_FOR_CANDIDATE * 1.5 : # Heuristic
                final_suggested_containers.append(parent_xpath)
    else:
        detector_logger.info("  未找到明显的父容器。")

    if not final_suggested_containers and candidate_containers:
        detector_logger.info("\n由于未找到强父容器信号，直接建议变化最频繁的元素作为容器:")
        final_suggested_containers.extend(candidate_containers[:3]) # Fallback to top direct changes

    detector_logger.info("\n--- 总结：建议的流式容器 XPath (按推断可能性排序) ---")
    if final_suggested_containers:
        for i, xpath_sugg in enumerate(list(dict.fromkeys(final_suggested_containers))[:5]): # Unique, top 5
            detector_logger.info(f"  建议 {i+1}: {xpath_sugg}")
    else:
        detector_logger.info("  未能自动推断出明确的流式容器 XPath。请检查详细的 XPath 变化频率统计，并结合 Playwright Trace 进行分析。")

    detector_logger.info("\n提示: 请将以上建议的 XPath 在浏览器开发者工具中测试，并结合 Playwright Trace (如果启用) 进行验证。")


async def run_observer(page: Page, context: BrowserContext):
    detector_logger.info(f"动态流容器探测器启动。请在浏览器中手动输入 Prompt 并提交。")
    input("完成手动提交后，请按 Enter 键开始监控 DOM 变化...")

    detector_logger.info(f"正在页面注入 MutationObserver 并开始监听 {MONITOR_DURATION_SECONDS} 秒...")
    await page.evaluate(JS_MUTATION_OBSERVER_SCRIPT, {"minTextChangeLength": MIN_TEXT_CHANGE_LENGTH})

    all_dom_changes = []
    monitor_start_time = time.time()

    try:
        while time.time() - monitor_start_time < MONITOR_DURATION_SECONDS:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            changes_batch = await page.evaluate(JS_GET_CHANGES_SCRIPT)
            if changes_batch:
                detector_logger.debug(f"获取到 {len(changes_batch)} 条 DOM 变化记录。")
                all_dom_changes.extend(changes_batch)
        detector_logger.info(f"监控时间达到 {MONITOR_DURATION_SECONDS} 秒。总共记录 {len(all_dom_changes)} 条原始变化。")

    except KeyboardInterrupt:
        detector_logger.info("用户中断监控。")
    except Exception as e:
        detector_logger.error(f"监控过程中发生错误: {e}", exc_info=True)
    finally:
        detector_logger.info("正在停止页面 MutationObserver...")
        await page.evaluate(JS_STOP_OBSERVER_SCRIPT)

    if all_dom_changes:
        await analyze_changes(all_dom_changes)
    else:
        detector_logger.info("在监控期间未记录到任何 DOM 变化。")


async def main():
    async with async_playwright() as p:
        detector_logger.info("正在启动 Firefox...")
        browser = await p.firefox.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        trace_path = None
        if ENABLE_PLAYWRIGHT_TRACING:
            TRACE_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            trace_path = TRACE_DIR / f"dynamic_detect_trace_{timestamp}.zip"
            await context.tracing.start(name="dynamic_detect", screenshots=True, snapshots=True, sources=True)
            detector_logger.info(f"Playwright Tracing 已启动。Trace 文件将保存到: {trace_path}")

        try:
            detector_logger.info(f"正在导航到: {TARGET_URL}")
            await page.goto(TARGET_URL, timeout=60000)
            detector_logger.info(f"页面加载完成。")

            await run_observer(page, context)

        except Exception as e:
            detector_logger.error(f"主流程发生错误: {e}", exc_info=True)
        finally:
            detector_logger.info("正在关闭浏览器...")
            if ENABLE_PLAYWRIGHT_TRACING and trace_path and context:
                try:
                    await context.tracing.stop(path=str(trace_path))
                    detector_logger.info(f"Playwright Trace 已保存到: {trace_path}")
                    detector_logger.info(f"你可以使用 'playwright show-trace {trace_path}' 来查看。")
                except Exception as e_trace:
                    detector_logger.error(f"保存 Playwright Trace 时出错: {e_trace}")
            
            await browser.close()
            detector_logger.info("浏览器已关闭。脚本结束。")

if __name__ == "__main__":
    asyncio.run(main())
  ```

  # pw1.py
  
```
import asyncio
import time
from pathlib import Path
from playwright.async_api import async_playwright, Page, BrowserContext
import logging
from collections import Counter, defaultdict
import re

# --- 配置区 ---
TARGET_URL = "https://www.wenxiaobai.com/"
MONITOR_DURATION_SECONDS = 30
POLL_INTERVAL_SECONDS = 0.2
MIN_TEXT_CHANGE_LENGTH = 3
MAX_XPATH_DEPTH_FOR_RELATIVE = 5
MIN_OCCURRENCES_FOR_CANDIDATE = 3
LOG_FILE_PATH = "logs/advanced_stream_detector.log"
ENABLE_PLAYWRIGHT_TRACING = True
TRACE_DIR = Path("traces/advanced_stream_detection")
MAX_ANCESTORS_TO_COLLECT = 4  # 新增：要收集的祖先层级数量

# --- 日志设置 ---
detector_logger = logging.getLogger("AdvancedStreamDetector")
detector_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
detector_logger.addHandler(ch)
Path("logs").mkdir(exist_ok=True)
fh = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
fh.setFormatter(formatter)
detector_logger.addHandler(fh)

# --- JavaScript Code to Inject ---
JS_MUTATION_OBSERVER_SCRIPT = """
async (options) => {
    if (window.llmStreamChanges) {
        if (window.llmStreamChanges.observer) window.llmStreamChanges.observer.disconnect();
        window.llmStreamChanges.records = [];
    }
    window.llmStreamChanges = {
        records: [],
        observer: null,
        options: options || {}
    };
    const getElementXPath = (element) => {
        if (!element || !element.parentNode) return null;
        if (element === document.body) return '/html/body';
        if (element === document.documentElement) return '/html';
        let ix = 0;
        const siblings = element.parentNode.childNodes;
        let pathPart = element.tagName.toLowerCase();
        for (let i = 0; i < siblings.length; i++) {
            const sibling = siblings[i];
            if (sibling === element) {
                const parentPath = getElementXPath(element.parentNode);
                if (!parentPath) return null;
                return parentPath + '/' + pathPart + '[' + (ix + 1) + ']';
            }
            if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
                ix++;
            }
        }
        return null;
    };
    const getAncestorsInfo = (element, maxLevel) => {
        const ancestors = [];
        let current = element.parentElement;
        let levels = 0;
        while (current && current !== document.body && levels < maxLevel) {
            ancestors.push({
                tag: current.tagName.toLowerCase(),
                id: current.id || '',
                classes: Array.from(current.classList || [])
            });
            current = current.parentElement;
            levels++;
        }
        return ancestors;
    };
    const observer = new MutationObserver((mutationsList) => {
        for (const mutation of mutationsList) {
            let targetElement = null;
            let changeType = mutation.type;
            let textSample = "";
            if (mutation.type === 'childList') {
                targetElement = mutation.target;
                if (mutation.addedNodes.length > 0) {
                    const addedNode = mutation.addedNodes[0];
                    if (addedNode.textContent && addedNode.textContent.trim().length > 0) {
                        textSample = addedNode.textContent.trim().substring(0, 50);
                    } else if (addedNode.nodeType === Node.ELEMENT_NODE && addedNode.innerText) {
                         textSample = addedNode.innerText.trim().substring(0,50);
                    }
                }
            } else if (mutation.type === 'characterData') {
                targetElement = mutation.target.parentElement;
                if (targetElement && mutation.target.textContent &&
                    mutation.target.textContent.trim().length >= (window.llmStreamChanges.options.minTextChangeLength || 1)) {
                    textSample = mutation.target.textContent.trim().substring(0, 50);
                } else {
                    continue; 
                }
            }
            if (targetElement) {
                const absXpath = getElementXPath(targetElement);
                if (absXpath) {
                    const ancestors = getAncestorsInfo(targetElement, window.llmStreamChanges.options.maxAncestors || 2);
                    window.llmStreamChanges.records.push({
                        absXpath: absXpath,
                        type: changeType,
                        timestamp: Date.now(),
                        textSample: textSample.replace(/\\n/g, ' '),
                        targetTagName: targetElement.tagName.toLowerCase(),
                        targetId: targetElement.id || '',
                        targetClasses: Array.from(targetElement.classList || []),
                        ancestors: ancestors  // 新增：存储祖先元素信息
                    });
                }
            }
        }
    });
    observer.observe(document.documentElement, {
        childList: true,
        subtree: true,
        characterData: true,
        characterDataOldValue: false
    });
    window.llmStreamChanges.observer = observer;
    return true;
}
"""
JS_GET_CHANGES_SCRIPT = """
() => {
    if (window.llmStreamChanges && window.llmStreamChanges.records) {
        const records = window.llmStreamChanges.records;
        window.llmStreamChanges.records = []; // Clear after fetching
        return records;
    }
    return [];
}
"""
JS_STOP_OBSERVER_SCRIPT = """
() => {
    if (window.llmStreamChanges && window.llmStreamChanges.observer) {
        window.llmStreamChanges.observer.disconnect();
        delete window.llmStreamChanges;
        return true;
    }
    return false;
}
"""

# --- Helper functions ---
def extract_class_segment(classes):
    """提取有意义的CSS类片段"""
    if not classes:
        return ""
        
    # 找到最长类名
    longest = max(classes, key=len, default="")
    if not longest:
        return ""
    
    # 提取中间部分作为标识（避免前后缀变化）
    start = len(longest) // 3
    end = (len(longest) * 2) // 3
    return longest[start:end]
    
def generate_relative_xpath_candidates(abs_xpath: str, tag_name: str, target_id: str, target_classes: list, ancestors: list) -> list:
    """生成多个可能的相对XPath候选路径"""
    candidates = []
    
    # 1. 尝试基于目标元素自身属性创建候选
    if target_id and not target_id.isnumeric():
        candidates.append(f"//{tag_name}[@id='{target_id}']")
    
    if target_classes:
        # 过滤掉无意义的类名
        meaningful_classes = [
            c for c in target_classes 
            if c and len(c) > 2 and not c.isnumeric() 
            and not re.match(r"^(js-|is-|has-|active|selected|hidden|focused|container|wrapper|base|main|content|item|block|module|component)", c, re.I)
        ]
        
        if meaningful_classes:
            # 基于单个类名
            for cls in meaningful_classes[:3]:
                candidates.append(f"//{tag_name}[contains(@class, '{cls}')]")
            
            # 基于多个类名组合
            if len(meaningful_classes) >= 2:
                candidates.append(f"//{tag_name}[contains(@class, '{meaningful_classes[0]}') and contains(@class, '{meaningful_classes[1]}')]")
            
            # 基于类名片段
            class_segment = extract_class_segment(meaningful_classes)
            if class_segment:
                candidates.append(f"//{tag_name}[contains(@class, '{class_segment}')]")
    
    # 2. 尝试基于祖先元素的id或class创建候选路径
    parts = abs_xpath.split('/')
    
    for i, ancestor_info in enumerate(ancestors):
        if i >= MAX_XPATH_DEPTH_FOR_RELATIVE: 
            break
            
        ancestor_tag = ancestor_info.get('tag', 'div')
        ancestor_id = ancestor_info.get('id', '')
        ancestor_classes = ancestor_info.get('classes', [])
        
        # 基于祖先ID创建路径
        if ancestor_id and not ancestor_id.isnumeric():
            # 计算从祖先到目标的相对层级数
            levels = i + 1
            if len(parts) > levels:
                # 获取从祖先到目标的XPath部分
                relative_part = "/".join(parts[-(levels+1):])
                candidate = f"//{ancestor_tag}[@id='{ancestor_id}']//{relative_part}"
                candidates.append(candidate)
                # 找到一个就足够好了，可以跳出循环
                break
        
        # 如果祖先没有ID，尝试使用祖先的类名
        elif ancestor_classes:
            # 过滤有意义的祖先类名
            meaningful_ancestor_classes = [
                c for c in ancestor_classes 
                if c and len(c) > 2 and not c.isnumeric() 
                and not re.match(r"^(js-|is-|has-|active|selected|hidden|focused|container|wrapper|base|main|content|item|block|module|component)", c, re.I)
            ]
            
            if meaningful_ancestor_classes:
                # 计算从祖先到目标的相对层级数
                levels = i + 1
                if len(parts) > levels:
                    # 获取从祖先到目标的XPath部分
                    relative_part = "/".join(parts[-(levels+1):])
                    # 使用多个类名组合提高准确性
                    if len(meaningful_ancestor_classes) >= 2:
                        candidate = f"//{ancestor_tag}[contains(@class, '{meaningful_ancestor_classes[0]}') and contains(@class, '{meaningful_ancestor_classes[1]}')]//{relative_part}"
                    else:
                        candidate = f"//{ancestor_tag}[contains(@class, '{meaningful_ancestor_classes[0]}')]//{relative_part}"
                    candidates.append(candidate)
    
    # 3. 如果仍然没有候选，尝试基于body和类名创建后备路径
    if not candidates and target_classes:
        meaningful_classes = [
            c for c in target_classes 
            if c and len(c) > 2 and not c.isnumeric() 
            and not re.match(r"^(js-|is-|has-|active|selected|hidden|focused|container|wrapper|base|main|content|item|block|module|component)", c, re.I)
        ]
        
        if meaningful_classes:
            candidates.append(f"//body//{tag_name}[contains(@class, '{meaningful_classes[0]}')]")
    
    # 4. 最后的备用方案：尝试使用父元素的ID或类名
    if not candidates and ancestors:
        first_ancestor = ancestors[0] if ancestors else {}
        if first_ancestor.get('id'):
            levels = 1
            if len(parts) > levels:
                relative_part = "/".join(parts[-(levels+1):])
                candidates.append(f"//{first_ancestor.get('tag')}[@id='{first_ancestor.get('id')}']//{relative_part}")
        elif first_ancestor.get('classes'):
            meaningful_classes = [
                c for c in first_ancestor.get('classes') 
                if c and len(c) > 2 and not c.isnumeric() 
                and not re.match(r"^(js-|is-|has-|active|selected|hidden|focused|container|wrapper|base|main|content|item|block|module|component)", c, re.I)
            ]
            if meaningful_classes:
                levels = 1
                if len(parts) > levels:
                    relative_part = "/".join(parts[-(levels+1):])
                    candidates.append(f"//{first_ancestor.get('tag')}[contains(@class, '{meaningful_classes[0]}')]//{relative_part}")
    
    # 去重并返回结果
    return list(dict.fromkeys(c for c in candidates if c))

async def analyze_changes(all_changes: list, current_session_id: int):
    detector_logger.info(f"\n--- 分析会话 {current_session_id} 的 {len(all_changes)} 条 DOM 变化记录 ---")
    if not all_changes:
        detector_logger.info("没有检测到 DOM 变化。")
        return {}
    
    xpath_details = defaultdict(lambda: {
        'count': 0, 'types': Counter(), 'texts': [], 'tag_name': '',
        'target_id': '', 'target_classes': [], 'relative_candidates': [],
        'ancestors': []
    })
    
    for change in all_changes:
        abs_xpath = change['absXpath']
        details = xpath_details[abs_xpath]
        details['count'] += 1
        details['types'][change['type']] += 1
        if change['textSample']: 
            details['texts'].append(change['textSample'])
        if not details['tag_name']:
            details['tag_name'] = change['targetTagName']
            details['target_id'] = change['targetId']
            details['target_classes'] = change['targetClasses']
            details['ancestors'] = change.get('ancestors', [])
            details['relative_candidates'] = generate_relative_xpath_candidates(
                abs_xpath, 
                change['targetTagName'], 
                change['targetId'], 
                change['targetClasses'],
                change.get('ancestors', [])  # 新增：传递祖先信息
            )
    
    detector_logger.info("\n--- XPath 变化频率统计 (按变化次数排序) ---")
    sorted_xpaths_items = sorted(xpath_details.items(), key=lambda item: (item[1]['count'], -len(item[0])), reverse=True)
    processed_results = {}
    
    for abs_xpath, details in sorted_xpaths_items:
        if details['count'] < MIN_OCCURRENCES_FOR_CANDIDATE: 
            continue
            
        processed_results[abs_xpath] = details
        detector_logger.info(f"绝对 XPath: {abs_xpath}")
        detector_logger.info(f"  元素标签: {details['tag_name']}, ID: '{details['target_id']}', Classes: {details['target_classes'][:5]}")
        detector_logger.info(f"  发生变化次数: {details['count']}")
        
        types_str = ", ".join([f"{t}:{c}" for t, c in details['types'].items()])
        detector_logger.info(f"  变化类型: {types_str}")
        
        text_samples = list(set(details['texts']))
        if text_samples: 
            detector_logger.info(f"  关联文本片段 (最多显示3个不同样本): {text_samples[:3]}")
        
        if details['ancestors']:
            detector_logger.info(f"  祖先元素信息:")
            for i, ancestor in enumerate(details['ancestors'][:3]):
                classes = ancestor.get('classes', [])[:3]
                classes_display = ', '.join(classes) + ('...' if len(classes) > 3 else '')
                detector_logger.info(f"    层级 {i+1}: {ancestor.get('tag', '?')}, ID: '{ancestor.get('id', '')}', Classes: [{classes_display}]")
        
        if details['relative_candidates']:
            detector_logger.info(f"  启发式相对 XPath 候选:")
            for rel_xpath in details['relative_candidates'][:3]: 
                detector_logger.info(f"    - {rel_xpath}")
        else:
            detector_logger.info("  无法生成相对 XPath 候选（元素缺少标识属性）")
        
        detector_logger.info("-" * 20)
    
    parent_xpath_counts = Counter()
    for abs_xpath in processed_results.keys():
        parts = abs_xpath.split('/')
        if len(parts) > 2:
            parent_xpath = "/".join(parts[:-1])
            parent_xpath_counts[parent_xpath] += processed_results[abs_xpath]['count']
    
    detector_logger.info("\n--- 潜在父容器 (基于其下子元素总变化量) ---")
    most_active_parents = parent_xpath_counts.most_common(5)
    if most_active_parents:
        for parent_xpath, total_child_changes in most_active_parents:
             detector_logger.info(f"  父 XPath: {parent_xpath} (子元素总变化: {total_child_changes})")
    else:
        detector_logger.info("  未找到明显活跃的父容器。")
    
    detector_logger.info(f"--- 会话 {current_session_id} 分析结束 ---\n")
    return processed_results

def analyze_aggregated_changes(all_sessions_data: dict, num_total_sessions: int):
    detector_logger.info(f"\n--- 分析所有 {num_total_sessions} 个会话的聚合 DOM 变化数据 ---")
    if not all_sessions_data:
        detector_logger.info("没有来自任何会话的分析数据。")
        return
    
    aggregated_details = defaultdict(lambda: {
        'total_count': 0, 'session_appearances': 0, 'session_counts': Counter(),
        'texts': [], 'tag_name': '', 'target_id': '', 'target_classes': [], 
        'relative_candidates': [], 'ancestors': []
    })
    
    for session_id, session_results in all_sessions_data.items():
        for abs_xpath, details in session_results.items():
            agg_detail = aggregated_details[abs_xpath]
            agg_detail['total_count'] += details['count']
            agg_detail['session_appearances'] += 1
            agg_detail['session_counts'][session_id] = details['count']
            agg_detail['texts'].extend(details['texts'])
            
            if not agg_detail['tag_name']:
                agg_detail['tag_name'] = details['tag_name']
                agg_detail['target_id'] = details['target_id']
                agg_detail['target_classes'] = details['target_classes']
                agg_detail['relative_candidates'] = details['relative_candidates']
                agg_detail['ancestors'] = details['ancestors']
    
    sorted_aggregated_xpaths = sorted(
        aggregated_details.items(),
        key=lambda item: (item[1]['session_appearances'], item[1]['total_count'], -len(item[0])),
        reverse=True
    )
    
    detector_logger.info("\n--- 跨会话 XPath 变化一致性与频率 (按推荐度排序) ---")
    final_recommendations = []
    
    for abs_xpath, details in sorted_aggregated_xpaths:
        if details['session_appearances'] < max(1, num_total_sessions // 2) and num_total_sessions > 1: 
            continue
            
        if details['total_count'] < MIN_OCCURRENCES_FOR_CANDIDATE * details['session_appearances'] * 0.3: 
            continue
            
        final_recommendations.append((abs_xpath, details))
        detector_logger.info(f"绝对 XPath: {abs_xpath}")
        detector_logger.info(f"  元素标签: {details['tag_name']}, ID: '{details['target_id']}', Classes: {details['target_classes'][:5]}")
        detector_logger.info(f"  总变化次数: {details['total_count']}")
        detector_logger.info(f"  出现在 {details['session_appearances']}/{num_total_sessions} 个会话中。")
        
        if details['ancestors']:
            detector_logger.info(f"  祖先元素信息 (来自一个会话样本):")
            for i, ancestor in enumerate(details['ancestors'][:2]):
                classes = ancestor.get('classes', [])[:3]
                classes_display = ', '.join(classes) + ('...' if len(classes) > 3 else '')
                detector_logger.info(f"    层级 {i+1}: {ancestor.get('tag', '?')}, ID: '{ancestor.get('id', '')}', Classes: [{classes_display}]")
        
        text_samples = list(set(details['texts']))
        if text_samples: 
            detector_logger.info(f"  关联文本片段 (部分样本): {text_samples[:3]}")
        
        if details['relative_candidates']:
            detector_logger.info(f"  启发式相对 XPath 候选:")
            for rel_xpath in details['relative_candidates'][:3]: 
                detector_logger.info(f"    - {rel_xpath}")
        else:
            detector_logger.info("  无法生成相对 XPath 候选（元素缺少标识属性）")
        
        detector_logger.info("-" * 30)
    
    if not final_recommendations:
        detector_logger.info("未能从多次会话中找到足够一致或频繁的流式容器 XPath。")
    else:
        detector_logger.info("\n--- 总结：高度推荐的流式容器 XPath (请重点验证以下条目) ---")
        for i, (abs_xpath, details) in enumerate(final_recommendations[:5]):
            detector_logger.info(f"  推荐 {i+1}:")
            detector_logger.info(f"    绝对 XPath: {abs_xpath}")
            
            if details['relative_candidates']:
                 detector_logger.info(f"    相对候选 (选1-2个最可能的):")
                 for rel_xpath in details['relative_candidates'][:2]: 
                     detector_logger.info(f"      - {rel_xpath}")
            else:
                detector_logger.info(f"    (未能生成简洁的相对 XPath 候选 - 尝试使用绝对路径)")
                
            detector_logger.info(f"    (总变化: {details['total_count']}, 出现会话数: {details['session_appearances']})")
    
    detector_logger.info("\n提示: 请将以上建议的 XPath 在浏览器开发者工具中测试 ($x(\"xpath\"))，并结合 Playwright Trace (如果启用) 进行验证其是否准确捕获了流式文本。")

# --- 使用非阻塞输入 ---
async def get_user_input(prompt: str) -> str:
    """Gets user input in a non-blocking way by running input() in a separate thread."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)

async def run_observer_session(page: Page, context: BrowserContext, session_id: int) -> list:
    detector_logger.info(f"\n--- 会话 {session_id} ---")
    detector_logger.info(f"请在浏览器中为会话 {session_id} 手动输入 Prompt 并提交。")
    
    # 循环直到用户按下Enter键或's'键
    while True:
        user_action_prompt = (
            f"完成会话 {session_id} 的手动提交后，请按 Enter 键开始监控 DOM 变化 "
            f"(或输入 's' 跳过此会话, 'q' 退出整个脚本): "
        )
        input_value = await get_user_input(user_action_prompt)
        input_value = input_value.strip().lower()
        if input_value == "" or input_value == 's' or input_value == 'q':
            break
        else:
            detector_logger.info("无效输入，请按 Enter, 's', 或 'q'.")

    if input_value == 's':
        detector_logger.info(f"跳过会话 {session_id}。")
        return []
    if input_value == 'q':
        detector_logger.info(f"用户请求退出脚本。")
        return "quit"  # 返回特殊值表示退出

    detector_logger.info(f"会话 {session_id}: 正在页面注入 MutationObserver 并开始监听 {MONITOR_DURATION_SECONDS} 秒...")
    await page.evaluate(JS_MUTATION_OBSERVER_SCRIPT, {
        "minTextChangeLength": MIN_TEXT_CHANGE_LENGTH,
        "maxAncestors": MAX_ANCESTORS_TO_COLLECT  # 新增：传递祖先收集层级参数
    })

    session_dom_changes = []
    monitor_start_time = time.time()

    try:
        while time.time() - monitor_start_time < MONITOR_DURATION_SECONDS:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            changes_batch = await page.evaluate(JS_GET_CHANGES_SCRIPT)
            if changes_batch:
                detector_logger.debug(f"会话 {session_id}: 获取到 {len(changes_batch)} 条 DOM 变化记录。")
                for change in changes_batch:
                    change['sessionId'] = session_id
                session_dom_changes.extend(changes_batch)
        detector_logger.info(f"会话 {session_id}: 监控时间达到。总共记录 {len(session_dom_changes)} 条原始变化。")

    except KeyboardInterrupt:
        detector_logger.info(f"会话 {session_id}: 用户中断监控。")
        raise 
    finally:
        detector_logger.info(f"会话 {session_id}: 正在停止页面 MutationObserver...")
        await page.evaluate(JS_STOP_OBSERVER_SCRIPT)
    
    return session_dom_changes

async def main():
    all_sessions_analysis_data = {}
    session_count = 0

    async with async_playwright() as p:
        detector_logger.info("正在启动 Firefox...")
        browser = await p.firefox.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        tracing_started = False
        trace_path = None
        
        # 处理Playwright Tracing
        if ENABLE_PLAYWRIGHT_TRACING:
            TRACE_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            trace_path = TRACE_DIR / f"adv_detect_multisession_trace_{timestamp}.zip"
            try:
                await context.tracing.start(name="adv_detect_multisession", screenshots=True, snapshots=True, sources=True)
                tracing_started = True
                detector_logger.info(f"Playwright Tracing 已启动 (覆盖所有会话)。Trace 文件将保存到: {trace_path}")
            except Exception as e_trace_start:
                detector_logger.error(f"启动 Playwright Tracing 时出错: {e_trace_start}")
                tracing_started = False

        try:
            detector_logger.info(f"正在导航到: {TARGET_URL}")
            await page.goto(TARGET_URL, timeout=60000)
            detector_logger.info(f"页面加载完成。")

            while True:
                session_count += 1
                try:
                    session_result = await run_observer_session(page, context, session_count)
                    
                    # 检查是否要退出
                    if session_result == "quit":
                        detector_logger.info("用户选择退出整个脚本。")
                        break
                        
                    if session_result and isinstance(session_result, list) and len(session_result) > 0:
                        session_analysis_result = await analyze_changes(session_result, session_count)
                        if session_analysis_result:
                            all_sessions_analysis_data[session_count] = session_analysis_result
                    else:
                        detector_logger.info(f"会话 {session_count} 未记录到 DOM 变化。")
                except Exception as e:
                    detector_logger.error(f"执行会话 {session_count} 时发生错误: {e}", exc_info=True)
                
                # 询问是否继续
                if session_count > 0:
                    cont_prompt = "是否要进行下一次 Prompt 观察? (y/n, q退出): "
                    cont = await get_user_input(cont_prompt)
                    cont = cont.strip().lower()
                    if cont == 'q':
                        detector_logger.info("用户选择退出。")
                        break
                    if cont != 'y':
                        break
            
            # 聚合分析所有会话
            if all_sessions_analysis_data:
                analyze_aggregated_changes(all_sessions_analysis_data, session_count)
            elif session_count > 0:
                detector_logger.info("所有会话均未记录到足够用于分析的 DOM 变化。")

        except Exception as e:
            detector_logger.error(f"主流程发生错误: {e}", exc_info=True)
        finally:
            detector_logger.info("正在关闭浏览器...")
            
            # 停止并保存Trace文件
            if tracing_started:
                try:
                    await context.tracing.stop(path=str(trace_path))
                    detector_logger.info(f"Playwright Trace 已保存到: {trace_path}")
                    detector_logger.info(f"你可以使用 'playwright show-trace {trace_path}' 来查看。")
                except Exception as e_trace_stop:
                    detector_logger.error(f"保存 Playwright Trace 时出错: {e_trace_stop}")
            
            await browser.close()
            detector_logger.info("浏览器已关闭。脚本结束。")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        detector_logger.info("\n脚本被用户强制退出 (主程序级别)。")
    except SystemExit:
        detector_logger.info("\n脚本正常退出。")
  ```

  # pytest.ini
  
```
[pytest]
asyncio_default_fixture_loop_scope = function
  ```

  # requirements.txt
  
```
fastapi==0.110.0
uvicorn[standard]==0.29.0
playwright>=1.42.0
python-dotenv>=1.0.0
PyYAML>=6.0
psutil>=5.9.0
tenacity>=8.2.3

# 测试相关的库
pytest>=8.0.0
pytest-asyncio>=0.23.0
  ```

  # test_handler.py
  
```
import asyncio
import logging
from browser_handler import LLMWebsiteAutomator
from config_manager import get_config, get_llm_site_config
from utils import setup_logging

# 直接在这里配置日志，用于本次测试
setup_logging(log_level="DEBUG")
logger = logging.getLogger("wrapper_api")

async def main_test():
    """
    一个独立的测试函数，用于验证 LLMWebsiteAutomator 的核心功能。
    """
    logger.info("--- 开始独立测试 browser_handler ---")
    
    try:
        # 1. 加载配置
        logger.info("正在加载配置...")
        # 注意：这里直接用 get_config() 可能会因为没有 lifespan 而出问题
        # 我们手动调用 load_config 来模拟
        from config_manager import load_config
        config = load_config()
        logger.info("配置加载成功。")

        # 2. 获取启用的站点配置
        enabled_sites = [s for s in config.llm_sites if s.enabled]
        if not enabled_sites:
            logger.error("在 config.yaml 中没有找到启用的站点！测试无法继续。")
            return
        
        site_config = enabled_sites[0]
        logger.info(f"将要测试的站点: {site_config.id}")

        # 3. 创建并初始化 Automator 实例
        logger.info(f"正在为 {site_config.id} 创建 LLMWebsiteAutomator 实例...")
        automator = LLMWebsiteAutomator(site_config)
        
        logger.info("正在调用 automator.initialize()...")
        await automator.initialize()
        logger.info("automator.initialize() 调用成功！Playwright 实例已启动。")

        # 4. (可选) 进行一次简单的健康检查
        is_healthy = await automator.is_healthy()
        logger.info(f"实例健康检查结果: {'健康' if is_healthy else '不健康'}")
        
        # 5. (可选) 进行一次模拟请求
        # logger.info("正在发送一个测试消息...")
        # response = await automator.send_message("你好")
        # logger.info(f"收到响应: {response[:100]}...")

    except Exception as e:
        logger.critical("独立测试过程中发生严重错误！", exc_info=True)
    finally:
        if 'automator' in locals() and automator:
            logger.info("正在清理 automator...")
            await automator.cleanup()
            logger.info("清理完成。")
    
    logger.info("--- 独立测试结束 ---")

if __name__ == "__main__":
    asyncio.run(main_test())
  ```

  # utils.py
  
```
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# 注意：我们不再从这里导入 models，以避免潜在的循环导入问题
# from models import NotificationConfig

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    为应用设置日志记录配置。
    """
    logger_instance = logging.getLogger("wrapper_api")
    
    # 如果已经有处理器，说明已配置过，直接返回，防止重复添加
    if logger_instance.hasHandlers():
        return logger_instance

    log_level_const = getattr(logging, log_level.upper(), logging.INFO)
    logger_instance.setLevel(log_level_const)
    
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger_instance.addHandler(console_handler)
    
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(detailed_formatter)
        logger_instance.addHandler(file_handler)
    
    logger_instance.propagate = False
    
    logger_instance.info(f"Logging setup completed. Level: {logging.getLevelName(log_level_const)}")
    if log_file:
        logger_instance.info(f"Log file: {log_file}")
    
    return logger_instance

def get_env_log_level() -> str:
    """
    从环境变量获取日志级别或返回默认值。
    """
    return os.getenv("LOG_LEVEL", "INFO").upper()

def notify_on_critical_error(message: str, config=None): # config 类型提示暂时移除
    """
    发送严重错误通知。
    目前只记录到 CRITICAL 级别，可以扩展为邮件/webhook。
    """
    # 直接获取已配置的 logger
    logger = logging.getLogger("wrapper_api")
    alert_message = f"CRITICAL_ALERT: {message}"
    logger.critical(alert_message)

    if config and hasattr(config, 'log_only') and config.log_only:
        pass # 已经记录
    # 未来：在此处添加邮件/webhook逻辑
  ```

