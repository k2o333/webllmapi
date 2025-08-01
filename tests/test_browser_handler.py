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