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