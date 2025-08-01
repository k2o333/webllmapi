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