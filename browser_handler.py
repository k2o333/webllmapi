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