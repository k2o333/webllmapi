# 文件: browser_handler.py

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

logger = logging.getLogger("wrapper_api.browser_handler")


async def robust_click(page: Page, selector: str, timeout: int = 10000):
    """
    一个健壮的点击辅助函数，它会按顺序尝试多种方法来点击一个元素。
    简化版：移除了窗口激活逻辑，因为浏览器启动参数已处理后台节流问题。
    """
    log_prefix = f"[robust_click for '{selector[:60]}...']"
    
    # 策略 1: Playwright 标准点击 (现在应该更有效)
    try:
        logger.debug(f"{log_prefix} 尝试策略 1: 标准 page.click()")
        await page.click(selector, timeout=timeout)
        logger.info(f"{log_prefix} 标准 click 成功。")
        return
    except Exception as e:
        logger.warning(f"{log_prefix} 策略 1 (标准 click) 失败: {e}")

    # 如果策略1失败，获取元素句柄以用于后续策略
    element_handle = None
    try:
        element_handle = await page.wait_for_selector(selector, state="attached", timeout=5000)
    except Exception as find_exc:
        logger.error(f"{log_prefix} 无法定位元素，所有点击尝试中止。错误: {find_exc}")
        raise find_exc

    if not element_handle:
        raise RuntimeError(f"无法为选择器 '{selector}' 获取元素句柄")

    # 策略 2: JavaScript evaluation click
    try:
        logger.debug(f"{log_prefix} 尝试策略 2: JavaScript evaluation click")
        await page.evaluate("(element) => { if (element && typeof element.click === 'function') element.click(); }", element_handle)
        logger.info(f"{log_prefix} JavaScript evaluation click 成功。")
        return
    except Exception as e:
        logger.warning(f"{log_prefix} 策略 2 (JS evaluation) 失败: {e}")

    # 策略 3: Dispatch Event 'click'
    try:
        logger.debug(f"{log_prefix} 尝试策略 3: dispatch_event('click')")
        await element_handle.dispatch_event('click')
        logger.info(f"{log_prefix} dispatch_event('click') 成功。")
        return
    except Exception as e:
        logger.warning(f"{log_prefix} 策略 3 (dispatch_event) 失败: {e}")

    final_error_msg = f"{log_prefix} 所有点击策略均失败。"
    logger.error(final_error_msg)
    raise RuntimeError(final_error_msg)


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
        # 字段名保持不变，但内容现在是Chromium的Profile
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

            # --- 【核心修改】 ---
            logger.info(f"[{self.config.id}] Launching Chromium persistent context with options: {launch_options} and user_data_dir: {user_data_dir}")
            browser_context_obj = await playwright_obj.chromium.launch_persistent_context(
                user_data_dir,
                **launch_options
            )
            logger.info(f"[{self.config.id}] Chromium persistent context launched.")
            # --- 修改结束 ---
            
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

    # ... (后续所有代码都无需修改，和你上一版的一样) ...
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
                    await robust_click(page, selector, timeout=10000)
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
                if step.action == "click":
                    await robust_click(page, selector_str, timeout=10000)
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

        new_chat_selector = self.config.selectors.get("new_chat_button")
        if new_chat_selector:
            logger.info(f"[{self.config.id}] 尝试点击 '新对话' 按钮...")
            try:
                await robust_click(page, new_chat_selector, timeout=10000)
                logger.info(f"[{self.config.id}] '新对话' 按钮点击成功。")
                await asyncio.sleep(0.5) 
            except Exception as e:
                logger.warning(f"[{self.config.id}] 点击 '新对话' 按钮时出错 (将继续执行): {e}")
        else:
             logger.debug(f"[{self.config.id}] 配置中未定义 'new_chat_button'，跳过点击。")

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

            logger.debug(f"[{self.config.id}] Step 4 & 5: Attempting to robustly click submit button.")
            await robust_click(page, submit_selector, timeout=20000)
            logger.debug(f"[{self.config.id}] Step 4 & 5: Clicked submit button successfully.")

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
        
        stream_state = {
            '_stream_start_time': time.time(),
            'thinking_indicator_seen': False,
            '_text_stable_since': None
        }
        logger.debug(f"[{self.config.id}] Streaming from selector '{response_area_selector}', property '{text_property}'")

        try:
            logger.info(f"[{self.config.id}] 等待响应容器 '{response_area_selector}' 出现 (超时30秒)...")
            await page.wait_for_selector(response_area_selector, state="attached", timeout=30000)
            logger.info(f"[{self.config.id}] 响应容器已出现，开始流式轮询。")
        except Exception as e:
            logger.error(f"[{self.config.id}] 等待响应容器超时或失败: {e}. 流式传输可能不会开始。")

        loop_count = 0
        while True:
            try:
                loop_count += 1
                element = await page.query_selector(response_area_selector)
                current_text = ""
                if element:
                    if text_property == "textContent": current_text = await element.text_content() or ""
                    elif text_property == "innerText": current_text = await element.inner_text() or ""
                    elif text_property == "innerHTML": current_text = await element.inner_html() or ""
                    else:
                        current_text = await element.text_content() or ""
                
                if current_text != last_text:
                    new_content = current_text[len(last_text):]
                    if new_content:
                        yield new_content
                        last_text = current_text
                    stream_state['_text_stable_since'] = None
                else:
                    if stream_state['_text_stable_since'] is None:
                        stream_state['_text_stable_since'] = time.time()

                thinking_indicator_condition = next((c for c in self.config.response_handling.stream_end_conditions if c.type == "element_disappears"), None)
                if thinking_indicator_condition and not stream_state['thinking_indicator_seen']:
                    thinking_selector = self.config.selectors.get(thinking_indicator_condition.selector_key)
                    if thinking_selector and await page.is_visible(thinking_selector, timeout=50):
                        stream_state['thinking_indicator_seen'] = True

                sorted_end_conditions = sorted(
                    self.config.response_handling.stream_end_conditions,
                    key=lambda c: c.priority
                )
                should_end_stream = False
                for condition in sorted_end_conditions:
                    if await self._check_stream_end_condition(page, condition, current_text, stream_state):
                        should_end_stream = True
                        break
                
                if should_end_stream:
                    break

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
                if not stream_state.get('thinking_indicator_seen', False): return False
                selector = self.config.selectors.get(condition.selector_key)
                if not selector: return False
                result = await page.is_hidden(selector, timeout=100)
            elif condition.type == "element_appears":
                selector = self.config.selectors.get(condition.selector_key)
                if not selector: return False
                result = await page.is_visible(selector, timeout=100)
            elif condition.type == "text_stabilized":
                stable_since = stream_state.get('_text_stable_since')
                if stable_since and (time.time() - stable_since) * 1000 >= condition.stabilization_time_ms:
                    result = True
            elif condition.type == "timeout":
                start_time = stream_state.get('_stream_start_time', time.time())
                if (time.time() - start_time) >= condition.timeout_seconds:
                    result = True
        except Exception:
            result = False

        if result:
            logger.info(f"[{self.config.id}] >>> Stream END CONDITION MET: {condition.model_dump_json()}")
        return result

    async def _handle_full_response(self, page: Page) -> str:
        extraction_selector = self.config.selectors[self.config.response_handling.extraction_selector_key]
        extraction_method = self.config.response_handling.extraction_method

        if self.config.response_handling.full_text_wait_strategy:
            for strategy in self.config.response_handling.full_text_wait_strategy:
                try:
                    strat_selector = self.config.selectors.get(strategy.selector_key)
                    if not strat_selector: continue
                    
                    wait_timeout = 30000 
                    if strategy.type == "element_disappears":
                        await page.wait_for_selector(strat_selector, state="hidden", timeout=wait_timeout)
                    elif strategy.type == "element_appears":
                         await page.wait_for_selector(strat_selector, state="visible", timeout=wait_timeout)
                    elif strategy.type == "element_contains_text_or_is_not_empty":
                         await page.wait_for_function(f"() => {{ const el = document.querySelector('{strat_selector.replace("'", "\\'")}'); return el && (el.textContent || el.innerText || '').trim() !== ''; }}", timeout=wait_timeout)
                    elif strategy.type == "element_attribute_equals":
                        attribute_name = getattr(strategy, 'attribute_name', None)
                        attribute_value = getattr(strategy, 'attribute_value', None)
                        if attribute_name is None or attribute_value is None: continue
                        await page.wait_for_function(f"() => {{ const el = document.querySelector('{strat_selector.replace("'", "\\'")}'); return el && el.getAttribute('{attribute_name}') === '{attribute_value}'; }}", timeout=wait_timeout)
                except Exception:
                    pass
        
        element = await page.query_selector(extraction_selector)
        if not element:
            await asyncio.sleep(1)
            element = await page.query_selector(extraction_selector)
            if not element:
                raise ValueError(f"Response element not found: {extraction_selector} on site {self.config.id}")

        text_content = ""
        if extraction_method == "textContent": text_content = await element.text_content() or ""
        elif extraction_method == "innerText": text_content = await element.inner_text() or ""
        elif extraction_method == "innerHTML": text_content = await element.inner_html() or ""
        else:
            text_content = await element.text_content() or ""
        
        return text_content.strip()

    async def send_prompt_and_get_response(
        self, 
        prompt: str, 
        stream_callback=None, 
        target_model_variant_id: Optional[str] = None,
        return_raw_generator: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        await self._initialized_event.wait()
        start_time = time.time()
        logger.info(f"[{self.config.id}] Processing prompt (len {len(prompt)}): {prompt[:50]}... (Stream CB: {bool(stream_callback)}, Raw Gen: {return_raw_generator}, Variant: {target_model_variant_id or 'Default'})")

        try:
            response_or_generator = await self.send_message(prompt, target_model_variant_id=target_model_variant_id)
            
            if return_raw_generator:
                return response_or_generator

            if stream_callback and hasattr(response_or_generator, '__aiter__'):
                async def stream_wrapper():
                    full_response_text = ""
                    try:
                        async for chunk in response_or_generator:
                            await stream_callback(chunk)
                            full_response_text += chunk
                            yield chunk
                    except Exception as e:
                        raise
                return stream_wrapper()
            elif isinstance(response_or_generator, str):
                return response_or_generator
            elif hasattr(response_or_generator, '__aiter__'):
                full_response_text_list = []
                async for chunk in response_or_generator: full_response_text_list.append(chunk)
                return "".join(full_response_text_list)
            else:
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
            return True
        if not self.managed_browser_instance.page or self.managed_browser_instance.page.is_closed():
            return False

        page = self.managed_browser_instance.page
        hc_config = self.config.health_check
        try:
            title = await page.title()
            if not title: return False

            hc_selector = self.config.selectors.get(hc_config.check_element_selector_key)
            if not hc_selector: return False

            await page.wait_for_selector(hc_selector, state="visible", timeout=hc_config.timeout_seconds * 1000)
            
            return True
        except Exception:
            return False

    async def cleanup(self):
        logger.info(f"Cleaning up LLMWebsiteAutomator for {self.config.id}...")
        if self.managed_browser_instance:
            await self.managed_browser_instance.cleanup()
            self.managed_browser_instance = None
        if self._playwright_instance:
            try:
                await self._playwright_instance.stop()
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