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

logger = logging.getLogger("wrapper_api.browser_handler")

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

        # --- MODIFICATION START: Click "New Chat" button ---
        new_chat_selector = self.config.selectors.get("new_chat_button")
        if new_chat_selector:
            logger.info(f"[{self.config.id}] 尝试点击 '新对话' 按钮...")
            try:
                new_chat_button = await page.query_selector(new_chat_selector)
                if new_chat_button and await new_chat_button.is_visible():
                    await new_chat_button.click(timeout=10000)
                    logger.info(f"[{self.config.id}] '新对话' 按钮点击成功。")
                    await asyncio.sleep(0.5) # Click后短暂等待UI刷新
                else:
                    logger.info(f"[{self.config.id}] '新对话' 按钮未找到或不可见，跳过点击。")
            except Exception as e:
                logger.warning(f"[{self.config.id}] 点击 '新对话' 按钮时出错 (将继续执行): {e}")
        else:
             logger.debug(f"[{self.config.id}] 配置中未定义 'new_chat_button'，跳过点击。")
        # --- MODIFICATION END ---

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
        
        stream_state = {
            '_stream_start_time': time.time(),
            'thinking_indicator_seen': False,
            '_text_stable_since': None
        }
        logger.debug(f"[{self.config.id}] Streaming from selector '{response_area_selector}', property '{text_property}'")

        # --- MODIFICATION START: Wait for response container ---
        try:
            logger.info(f"[{self.config.id}] 等待响应容器 '{response_area_selector}' 出现 (超时30秒)...")
            # 使用 'attached' 状态，表示元素已在DOM中，但不一定可见，这对于流式响应的初始阶段更可靠
            await page.wait_for_selector(response_area_selector, state="attached", timeout=30000)
            logger.info(f"[{self.config.id}] 响应容器已出现，开始流式轮询。")
        except Exception as e:
            logger.error(f"[{self.config.id}] 等待响应容器超时或失败: {e}. 流式传输可能不会开始。")
            # 即使失败，也继续执行循环，让结束条件（如超时）来处理
        # --- MODIFICATION END ---

        loop_count = 0
        while True:
            try:
                loop_count += 1
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
                else:
                    if stream_state['_text_stable_since'] is None:
                        stream_state['_text_stable_since'] = time.time()

                thinking_indicator_condition = next((c for c in self.config.response_handling.stream_end_conditions if c.type == "element_disappears"), None)
                if thinking_indicator_condition and not stream_state['thinking_indicator_seen']:
                    thinking_selector = self.config.selectors.get(thinking_indicator_condition.selector_key)
                    if thinking_selector and await page.is_visible(thinking_selector, timeout=50):
                        stream_state['thinking_indicator_seen'] = True
                        logger.info(f"[{self.config.id}] 'thinking_indicator' 已出现。现在开始监控其消失。")

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
                    logger.info(f"[{self.config.id}] Stream loop ended by condition on iteration {loop_count}.")
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
                if not condition.selector_key: return False
                selector = self.config.selectors.get(condition.selector_key)
                if not selector: return False
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
                logger.info(f"[{self.config.id}] Returning raw generator as requested.")
                return response_or_generator

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