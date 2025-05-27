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