# main.py

import os
import asyncio
import time
import uuid
import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Optional, AsyncGenerator, List, Any
from pathlib import Path

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
        # 在开发环境中，我们可能希望它继续运行以便调试
        # 在生产环境中，这里可能应该直接 raise e 终止应用
        pass
    
    gs = app.state.core.global_settings
    if gs and gs.idle_instance_check_interval_seconds > 0:
        logger.info("启动空闲实例监控任务...")
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
                automator = await pool.get_nowait()
                await automator.cleanup()
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

# --- Per-request logging helper ---
@asynccontextmanager
async def per_request_logging(request_id: str, site_id: str, logger: logging.Logger):
    """一个上下文管理器，用于为单个请求动态添加和移除日志文件处理器。"""
    log_dir = Path("logs") / site_id
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    request_log_file = log_dir / f"{timestamp}_{request_id}.log"
    
    handler = logging.FileHandler(request_log_file, encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # 将处理器添加到根 logger
    root_logger = logging.getLogger("wrapper_api")
    root_logger.addHandler(handler)
    logger.info(f"动态日志已启动，日志文件: {request_log_file}")
    
    try:
        yield
    finally:
        logger.info(f"动态日志已关闭，日志文件: {request_log_file}")
        handler.close()
        root_logger.removeHandler(handler)


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
                break
    logger.info("浏览器实例池初始化/更新完成。")


# --- API 端点 ---
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(key: str = Security(api_key_header)):
    expected_key = os.getenv("API_KEY")
    if not expected_key:
        return "development_key"
    if key == expected_key:
        return key
    raise HTTPException(status_code=403, detail="无效的 API 密钥")

async def _stream_generator(automator: LLMWebsiteAutomator, prompt: str, variant_id: Optional[str], model_name: str, logger: logging.Logger, request_id: str):
    response_id = f"chatcmpl-{request_id}"
    
    async with per_request_logging(request_id, automator.config.id, logger):
        try:
            raw_generator = await automator.send_prompt_and_get_response(
                prompt, 
                None, 
                variant_id, 
                return_raw_generator=True
            )

            if not hasattr(raw_generator, '__aiter__'):
                raise TypeError("Expected an async generator for streaming, but did not receive one.")

            async for chunk in raw_generator:
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
            error_payload = {
                "error": {
                    "message": f"在流式传输期间发生服务器错误: {str(e)}",
                    "type": "internal_server_error",
                    "code": None
                }
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
            yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions_endpoint(request: OpenAIChatCompletionRequest, req: Request):
    logger = req.app.state.logger
    core_state = req.app.state.core
    
    site_id, variant_id = (request.model.split("/", 1) + [None])[:2]
    request_id = str(uuid.uuid4())[:8]
    
    logger.info(f"收到请求 [ID: {request_id}] -> 站点: '{site_id}', 变体: '{variant_id or '默认'}'")

    site_config = core_state.site_configs.get(site_id)
    if not site_config:
        raise HTTPException(status_code=404, detail=f"模型 '{site_id}' 未找到或未启用。")
    pool = core_state.automator_pools.get(site_id)
    if not pool or pool.empty():
        raise HTTPException(status_code=503, detail=f"服务暂时不可用：模型 '{site_id}' 的实例池为空或不可用。")

    automator = None
    try:
        timeout = core_state.global_settings.timeout if core_state.global_settings else 60
        automator = await asyncio.wait_for(pool.get(), timeout=timeout)
        
        user_message = next((msg.content for msg in reversed(request.messages) if msg.role == 'user'), None)
        if user_message is None:
            raise HTTPException(status_code=400, detail="请求的 messages 列表中没有找到 user角色的消息。")
        prompt = user_message

        if request.stream:
            return StreamingResponse(
                _stream_generator(automator, prompt, variant_id, request.model, logger, request_id),
                media_type="text/event-stream"
            )
        else:
            async with per_request_logging(request_id, site_id, logger):
                resp_text = await automator.send_prompt_and_get_response(prompt, None, variant_id)
            return OpenAIChatCompletionResponse(
                id=f"chatcmpl-{request_id}", object="chat.completion", created=int(time.time()), model=request.model,
                choices=[OpenAIChatChoice(index=0, message=OpenAIMessage(role="assistant", content=resp_text), finish_reason="stop")],
                usage=OpenAIUsage(prompt_tokens=len(prompt.split()), completion_tokens=len(resp_text.split()), total_tokens=len(prompt.split())+len(resp_text.split()))
            )
    except asyncio.TimeoutError:
        logger.error(f"请求 [ID: {request_id}] 获取自动化器实例超时。")
        raise HTTPException(status_code=503, detail="服务暂时不可用，请稍后重试。")
    except Exception as e:
        logger.error(f"请求 [ID: {request_id}] 处理时发生未知错误: {e}", exc_info=True)
        # 归还实例，即使它可能已损坏
        if automator:
            await pool.put(automator)
            automator = None # 确保 finally 块不会再次归还
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {e}")
    finally:
        if automator:
            await pool.put(automator)