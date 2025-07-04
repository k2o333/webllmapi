import os
import asyncio
import time
import uuid
import json
from typing import Dict, Set, Optional, AsyncGenerator, List, Any, Tuple

from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool # For reading request body safely
from dotenv import load_dotenv

from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_client import Counter, Gauge, Histogram

from config_manager import (
    get_config,
    get_llm_site_config,
    get_enabled_llm_sites,
    LLMSiteConfig as SiteConfigModel,
    reload_config as actual_reload_config,
    AppConfig
)
from browser_handler import LLMWebsiteAutomator
from utils import setup_logging, notify_on_critical_error
from models import (
    OpenAIChatCompletionRequest as ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    OpenAIMessage,
    OpenAIChatChoice,
    OpenAIUsage,
    OpenAIChatStreamChoice,
    OpenAIChatStreamDelta,
    GlobalSettings as AppGlobalSettings,
    NotificationConfig,
    BaseModel
)

load_dotenv()
logger = setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"), log_file="logs/wrapper_api.log")

API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
EXPECTED_API_KEY = os.getenv("API_KEY")

async def get_api_key(key: str = Security(api_key_header)):
    if not EXPECTED_API_KEY:
        logger.warning("API_KEY environment variable is not set. Endpoint security is compromised.")
        raise HTTPException(status_code=500, detail="API Key not configured on server.")
    if key == EXPECTED_API_KEY: return key
    else:
        logger.warning(f"Invalid API Key received for protected endpoint.")
        raise HTTPException(status_code=403, detail="Invalid API Key")

app = FastAPI(
    title="LLM API Wrapper (Await Fix)",
    description="Wrapper API with fix for await issue in idle monitor.",
    version="1.2.2_await_fix"
)

# --- Middleware and Instrumentator are correctly placed here ---
@app.middleware("http")
async def log_full_request_middleware(request: Request, call_next):
    log_id = str(uuid.uuid4()); logger.info(f"Request IN [ID:{log_id}]")
    logger.info(f"[ID:{log_id}] Path: {request.method} {request.url.path}")
    logger.info(f"[ID:{log_id}] Client: {request.client.host}:{request.client.port}")
    headers_log = "\n".join([f"    {k}: {v}" for k, v in request.headers.items()])
    logger.info(f"[ID:{log_id}] Headers:\n{headers_log}")
    body_bytes = await request.body()
    if body_bytes:
        try:
            body_json = json.loads(body_bytes)
            pretty_body = json.dumps(body_json, indent=2, ensure_ascii=False)
            logger.info(f"[ID:{log_id}] Body (JSON):\n{pretty_body}")
        except json.JSONDecodeError:
            logger.info(f"[ID:{log_id}] Body (Raw Text):\n{body_bytes.decode(errors='ignore')}")
    else: logger.info(f"[ID:{log_id}] Body: [Empty]")
    async def receive(): return {"type": "http.request", "body": body_bytes}
    request = Request(request.scope, receive)
    response = await call_next(request)
    logger.info(f"Request OUT [ID:{log_id}] Status: {response.status_code}")
    return response

instrumentator = Instrumentator(should_group_status_codes=False, excluded_handlers=["/metrics"])
instrumentator.instrument(app).expose(app)
logger.info("Prometheus instrumentator configured and /metrics endpoint exposed.")
# --- End Middleware and Instrumentator Setup ---

# Prometheus Custom Metrics (Definitions are fine here)
automator_pool_size_configured = Gauge('automator_pool_size_configured', 'Configured size of the automator pool', ['model_id'])
automator_pool_size_active = Gauge('automator_pool_size_active', 'Current number of active/available instances in the pool', ['model_id'])
automator_requests_total = Counter('automator_requests_total', 'Total number of requests processed', ['model_id', 'status_code', 'stream'])
automator_request_duration_seconds = Histogram('automator_request_duration_seconds', 'Duration of requests processed by automators', ['model_id', 'stream'])
automator_instance_recycles_total = Counter('automator_instance_recycles_total', 'Total number of recycled automator instances', ['model_id', 'reason'])
automator_health_status_gauge = Gauge('automator_health_status', 'Health status of a model pool (1=healthy, 0.5=degraded, 0=unhealthy/empty)', ['model_id'])
config_reloads_total = Counter('config_reloads_total', 'Total number of configuration reload attempts', ['status'])

# Global State
automator_pools: Dict[str, asyncio.Queue[LLMWebsiteAutomator]] = {}
site_configs: Dict[str, SiteConfigModel] = {}
app_global_settings: Optional[AppGlobalSettings] = None
idle_monitor_task: Optional[asyncio.Task] = None
stale_automators: Dict[LLMWebsiteAutomator, float] = {}

class HealthResponse(BaseModel):
    status: str
    details: Dict[str, Any]

async def _create_new_automator_instance(site_config: SiteConfigModel) -> LLMWebsiteAutomator:
    logger.info(f"Creating new automator instance for pool: {site_config.id}")
    automator = LLMWebsiteAutomator(site_config)
    try:
        await automator.initialize()
        logger.info(f"Successfully initialized new automator for {site_config.id}")
        return automator
    except Exception as e:
        logger.error(f"Failed to initialize new automator for {site_config.id} during creation: {e}", exc_info=True)
        automator_instance_recycles_total.labels(model_id=site_config.id, reason="creation_failure").inc()
        raise

async def _cleanup_stale_automator(automator: LLMWebsiteAutomator, reason: str):
    if automator in stale_automators: del stale_automators[automator]
    try:
        model_id_label = automator.config.id if automator.config else "unknown"
        logger.info(f"Cleaning up stale/failed automator for {model_id_label} due to: {reason}")
        await automator.cleanup()
        automator_instance_recycles_total.labels(model_id=model_id_label, reason=reason).inc()
    except Exception as e:
        logger.error(f"Error during automator cleanup for {automator.config.id if automator.config else 'unknown'}: {e}")

async def monitor_idle_instances_periodically():
    if not app_global_settings:
        logger.error("Idle instance monitor cannot start: Global settings not loaded.")
        return
    logger.info("Starting idle instance monitor task...")
    await asyncio.sleep(15)
    while True:
        try:
            check_interval = app_global_settings.idle_instance_check_interval_seconds
            max_stale_life = app_global_settings.max_stale_instance_lifetime_seconds
            logger.debug(f"Idle/Stale monitor running. Next check in {check_interval} seconds.")
            await asyncio.sleep(check_interval)
            current_stale_instances = list(stale_automators.items())
            for automator, stale_since in current_stale_instances:
                if (time.time() - stale_since) > max_stale_life:
                    logger.warning(f"Stale automator for {automator.config.id} exceeded max lifetime. Forcing cleanup.")
                    await _cleanup_stale_automator(automator, "max_stale_lifetime_exceeded")
            for model_id, pool in list(automator_pools.items()):
                site_config = site_configs.get(model_id)
                if not site_config: continue
                try:
                    idle_instance = pool.get_nowait()
                except asyncio.QueueEmpty:
                    automator_pool_size_active.labels(model_id=model_id).set(0)
                    continue
                
                automator_pool_size_active.labels(model_id=model_id).set(pool.qsize() + 1)
                recycled_this_cycle = False
                try:
                    if idle_instance in stale_automators:
                        await _cleanup_stale_automator(idle_instance, "stale_and_idle")
                        recycled_this_cycle = True
                    else:
                        is_healthy = await idle_instance.is_healthy()
                        # --- FIX: ADDED AWAIT HERE ---
                        should_recycle_metrics = await idle_instance.should_recycle_based_on_metrics()

                        if not is_healthy or should_recycle_metrics:
                            reason = ("unhealthy" if not is_healthy else "")
                            if should_recycle_metrics:
                                reason = reason + ("_and_" if reason else "") + "metrics"
                            reason = reason or "unknown_idle_recycle"
                            logger.warning(f"Idle instance for {model_id} is {reason}. Recycling.")
                            await _cleanup_stale_automator(idle_instance, reason)
                            new_instance = await _create_new_automator_instance(site_config)
                            await pool.put(new_instance)
                            recycled_this_cycle = True
                except Exception as check_err:
                    logger.error(f"Error during idle instance check for {model_id}: {check_err}. Recycling.")
                    await _cleanup_stale_automator(idle_instance, "check_error")
                    try:
                        new_instance = await _create_new_automator_instance(site_config)
                        await pool.put(new_instance)
                    except Exception as create_e:
                        logger.error(f"Failed to create replacement for {model_id}: {create_e}")
                    recycled_this_cycle = True
                finally:
                    if not recycled_this_cycle:
                        try:
                            await pool.put(idle_instance)
                        except asyncio.QueueFull:
                            await _cleanup_stale_automator(idle_instance, "pool_full_on_return")
                    automator_pool_size_active.labels(model_id=model_id).set(pool.qsize())
        except asyncio.CancelledError:
            logger.info("Idle/Stale instance monitor task cancelled.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in idle/stale monitor: {e}", exc_info=True)
            await asyncio.sleep(60)

async def _initialize_pools(config_to_load: AppConfig):
    # ... (This function is correct and remains unchanged) ...
    global app_global_settings, automator_pools, site_configs, stale_automators
    app_global_settings = config_to_load.global_settings
    new_site_configs_map = {site.id: site for site in config_to_load.llm_sites if site.enabled}
    current_pool_ids = set(automator_pools.keys())
    new_config_ids = set(new_site_configs_map.keys())
    for model_id in current_pool_ids - new_config_ids:
        if model_id in automator_pools:
            pool = automator_pools.pop(model_id)
            while not pool.empty():
                try: stale_automators[pool.get_nowait()] = time.time()
                except asyncio.QueueEmpty: break
            if model_id in site_configs: del site_configs[model_id]
            automator_pool_size_configured.labels(model_id=model_id).set(0)
            automator_pool_size_active.labels(model_id=model_id).set(0)
    for model_id, new_site_cfg in new_site_configs_map.items():
        site_configs[model_id] = new_site_cfg
        automator_pool_size_configured.labels(model_id=model_id).set(new_site_cfg.pool_size)
        if model_id not in automator_pools:
            automator_pools[model_id] = asyncio.Queue(maxsize=new_site_cfg.pool_size)
        pool = automator_pools[model_id]
        current_instances_in_pool_list = []
        while not pool.empty():
            try: current_instances_in_pool_list.append(pool.get_nowait())
            except asyncio.QueueEmpty: break
        num_to_have = new_site_cfg.pool_size
        num_current = len(current_instances_in_pool_list)
        if num_current > num_to_have:
            for i in range(num_current - num_to_have):
                stale_automators[current_instances_in_pool_list.pop()] = time.time()
        for inst in current_instances_in_pool_list: await pool.put(inst)
        num_to_add = num_to_have - pool.qsize()
        if num_to_add > 0:
            for _ in range(num_to_add):
                if pool.full(): break
                try: await pool.put(await _create_new_automator_instance(new_site_cfg))
                except Exception as e: logger.error(f"Failed to add new instance to pool {model_id}: {e}")
        automator_pool_size_active.labels(model_id=model_id).set(pool.qsize())
    for model_id_to_cleanup in list(site_configs.keys()):
        if model_id_to_cleanup not in new_site_configs_map:
            if model_id_to_cleanup in site_configs: del site_configs[model_id_to_cleanup]

@app.on_event("startup")
async def startup_event():
    # ... (This function is correct and remains unchanged) ...
    global idle_monitor_task
    logger.info("Starting LLM API Wrapper application")
    try:
        initial_config = get_config()
        await _initialize_pools(initial_config)
        logger.info(f"Initialized {len(automator_pools)} automator pools based on initial config.")
    except Exception as e:
        logger.critical(f"FATAL: Failed to initialize pools during startup: {e}", exc_info=True)
    if app_global_settings and app_global_settings.idle_instance_check_interval_seconds > 0:
        idle_monitor_task = asyncio.create_task(monitor_idle_instances_periodically())
    else:
        logger.info("Idle instance monitor is disabled or global settings not loaded properly.")

@app.on_event("shutdown")
async def shutdown_event():
    # ... (This function is correct and remains unchanged) ...
    logger.info("Shutting down LLM API Wrapper application")
    if idle_monitor_task and not idle_monitor_task.done():
        idle_monitor_task.cancel()
        try: await idle_monitor_task
        except asyncio.CancelledError: logger.info("Idle monitor task cancelled.")
    for automator, stale_since in list(stale_automators.items()):
        await _cleanup_stale_automator(automator, "shutdown")
    stale_automators.clear()
    for model_id, pool in automator_pools.items():
        while not pool.empty():
            try: await pool.get_nowait().cleanup()
            except asyncio.QueueEmpty: break
            except Exception as e: logger.error(f"Error cleaning automator from {model_id} on shutdown: {e}")
    automator_pools.clear(); site_configs.clear()
    logger.info("All automators cleaned up.")

@app.post("/v1/chat/completions")
async def chat_completions_endpoint(request: ChatCompletionRequest):
    # ... (This function is correct and remains unchanged) ...
    requested_model_full_id = request.model
    site_id_to_find: str
    target_model_variant_id: Optional[str] = None
    if "/" in requested_model_full_id:
        parts = requested_model_full_id.split("/", 1)
        site_id_to_find = parts[0]
        target_model_variant_id = parts[1]
    else:
        site_id_to_find = requested_model_full_id
    model_id_metric_label = site_id_to_find
    status_code_label = "200"
    is_stream_label = "true" if request.stream else "false"
    request_start_time = time.time()
    try:
        site_config = site_configs.get(site_id_to_find)
        if not site_config:
            status_code_label = "400"
            raise HTTPException(status_code=400, detail=f"Site '{site_id_to_find}' not found/disabled.")
        if target_model_variant_id:
            if not site_config.model_variants or not any(v.id == target_model_variant_id for v in site_config.model_variants):
                status_code_label = "400"
                raise HTTPException(status_code=400, detail=f"Variant '{target_model_variant_id}' not for site '{site_id_to_find}'.")
        pool = automator_pools.get(site_id_to_find)
        if not pool:
            status_code_label = "500"
            raise HTTPException(status_code=500, detail="Internal error: Model pool unavailable.")
        logger.info(f"Req for {site_id_to_find}. Pool: {pool.qsize()}/{site_config.pool_size}. Waiting...")
        automator_instance: Optional[LLMWebsiteAutomator] = None
        try:
            timeout_val = (app_global_settings.timeout if app_global_settings else 30) + 5
            automator_instance = await asyncio.wait_for(pool.get(), timeout=timeout_val)
            logger.info(f"Instance acquired for {site_id_to_find}. Processing...")
            prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
            if request.stream:
                async def generate_stream() -> AsyncGenerator[str, None]:
                    stream_start_time = time.time(); full_completion_text = ""; response_id = f"chatcmpl-{uuid.uuid4()}"
                    created_time = int(time.time()); sent_role_once = False
                    try:
                        async for chunk in automator_instance.send_prompt_and_get_response(
                            prompt, stream_callback=None, target_model_variant_id=target_model_variant_id
                        ):
                            full_completion_text += chunk
                            delta = OpenAIChatStreamDelta(content=chunk)
                            if not sent_role_once: delta.role = "assistant"; sent_role_once = True
                            choice = OpenAIChatStreamChoice(index=0, delta=delta, finish_reason=None)
                            resp_chunk = ChatCompletionStreamResponse(id=response_id, created=created_time, model=requested_model_full_id, choices=[choice])
                            yield f"data: {resp_chunk.model_dump_json(exclude_none=True)}\n\n"
                        final_choice = OpenAIChatStreamChoice(index=0, delta=OpenAIChatStreamDelta(), finish_reason="stop")
                        final_resp_chunk = ChatCompletionStreamResponse(id=response_id, created=created_time, model=requested_model_full_id, choices=[final_choice])
                        yield f"data: {final_resp_chunk.model_dump_json(exclude_none=True)}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as stream_err:
                        logger.error(f"Stream error for {site_id_to_find} (var: {target_model_variant_id}): {stream_err}", exc_info=True)
                        yield f"data: {json.dumps({'error': {'message': str(stream_err), 'type': 'stream_error'}})}\n\n"
                        yield "data: [DONE]\n\n"
                    finally:
                        logger.info(f"Stream for {site_id_to_find} (var: {target_model_variant_id}) done. Dur: {time.time()-stream_start_time:.2f}s, Len: {len(full_completion_text)}.")
                return StreamingResponse(generate_stream(), media_type="text/event-stream")
            else:
                non_stream_start = time.time()
                response_text = await automator_instance.send_prompt_and_get_response(
                    prompt, stream_callback=None, target_model_variant_id=target_model_variant_id
                )
                logger.info(f"Non-stream for {site_id_to_find} (var: {target_model_variant_id}) done. Dur: {time.time()-non_stream_start:.2f}s, Len: {len(response_text)}.")
                return ChatCompletionResponse(
                    id=f"chatcmpl-{uuid.uuid4()}", object="chat.completion", created=int(time.time()), model=requested_model_full_id,
                    choices=[OpenAIChatChoice(index=0, message=OpenAIMessage(role="assistant", content=response_text), finish_reason="stop")],
                    usage=OpenAIUsage(prompt_tokens=len(prompt.split()), completion_tokens=len(response_text.split()), total_tokens=len(prompt.split())+len(response_text.split()))
                )
        except asyncio.TimeoutError:
            status_code_label = "503"
            automator_instance_recycles_total.labels(model_id=model_id_metric_label, reason="timeout_acquiring_instance").inc()
            raise HTTPException(status_code=503, detail="Service Unavailable: No instances for model.")
        finally:
            if automator_instance:
                should_recycle = (site_config.max_requests_per_instance > 0 and
                                  automator_instance.get_request_count() >= site_config.max_requests_per_instance)
                if should_recycle:
                    logger.info(f"Recycling instance for {site_id_to_find} due to max_requests.")
                    await _cleanup_stale_automator(automator_instance, "max_requests")
                    try: await pool.put(await _create_new_automator_instance(site_config))
                    except Exception as create_e: logger.error(f"Failed to create replacement for {site_id_to_find}: {create_e}")
                else:
                    try: await pool.put(automator_instance)
                    except asyncio.QueueFull: await _cleanup_stale_automator(automator_instance, "pool_full_on_return_active")
                automator_pool_size_active.labels(model_id=model_id_metric_label).set(pool.qsize())
    except HTTPException as http_exc: status_code_label = str(http_exc.status_code); raise
    except Exception as e:
        status_code_label = "500"
        logger.error(f"Error for {site_id_to_find} (var: {target_model_variant_id}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        automator_requests_total.labels(model_id=model_id_metric_label, status_code=status_code_label, stream=is_stream_label).inc()
        automator_request_duration_seconds.labels(model_id=model_id_metric_label, stream=is_stream_label).observe(time.time() - request_start_time)

@app.post("/reload_config", status_code=200)
async def reload_configuration_endpoint(api_key: str = Depends(get_api_key)):
    # ... (This function is correct and remains unchanged) ...
    logger.info(f"Received request to reload configuration. API Key validated.")
    try:
        new_config_data = actual_reload_config()
        global idle_monitor_task
        if idle_monitor_task and not idle_monitor_task.done():
            idle_monitor_task.cancel(); await idle_monitor_task; idle_monitor_task = None
        await _initialize_pools(new_config_data)
        if app_global_settings and app_global_settings.idle_instance_check_interval_seconds > 0:
            idle_monitor_task = asyncio.create_task(monitor_idle_instances_periodically())
        logger.info("Configuration reloaded and pools updated.")
        config_reloads_total.labels(status="success").inc()
        return {"message": "Configuration reloaded successfully."}
    except FileNotFoundError: config_reloads_total.labels(status="failure_file_not_found").inc(); raise HTTPException(status_code=500, detail="config.yaml not found.")
    except ValueError as ve: config_reloads_total.labels(status="failure_validation_error").inc(); raise HTTPException(status_code=400, detail=f"Invalid config: {ve}")
    except Exception as e: config_reloads_total.labels(status="failure_unexpected").inc(); raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check_endpoint():
    # ... (This function is correct and remains unchanged) ...
    global_health = 1.0; details: Dict[str, Any] = {"pools": {}, "stale_count": len(stale_automators)}
    if not app_global_settings: automator_health_status_gauge.labels(model_id="overall_service").set(0.0); return HealthResponse(status="unhealthy", details={"error": "Global settings not loaded."})
    active_ids = set(site_configs.keys())
    for model_id in active_ids:
        pool = automator_pools.get(model_id); site_cfg = site_configs.get(model_id)
        p_detail: Dict[str, Any] = {"cfg_size": site_cfg.pool_size if site_cfg else "N/A", "q_size": pool.qsize() if pool else 0, "status": "unknown", "sample_health": "N/A"}
        m_health = 0.0
        if not pool or not site_cfg: p_detail["status"] = "error_cfg_pool"; global_health = 0.0
        elif site_cfg.pool_size == 0: p_detail["status"] = "ok_zero_config"; m_health = 1.0
        elif pool.empty(): p_detail["status"] = "empty_warn"; global_health = min(global_health, 0.5); m_health = 0.0
        else:
            tmp_inst: Optional[LLMWebsiteAutomator] = None
            try:
                tmp_inst = pool.get_nowait()
                is_inst_h = await tmp_inst.is_healthy()
                p_detail["sample_health"] = "healthy" if is_inst_h else "unhealthy"
                if not is_inst_h: p_detail["status"] = "degraded_one_bad"; global_health = min(global_health, 0.5); m_health = 0.5
                else: p_detail["status"] = "healthy_min_one_ok"; m_health = 1.0
            except asyncio.QueueEmpty: p_detail["status"] = "empty_on_check"; global_health = min(global_health, 0.5); m_health = 0.0
            except Exception as e: p_detail["status"] = "error_checking"; p_detail["err_msg"] = str(e); global_health = 0.0; m_health = 0.0
            finally:
                if tmp_inst:
                    try: await pool.put(tmp_inst)
                    except asyncio.QueueFull: await _cleanup_stale_automator(tmp_inst, "pool_full_on_health_return")
        details["pools"][model_id] = p_detail
        automator_health_status_gauge.labels(model_id=model_id).set(m_health)
        automator_pool_size_active.labels(model_id=model_id).set(pool.qsize() if pool else 0)
        automator_pool_size_configured.labels(model_id=model_id).set(site_cfg.pool_size if site_cfg else 0)
    overall_status_str = "healthy" if global_health == 1.0 else ("degraded" if global_health == 0.5 else "unhealthy")
    if not active_ids and get_enabled_llm_sites(): overall_status_str = "unhealthy"; details["error"] = "No pools initialized despite config."; global_health = 0.0
    elif not get_enabled_llm_sites(): overall_status_str = "healthy"; details["info"] = "No sites enabled."; global_health = 1.0
    automator_health_status_gauge.labels(model_id="overall_service").set(global_health)
    return HealthResponse(status=overall_status_str, details=details)


if __name__ == "__main__":
    import uvicorn
    # Use reload=True for development to see changes instantly
    uvicorn.run("main:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")), reload=True)