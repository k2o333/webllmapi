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