
**开发文档：LLM 网页自动化 OpenAI API 适配器 (Windows 本地版)**

**版本：** 1.2


目录：

1.  项目概述与目标
2.  核心原则
3.  技术栈 (Windows 环境)
4.  项目文件结构
5.  核心模块设计
    *   5.1 配置管理 (config\_manager.py & config.yaml)
    *   5.2 浏览器自动化处理器 (browser\_handler.py)
    *   5.3 API 服务 (main.py)
    *   5.4 数据模型 (models.py)
    *   5.5 工具与日志 (utils.py)
6.  分阶段实施计划 (Windows 环境优化)
    *   阶段 0: 环境搭建与基础框架 (Windows 重点)
    *   阶段 1: 单一网站核心自动化 (非流式) 与基础测试
    *   阶段 2: FastAPI 接口封装 (非流式) 与初步并发控制
    *   阶段 3: 实现流式响应与性能考量
    *   阶段 4: 浏览器实例池与资源监控
    *   阶段 5: 鲁棒性、高级功能与可维护性提升
7.  规范与最佳实践
    *   7.1 日志规范与管理
    *   7.2 错误处理与快照
    *   7.3 安全注意事项 (Windows 本地)
    *   7.4 测试策略与节奏
    *   7.5 依赖管理
8.  部署与运行 (Windows 本地)

## 1. 项目概述与目标

本项目旨在通过 Firefox 浏览器自动化技术，在 **Windows 本地环境**下模拟用户访问各种 LLM 网站的操作，并将这些操作封装成符合 OpenAI Chat Completions API 规范 (/v1/chat/completions) 的本地服务。

**主要目标：**

*   **统一接口：** 为不同的 LLM 网站提供统一的、类 OpenAI 的 API 调用方式。
*   **配置驱动：** 允许用户通过配置文件灵活定义目标网站、CSS 选择器、交互逻辑等。
*   **Windows 本地运行：** 所有操作在用户 Windows 本地环境中执行，确保数据隐私和对本地部署模型的支持。
*   **鲁棒性与可维护性：** 构建稳定、易于调试和扩展的系统。
*   **流式响应支持：** 尽可能支持 LLM 的流式输出，提升用户体验。

## 2. 核心原则

*   **配置驱动：** 所有网站特定信息通过 `config.yaml` 文件管理。
*   **模块化：** 清晰分离浏览器自动化、API 接口、配置管理和工具函数。
*   **日志先行：** 详尽的日志记录，便于调试、监控和问题追溯。
*   **状态同步：** 在与网页交互前，检查并（如果可能）自动调整网页上的配置项以匹配 API 请求。
*   **鲁棒设计：** 包含错误处理、重试机制、失败快照等，提升系统稳定性。
*   **迭代测试：** 每个阶段都伴随相应的测试，确保代码质量。

## 3. 技术栈 (Windows 环境)

*   **Python 3.8+** (确保在 Windows 上正确安装并配置 PATH)
*   **Playwright (with Firefox driver)**
    *   **playwright-stealth**
*   **FastAPI**
*   **Uvicorn**
*   **Pydantic**
*   **PyYAML**
*   **python-dotenv**
*   **psutil**: 用于系统资源监控。
*   **Tenacity**: 实现重试逻辑。

## 4. 项目文件结构

```
llm_api_adapter/
├── main.py                 # FastAPI 应用主入口
├── browser_handler.py      # 浏览器自动化核心逻辑 (LLMWebsiteAutomator 类)
├── config_manager.py       # 加载和管理 config.yaml 的逻辑
├── models.py               # Pydantic 模型 (OpenAI API 及内部数据结构)
├── utils.py                # 通用工具函数 (日志配置, 通知等)
├── config.yaml             # LLM 网站详细配置
├── .env                    # 环境变量 (CONFIG_FILE_PATH, LOG_LEVEL等)
├── .env.example            # .env 文件示例
├── requirements.txt        # 项目依赖
├── logs/                   # 日志文件存放目录 (通过 .gitignore 排除)
├── error_snapshots/        # 错误发生时的快照 (截图, DOM)
└── profiles/               # Firefox Profile 存放目录 (Windows 绝对路径或相对于项目根目录的路径)
    └── site_A_profile/
└── tests/                  # 单元测试和集成测试
    ├── conftest.py
    ├── unit/
    └── integration/
```

## 5. 核心模块设计

### 5.1 配置管理 (`config_manager.py` & `config.yaml`)

*   **`config.yaml`**: 核心配置文件，定义全局设置和每个 LLM 站点的详细参数。
    *   **结构示例 (重点部分)：**
        ```yaml
# config.yaml

# 全局设置，适用于所有站点，除非被站点特定配置覆盖
global_settings:
  # Playwright 相关默认超时 (毫秒)
  default_page_load_timeout_ms: 60000   # 页面加载超时
  default_action_timeout_ms: 30000      # 点击、填充等操作的超时
  default_navigation_timeout_ms: 60000 # 导航操作超时
  
  # Firefox Profile 相关
  default_firefox_profile_base_path: "./profiles" # Profile的相对路径基准 (相对于项目根目录)
                                                # 也可使用Windows绝对路径，例如: "C:/Users/YourUser/AppData/Roaming/Mozilla/Firefox/ProfilesAdapter"

  # 日志与快照
  log_level: "INFO"  # 应用日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL) - 可被.env覆盖
  error_snapshot_dir: "./error_snapshots" # 错误快照保存目录 (相对路径或绝对路径)
  
  # 资源管理与健康检查
  max_stale_instance_lifetime_seconds: 300 # 热更新后，旧配置实例的最大存活时间（秒）
  default_max_requests_per_instance: 100   # 每个浏览器实例处理多少请求后重启
  default_max_memory_per_instance_mb: 1024 # 单个Firefox实例允许的最大内存占用 (MB)，用于健康检查
  default_instance_health_check_interval_seconds: 60 # 后台健康检查空闲实例的频率（秒）

# LLM 站点配置列表
llm_sites:
  - id: "my-local-chatgpt-clone" # API 调用时使用的 model 名称 (必须唯一)
    name: "My Local ChatGPT Clone Interface" # 人类可读名称
    enabled: true # 是否启用此站点配置
    url: "http://localhost:3000/chat/custom_model_A" # LLM 网站的完整 URL

    # Firefox Profile 配置
    firefox_profile_dir: "local_chatgpt_clone_profile" # Profile目录名 (相对于 global_settings.default_firefox_profile_base_path)
                                                      # 或 Windows 绝对路径，例如: "D:/MyFirefoxProfiles/chatgpt_clone"

    # Playwright 浏览器启动选项
    playwright_launch_options:
      headless: false # true 为无头模式, false 为有头模式 (调试时建议false)
      # user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0" # 可选，自定义UA
      # viewport: { width: 1920, height: 1080 } # 可选，设置浏览器视口大小
      # slow_mo: 50 # 可选，减慢 Playwright 操作速度 (毫秒)，用于调试
    use_stealth: true # 是否启用 playwright-stealth 规避检测

    # 浏览器实例池大小 (针对此模型)
    pool_size: 2 # 此模型允许同时运行的浏览器实例数量

    # 实例资源限制 (覆盖全局设置)
    max_requests_per_instance: 150
    max_memory_per_instance_mb: 1536

    # CSS 选择器 (必须准确)
    selectors:
      input_area: "textarea#prompt-input-field"
      submit_button: "button[data-testid='send-message-button']"
      response_container: "div.chat-messages-container" # 包含所有消息的父容器
      last_response_message: "div.chat-messages-container > div.message-bubble[data-role='assistant']:last-child" # 精确获取最后一条助手消息
      streaming_response_target: "div.chat-messages-container > div.message-bubble[data-role='assistant']:last-child > div.content" # 流式文本更新的目标元素
      thinking_indicator: "div.spinner-overlay.active" # "思考中"指示器
      # 用于页面选项同步的选择器
      model_selector_dropdown: "select#llm-model-selector"
      theme_toggle_dark: "button#theme-toggle[aria-label='Switch to dark theme']"
      # ... 其他特定于此站点的元素选择器 ...
    
    # 备用选择器 (当主选择器失败时，按顺序尝试)
    backup_selectors:
      input_area: 
        - "textarea[name='user_prompt']"
        - "div.input-wrapper > textarea"
      submit_button:
        - "button.submit-chat"

    # 页面选项同步配置 (在发送 prompt 前执行)
    options_to_sync:
      - id: "model_selection_on_page" # 内部操作标识
        description: "Ensure the 'SuperChat-v3' model is selected on the page."
        selector_key: "model_selector_dropdown" # 引用上面 selectors 中的键名
        action: "select_by_value" # 操作类型: 'select_by_value', 'select_by_label', 'ensure_checked', 'ensure_unchecked', 'click'
        target_value_on_page: "superchat-v3-turbo" # 对于 select_by_value, 这是 option 的 value 属性
                                                    # 对于 ensure_checked/unchecked, 此字段通常不用, 检查元素checked状态
        on_failure: "abort" # 同步失败时的行为: "skip" (跳过), "warn_and_skip" (警告并跳过), "abort" (中断API请求)
      
      - id: "dark_theme_activation"
        description: "Ensure dark theme is active."
        selector_key: "theme_toggle_dark" # 假设这是一个需要点击才能切换到暗色模式的按钮 (如果它当前不是暗色)
        action: "click_if_not_active" # 假设这是一个自定义的或需要特定逻辑判断的action
                                      # 或者更通用的: 'click' (总是点击), 'ensure_attribute_value'
        # target_attribute: "aria-pressed" # 假设用此属性判断是否已激活
        # target_attribute_value: "true"
        # 如果不用自定义action，可能需要多个步骤或更复杂的选择器逻辑
        on_failure: "warn_and_skip"

    # 响应处理配置
    response_handling:
      type: "stream" # 响应类型: "stream" (流式) 或 "full_text" (等待完整文本)
      stream_poll_interval_ms: 150 # 流式响应时 DOM 轮询的间隔时间（毫秒）
      stream_text_property: "textContent" # 从流式目标元素提取文本时使用的属性: "textContent", "innerText", "innerHTML"
      
      # 流式响应结束条件 (满足任一即可，按 priority 从低到高评估，0为最高)
      stream_end_conditions:
        - type: "element_disappears" # 类型: "element_disappears", "element_appears", "text_stabilized", "timeout"
          selector_key: "thinking_indicator" # 引用 selectors 中的键名
          priority: 0 # 最高优先级
        - type: "text_stabilized" # 文本在 stabilization_time_ms 内无变化
          stabilization_time_ms: 2500
          priority: 1
        - type: "element_appears" # 例如“停止生成”按钮出现
          selector_key: "stop_generation_button" # 假设 selectors 中定义了此键
          priority: 0 # 与 thinking_indicator 消失同级
        - type: "timeout" # 流式响应的全局超时
          timeout_seconds: 180 # (秒)
          priority: 2 # 最低优先级，作为最终保障
      
      # 如果 response_handling.type 为 "full_text"，则使用此策略等待响应完成
      full_text_wait_strategy: # 按顺序检查，满足一个即可 (或者可以设计成 all_must_be_true)
        - type: "element_disappears"
          selector_key: "thinking_indicator"
        - type: "element_contains_text_or_is_not_empty" # 确保响应元素不再是初始空状态或加载状态
          selector_key: "last_response_message"
      
      # 从哪个元素提取最终的完整响应文本 (流式和非流式都可能用到)
      extraction_selector_key: "last_response_message"
      # 提取文本时使用的方法
      extraction_method: "innerText" # "innerText", "textContent", "innerHTML"

    # 站点健康检查配置 (用于后台定期检查和 /health API)
    health_check:
      enabled: true # 是否对此站点进行后台健康检查
      interval_seconds: 300 # 后台定期检查的频率（秒）
      timeout_seconds: 45   # 单次健康检查操作的超时（秒）
      failure_threshold: 3  # 连续失败多少次后认为此站点实例不健康
      check_element_selector_key: "input_area" # 健康检查时，验证此元素是否存在且可见

  # --- 可以添加更多站点配置 ---
  - id: "another-ollama-frontend"
    name: "Another Ollama WebUI"
    enabled: false # 此站点当前禁用
    url: "http://localhost:8080"
    firefox_profile_dir: "ollama_frontend_profile"
    playwright_launch_options:
      headless: true
    use_stealth: false
    pool_size: 1
    selectors:
      input_area: "textarea[placeholder*='Send a message']"
      submit_button: "button:has-text('Send')"
      response_container: "main" # 简化示例
      last_response_message: "article:last-of-type div[data-message-author-role='assistant']"
      thinking_indicator: "button[aria-label='Stop generating']" # 等待它出现然后消失，或者它本身就是思考的标志
    # ... 其他配置参考上面的例子 ...
    response_handling:
      type: "full_text"
      stream_poll_interval_ms: 200 # 即使是full_text，也可以保留，只是不用于流式回调
      full_text_wait_strategy:
        - type: "element_attribute_equals"
          selector_key: "thinking_indicator" # 假设这个按钮在生成时有特定状态
          attribute_name: "disabled" # 例如，生成时按钮是 disabled
          attribute_value: "false" # 等待它变为 non-disabled (即 false)
      extraction_selector_key: "last_response_message"
      extraction_method: "textContent"
    health_check:
      enabled: true
      check_element_selector_key: "input_area"
        ```
    *   **`global_settings.default_firefox_profile_base_path`**: 使用相对于项目根目录的路径 (e.g., `"./profiles"`) 或 Windows 绝对路径。
    *   **配置热更新策略**:
        1.  当 `/reload_config` API 端点被调用时，`config_manager` 加载并验证新的 `config.yaml`。
        2.  `main.py` 模块在确认新配置有效后，将当前活动的浏览器实例池标记为“待弃用”状态。
        3.  所有新的 API 请求将从基于新配置动态创建的新实例池中获取浏览器实例。
        4.  待弃用池中的实例在完成其当前正在处理的请求并被归还后，将被立即关闭，并不再加入任何池中。
        5.  设置一个固定的“最大孤立实例运行时长”（例如，在 `global_settings` 中配置 `max_stale_instance_lifetime_seconds: 300`），如果旧配置下的实例在此时间内未完成任务，将被强制关闭以回收资源。

*   **`config_manager.py`**:
    *   `load_config(config_path: str, force_reload: bool = False) -> AppConfig`: 解析 YAML，使用 Pydantic 模型 (`AppConfig` 包含 `GlobalSettings` 和 `List[LLMSiteConfig]`) 进行校验和结构化。支持缓存和强制重载。(实现时需支持热重载通知机制，以便 `main.py` 响应)
    *   `get_config() -> AppConfig`: 获取已加载的配置。
    *   `get_site_config(model_id: str) -> Optional[LLMSiteConfig]`: 根据模型ID获取特定站点配置。

### 5.2 浏览器自动化处理器 (`browser_handler.py`)

*   **`LLMWebsiteAutomator` 类**: 封装单个 LLM 网站的所有 Playwright 交互。
    *   `__init__(self, site_config: LLMSiteConfig, global_settings: GlobalSettings)`: 初始化，存储配置。
    *   `_launch_browser_if_needed(self)`:
        *   惰性启动或连接到 Firefox `PersistentContext` (使用 `user_data_dir`)。
        *   应用 `playwright-stealth`。
        *   设置视口、User-Agent 等 (根据配置)。
        *   导航到目标 URL。
    *   `_get_selector(self, selector_key: str) -> str`: 辅助函数，获取主选择器，如果失败则尝试备用选择器。
    *   `_perform_action(self, action: Callable, selector_key: str, *args, **kwargs)`: 封装 Playwright 操作（点击、填充等），包含超时、重试（使用 Tenacity）、失败快照逻辑。
    *   `_ensure_page_options(self)`: 根据 `site_config.options_to_sync` 检查并同步页面上的选项（模型选择、模式开关等）。处理同步失败的情况 (skip/abort)。
    *   `send_prompt_and_get_response(self, prompt_text: str, stream_callback: Optional[Callable[[str], None]] = None) -> Optional[str]`:
        *   核心方法，处理发送提示到获取响应的整个流程。
        *   调用 `_launch_browser_if_needed`, `_ensure_page_options`。
        *   使用 `_perform_action` 发送提示。
        *   根据 `response_handling.type` (stream/full\_text) 调用不同的等待和提取逻辑：
            *   **流式 (`stream_callback` 提供时)**: 监控 `streaming_response_target` 的文本变化（使用 `stream_text_property`），通过 `stream_callback` 回传增量文本。根据 `stream_end_conditions` (带优先级) 判断结束。
            *   **完整文本**: 等待 `full_text_wait_strategy` 条件满足，然后从 `extraction_selector_key` 提取文本。
    *   **流式响应结束条件优先级 (`stream_end_conditions`)**:
        *   在 `send_prompt_and_get_response` 方法的流式处理部分，轮询检查结束条件时，严格按照 `config.yaml` 中为每个条件定义的 `priority` 整数值进行（例如，0 代表最高优先级，数值越大优先级越低）。
        *   在每一轮检查中，从最高优先级的条件开始评估。一旦任何一个条件满足，即判定流已结束，并停止对其他（包括同级和更低优先级）条件的检查。
    *   **健康检查 (`is_healthy`)**:
        *   检查 `self.page.is_closed()` 返回 `False`。
        *   检查 `config.yaml` 中 `health_check.check_element_selector_key` 指定的元素是否存在且可见。
        *   执行一次 `self.page.title()` 调用并验证其是否返回非空字符串，以初步判断页面 JavaScript 环境是否崩溃。
        *   记录完整的健康检查步骤及结果到 DEBUG 日志。
    *   `close(self)`: 安全关闭浏览器和 Playwright 上下文。

### 5.3 API 服务 (`main.py`)

*   **FastAPI 应用**:
    *   **实例管理与并发控制**:
        *   **阶段 2 (初步并发控制)**: 为每个模型ID（`site.id`）维护一个独立的 `LLMWebsiteAutomator` 实例。使用与该实例绑定的 `asyncio.Lock()`来确保任何时刻只有一个 API 请求能够操作此浏览器实例。后续对同一模型的请求将异步等待锁的释放。
        *   **阶段 4 (浏览器实例池)**: 系统将升级为为每个模型ID维护一个 `asyncio.Queue` 作为 `LLMWebsiteAutomator` 实例池。池的大小在 `config.yaml` 中针对每个站点进行配置 (e.g., `pool_size: 3`)。API 请求时从对应模型的池中异步获取实例，使用完毕后异步归还。
    *   `active_automators: Dict[str, Set[LLMWebsiteAutomator]] = {}` (跟踪当前正在使用的实例，用于优雅关闭)。
    *   `@app.on_event("startup")`:
        *   初始化日志、加载配置。
        *   为每个启用的站点创建 `LLMWebsiteAutomator` 实例或实例池。
        *   启动后台健康检查任务。
    *   `@app.on_event("shutdown")`: 优雅关闭所有浏览器实例。
    *   `@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)`:
        *   从池中获取 `LLMWebsiteAutomator` 实例。
        *   根据 `request.stream` 调用 `automator.send_prompt_and_get_response` (传递 `stream_callback` 或不传)。
        *   如果是流式，返回 `StreamingResponse`。
        *   将实例归还到池中。
        *   处理 API 参数映射。
        *   处理临时选择器覆盖。
    *   **流式响应性能**:
        *   在 `config.yaml` 的站点配置中，增加 `response_handling.stream_poll_interval_ms` (e.g., `150`)，用于定义流式响应时 DOM 轮询的间隔时间（毫秒）。
    *   `@app.get("/health")`: 返回整体服务状态，可综合各 Automator 的健康状况。
    *   `@app.post("/reload_config")` (管理员接口，需鉴权): 触发配置热重载。

### 5.4 数据模型 (`models.py`)

*   **OpenAI API 模型**: `ChatMessage`, `ChatCompletionRequest` (带字段校验，如 `temperature` 范围), `ChatCompletionChoice`, `Usage`, `ChatCompletionResponse`。
*   **配置模型**: `LLMSiteConfig`, `GlobalSettings`, `AppConfig` (严格对应 `config.yaml` 结构，用于 Pydantic 校验)。
*   **内部辅助模型**: (如有需要)。

### 5.5 工具与日志 (`utils.py`)

*   `setup_logging(log_level: str, log_dir: str, global_config: GlobalSettings)`: 配置日志格式、级别、文件轮转、控制台输出。
*   `save_error_snapshot(page: Page, site_id: str, snapshot_dir: str)`: 保存截图和 DOM 内容到 `error_snapshots/` 目录。
*   `notify_on_critical_error(message: str, config: NotificationConfig)`: 发送告警 (e.g., Slack, Email)。

## 6. 分阶段实施计划 (Windows 环境优化)

### 阶段 0: 环境搭建与基础框架 (Windows 重点)

*   **任务：**
    1.  环境设置。
    2.  安装核心依赖 (`requirements.txt`)，包括 `psutil`, `tenacity`。
    3.  创建项目结构，`.env.example`, 初始 `config.yaml` (带 `global_settings` 和一个站点模板)。
    4.  `utils.py`: `setup_logging` 实现。
    5.  `models.py`: 定义所有 Pydantic 模型，包括 OpenAI API 模型（带字段校验）和配置模型。
    6.  `config_manager.py`: `load_config`, `get_config`, `get_site_config` 初步实现。
*   **产出：** 可加载配置的基础框架，日志系统可用。

### 阶段 1: 单一网站核心自动化 (非流式) 与基础测试

*   **任务：**
    1.  完善 `config.yaml` 中一个站点的详细配置 (selectors, profile, options\_to\_sync, response\_handling (full\_text), health\_check)。
    2.  `browser_handler.py` (`LLMWebsiteAutomator`):
        *   `__init__`, `_launch_browser_if_needed` (集成 `playwright-stealth`, 启动选项)。
        *   `_get_selector` (支持备用选择器)。
        *   `_perform_action` (封装 Playwright 调用，带超时、重试、失败快照)。
        *   `_ensure_page_options` 实现。
        *   `send_prompt_and_get_response` (非流式版本，即 `stream_callback=None`)。
        *   `is_healthy` 初步实现，包含基础的页面元素检查和标题检查。
        *   `close` 方法。
    3.  **基础资源监控实施**: 在 `LLMWebsiteAutomator` 的 `send_prompt_and_get_response` 方法执行完毕后（无论成功或失败，通过 `finally` 块确保），使用 `psutil` 获取当前 Python 进程的内存使用情况，并记录到 DEBUG 日志。
    4.  手动准备 Firefox Profile 并配置。
    5.  **同步测试**:
        *   为 `config_manager.py` 的核心功能编写单元测试。
        *   为 `LLMWebsiteAutomator` 的 `_ensure_page_options` 和非流式 `send_prompt_and_get_response` 方法编写集成测试（使用本地 mock HTML 页面）。
*   **产出：** 能够通过代码自动化一个网站的完整交互（登录复用、选项同步、发送提示、获取完整响应）。

### 阶段 2: FastAPI 接口封装 (非流式) 与初步并发控制

*   **任务：**
    1.  `main.py`:
        *   FastAPI 应用，`/v1/chat/completions` 端点 (非流式)。
        *   **并发控制**: 实现基于每个模型单实例 + `asyncio.Lock` 的并发访问控制。
        *   Startup/shutdown 事件处理 (加载配置、创建/关闭 automators)。
        *   `/health` 端点 (调用 `automator.is_healthy`)。
    2.  异步执行 `automator` 的阻塞方法 (`run_in_executor`)。
    3.  **同步测试**: 编写 API 端点（非流式，单模型串行访问）的集成测试。
*   **产出：** 可通过 OpenAI 兼容 API 调用单个网站的非流式聊天功能。

### 阶段 3: 实现流式响应与性能考量

*   **任务：**
    1.  `config.yaml`: 完善站点的 `response_handling` (stream 类型, `stream_text_property`, `stream_end_conditions` 带优先级和超时, `stream_poll_interval_ms`)。
    2.  `browser_handler.py` (`LLMWebsiteAutomator`):
        *   增强 `send_prompt_and_get_response` 以支持 `stream_callback`。
        *   实现流式文本监控、增量提取、结束条件判断逻辑，确保严格按照 `priority` 处理 `stream_end_conditions`。
    3.  `main.py`:
        *   修改 `/v1/chat/completions` 端点，当 `request.stream is True` 时返回 `StreamingResponse`。
    4.  **性能日志**: 记录流式轮询次数、每次轮询耗时、以及从发送提示到流结束的总耗时。
    5.  **同步测试**: 增加流式 API 的集成测试用例，覆盖不同的结束条件。
*   **产出：** API 支持流式响应。

### 阶段 4: 并发处理与浏览器实例管理

*   **任务：**
    1.  `main.py`:
        *   将 `automators` 从单实例字典改为 `automator_pools: Dict[str, asyncio.Queue[LLMWebsiteAutomator]]`。
        *   实现从池中获取/归还 Automator 实例的逻辑。
        *   配置池的大小 (`pool_size` 来自 `config.yaml`)。
    2.  `browser_handler.py`:
        *   `LLMWebsiteAutomator` 实例在处理完 `config.yaml` 中定义的 `max_requests_per_instance` 次请求后，在归还到池时，池将其关闭并异步创建一个新实例补充。
    3.  **资源监控增强与干预**:
        *   `main.py` 中启动一个后台 `asyncio` 任务，定期（例如，每 `default_instance_health_check_interval_seconds`）遍历所有池中的空闲实例，调用其 `is_healthy()` 方法。
        *   如果实例不健康，或通过 `psutil` 检测到其关联的 Firefox 进程（如果能够通过 PID 追踪，这部分较复杂，初期可能依赖主进程或整体资源判断）内存占用超过 `config.yaml` 中定义的 `max_memory_per_instance_mb`，则从池中移除该实例，执行 `close()`，并异步创建新实例补充。
*   **产出：** 提升并发处理能力，更有效地管理浏览器资源。

### 阶段 5: 鲁棒性、高级功能与可维护性提升

*   **任务：**
    1.  **配置热更新**: `main.py` 实现 `/reload_config` API 端点和相应逻辑。
    2.  **后台健康检查与实例重启**: `main.py` 中实现定期检查，并能重启不健康的 Automator 实例（从池中移除，关闭，创建新的补上）。
    3.  **监控与告警**:
        *   关键错误时通过 `utils.py` 中的通知函数发送告警。
    4.  **测试覆盖**: 编写更全面的单元测试和集成测试 (`pytest-playwright`)。
    5.  API参数映射/动态选择器注入: 根据需求实现。
*   **产出：** 更稳定、可监控、易于维护的服务。

## 7. 规范与最佳实践

### 7.1 日志规范与管理

*   **格式：** `%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s`
*   **级别：** DEBUG, INFO, WARNING, ERROR, CRITICAL。由 `config.yaml` 或 `.env` 控制。
*   **输出：** 控制台 + 日志文件 (按大小/日期轮转)。
*   **内容：** API请求详情、配置加载、浏览器关键操作、选项同步状态、响应提取、错误堆栈、性能指标（耗时）。
*   **日志清理**: `utils.py` 中的 `setup_logging` 函数配置 `logging.handlers.RotatingFileHandler` 时，必须设置 `backupCount` 参数（例如，`backupCount=10`），以自动限制保留的日志文件数量。

### 7.2 错误处理与快照

*   在 `_perform_action` 和其他关键浏览器交互点进行 `try-except`。
*   捕获 `playwright.sync_api.Error` (如 `TimeoutError`) 等。
*   失败时：
    1.  记录详细错误日志。
    2.  调用 `utils.save_error_snapshot` 保存当前页面截图和 DOM 到 `error_snapshots/`。
    3.  根据策略重试或向上抛出异常。
*   API层面返回标准HTTP错误码和OpenAI格式的错误JSON。

### 7.3 安全注意事项 (Windows 本地)

*   **Firefox Profile (`profiles/`)**: 包含登录凭证，目录权限应设为严格。在文档中强调。
*   **`.env` 文件**: 存储敏感信息（API Key等），不应提交到版本库。
*   **API 访问控制**:
    *   服务默认监听本地回环地址 (`127.0.0.1`)。
    *   如需局域网访问，在 Windows 防火墙中为应用或指定端口配置入站规则。
    *   在 `main.py` 中为所有 `/v1/*` 路径（或至少是修改配置的路径如 `/reload_config`）实现一个基于 HTTP `Authorization` Header 的简单 API Key 认证机制。API Key 在 `.env` 文件中配置 (e.g., `API_KEY="your_secret_key"`)。

### 7.4 测试策略与节奏

*   **单元测试 (`pytest`)**:
    *   `config_manager.py`: 测试配置加载、解析、校验（有效/无效配置）。
    *   `models.py`: Pydantic模型校验逻辑。
    *   `utils.py`: 工具函数。
*   **集成测试 (`pytest-playwright`)**:
    *   `browser_handler.py`: 针对一个或多个mock的本地HTML页面（模拟LLM网站结构）或一个稳定公开的测试网站，测试完整的交互流程（选项同步、提示发送、响应提取 - 包括流式和非流式）。
    *   `main.py`: 测试API端点的请求响应，模拟不同场景（有效请求、无效模型、流式/非流式、认证）。
*   **同步测试**: 在每个功能模块或重要特性开发完成后，立即编写并执行相关的单元测试和集成测试。测试应覆盖正常流程、边界条件和预期的错误场景。

### 7.5 依赖管理

*   使用 `requirements.txt` (或 `pyproject.toml` 若使用 Poetry/PDM)。
*   锁定主要依赖版本，以保证环境一致性。

## 8. 部署与运行 (Windows 本地)

*   **本地运行**: 主要场景。确保用户系统已安装 Firefox。

*   **运行步骤**:
    1.  确保 Python 和 pip 已安装并配置到系统 PATH。
    2.  克隆项目仓库。
    3.  创建并激活虚拟环境 (推荐):
        ```bash
        python -m venv venv
        .\venv\Scripts\activate  # Windows
        ```
    4.  安装依赖: `pip install -r requirements.txt`
    5.  安装 Playwright 浏览器驱动: `playwright install firefox`
    6.  复制 `.env.example` 为 `.env` 并根据需要修改其中的配置，例如 `API_KEY`。
    7.  手动为需要登录的站点准备 Firefox Profile。在 `config.yaml` 的站点配置中，`firefox_profile_dir` 字段指定 Profile 目录的路径（使用 Windows 绝对路径，例如 `"C:/path/to/your/profiles/site_A_profile"`，或相对于项目根目录的路径，例如 `"profiles/site_A_profile"`）。确保该目录存在并且包含有效的 Firefox Profile。
    8.  根据 `config.yaml` 配置各 LLM 站点信息。
    9.  运行服务: `uvicorn main:app --host 127.0.0.1 --port 8000` (或配置文件中指定的 host 和 port)。

---