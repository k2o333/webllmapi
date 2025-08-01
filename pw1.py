import asyncio
import time
from pathlib import Path
from playwright.async_api import async_playwright, Page, BrowserContext
import logging
from collections import Counter, defaultdict
import re

# --- 配置区 ---
TARGET_URL = "https://www.wenxiaobai.com/"
MONITOR_DURATION_SECONDS = 30
POLL_INTERVAL_SECONDS = 0.2
MIN_TEXT_CHANGE_LENGTH = 3
MAX_XPATH_DEPTH_FOR_RELATIVE = 5
MIN_OCCURRENCES_FOR_CANDIDATE = 3
LOG_FILE_PATH = "logs/advanced_stream_detector.log"
ENABLE_PLAYWRIGHT_TRACING = True
TRACE_DIR = Path("traces/advanced_stream_detection")
MAX_ANCESTORS_TO_COLLECT = 4  # 新增：要收集的祖先层级数量

# --- 日志设置 ---
detector_logger = logging.getLogger("AdvancedStreamDetector")
detector_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
detector_logger.addHandler(ch)
Path("logs").mkdir(exist_ok=True)
fh = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
fh.setFormatter(formatter)
detector_logger.addHandler(fh)

# --- JavaScript Code to Inject ---
JS_MUTATION_OBSERVER_SCRIPT = """
async (options) => {
    if (window.llmStreamChanges) {
        if (window.llmStreamChanges.observer) window.llmStreamChanges.observer.disconnect();
        window.llmStreamChanges.records = [];
    }
    window.llmStreamChanges = {
        records: [],
        observer: null,
        options: options || {}
    };
    const getElementXPath = (element) => {
        if (!element || !element.parentNode) return null;
        if (element === document.body) return '/html/body';
        if (element === document.documentElement) return '/html';
        let ix = 0;
        const siblings = element.parentNode.childNodes;
        let pathPart = element.tagName.toLowerCase();
        for (let i = 0; i < siblings.length; i++) {
            const sibling = siblings[i];
            if (sibling === element) {
                const parentPath = getElementXPath(element.parentNode);
                if (!parentPath) return null;
                return parentPath + '/' + pathPart + '[' + (ix + 1) + ']';
            }
            if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
                ix++;
            }
        }
        return null;
    };
    const getAncestorsInfo = (element, maxLevel) => {
        const ancestors = [];
        let current = element.parentElement;
        let levels = 0;
        while (current && current !== document.body && levels < maxLevel) {
            ancestors.push({
                tag: current.tagName.toLowerCase(),
                id: current.id || '',
                classes: Array.from(current.classList || [])
            });
            current = current.parentElement;
            levels++;
        }
        return ancestors;
    };
    const observer = new MutationObserver((mutationsList) => {
        for (const mutation of mutationsList) {
            let targetElement = null;
            let changeType = mutation.type;
            let textSample = "";
            if (mutation.type === 'childList') {
                targetElement = mutation.target;
                if (mutation.addedNodes.length > 0) {
                    const addedNode = mutation.addedNodes[0];
                    if (addedNode.textContent && addedNode.textContent.trim().length > 0) {
                        textSample = addedNode.textContent.trim().substring(0, 50);
                    } else if (addedNode.nodeType === Node.ELEMENT_NODE && addedNode.innerText) {
                         textSample = addedNode.innerText.trim().substring(0,50);
                    }
                }
            } else if (mutation.type === 'characterData') {
                targetElement = mutation.target.parentElement;
                if (targetElement && mutation.target.textContent &&
                    mutation.target.textContent.trim().length >= (window.llmStreamChanges.options.minTextChangeLength || 1)) {
                    textSample = mutation.target.textContent.trim().substring(0, 50);
                } else {
                    continue; 
                }
            }
            if (targetElement) {
                const absXpath = getElementXPath(targetElement);
                if (absXpath) {
                    const ancestors = getAncestorsInfo(targetElement, window.llmStreamChanges.options.maxAncestors || 2);
                    window.llmStreamChanges.records.push({
                        absXpath: absXpath,
                        type: changeType,
                        timestamp: Date.now(),
                        textSample: textSample.replace(/\\n/g, ' '),
                        targetTagName: targetElement.tagName.toLowerCase(),
                        targetId: targetElement.id || '',
                        targetClasses: Array.from(targetElement.classList || []),
                        ancestors: ancestors  // 新增：存储祖先元素信息
                    });
                }
            }
        }
    });
    observer.observe(document.documentElement, {
        childList: true,
        subtree: true,
        characterData: true,
        characterDataOldValue: false
    });
    window.llmStreamChanges.observer = observer;
    return true;
}
"""
JS_GET_CHANGES_SCRIPT = """
() => {
    if (window.llmStreamChanges && window.llmStreamChanges.records) {
        const records = window.llmStreamChanges.records;
        window.llmStreamChanges.records = []; // Clear after fetching
        return records;
    }
    return [];
}
"""
JS_STOP_OBSERVER_SCRIPT = """
() => {
    if (window.llmStreamChanges && window.llmStreamChanges.observer) {
        window.llmStreamChanges.observer.disconnect();
        delete window.llmStreamChanges;
        return true;
    }
    return false;
}
"""

# --- Helper functions ---
def extract_class_segment(classes):
    """提取有意义的CSS类片段"""
    if not classes:
        return ""
        
    # 找到最长类名
    longest = max(classes, key=len, default="")
    if not longest:
        return ""
    
    # 提取中间部分作为标识（避免前后缀变化）
    start = len(longest) // 3
    end = (len(longest) * 2) // 3
    return longest[start:end]
    
def generate_relative_xpath_candidates(abs_xpath: str, tag_name: str, target_id: str, target_classes: list, ancestors: list) -> list:
    """生成多个可能的相对XPath候选路径"""
    candidates = []
    
    # 1. 尝试基于目标元素自身属性创建候选
    if target_id and not target_id.isnumeric():
        candidates.append(f"//{tag_name}[@id='{target_id}']")
    
    if target_classes:
        # 过滤掉无意义的类名
        meaningful_classes = [
            c for c in target_classes 
            if c and len(c) > 2 and not c.isnumeric() 
            and not re.match(r"^(js-|is-|has-|active|selected|hidden|focused|container|wrapper|base|main|content|item|block|module|component)", c, re.I)
        ]
        
        if meaningful_classes:
            # 基于单个类名
            for cls in meaningful_classes[:3]:
                candidates.append(f"//{tag_name}[contains(@class, '{cls}')]")
            
            # 基于多个类名组合
            if len(meaningful_classes) >= 2:
                candidates.append(f"//{tag_name}[contains(@class, '{meaningful_classes[0]}') and contains(@class, '{meaningful_classes[1]}')]")
            
            # 基于类名片段
            class_segment = extract_class_segment(meaningful_classes)
            if class_segment:
                candidates.append(f"//{tag_name}[contains(@class, '{class_segment}')]")
    
    # 2. 尝试基于祖先元素的id或class创建候选路径
    parts = abs_xpath.split('/')
    
    for i, ancestor_info in enumerate(ancestors):
        if i >= MAX_XPATH_DEPTH_FOR_RELATIVE: 
            break
            
        ancestor_tag = ancestor_info.get('tag', 'div')
        ancestor_id = ancestor_info.get('id', '')
        ancestor_classes = ancestor_info.get('classes', [])
        
        # 基于祖先ID创建路径
        if ancestor_id and not ancestor_id.isnumeric():
            # 计算从祖先到目标的相对层级数
            levels = i + 1
            if len(parts) > levels:
                # 获取从祖先到目标的XPath部分
                relative_part = "/".join(parts[-(levels+1):])
                candidate = f"//{ancestor_tag}[@id='{ancestor_id}']//{relative_part}"
                candidates.append(candidate)
                # 找到一个就足够好了，可以跳出循环
                break
        
        # 如果祖先没有ID，尝试使用祖先的类名
        elif ancestor_classes:
            # 过滤有意义的祖先类名
            meaningful_ancestor_classes = [
                c for c in ancestor_classes 
                if c and len(c) > 2 and not c.isnumeric() 
                and not re.match(r"^(js-|is-|has-|active|selected|hidden|focused|container|wrapper|base|main|content|item|block|module|component)", c, re.I)
            ]
            
            if meaningful_ancestor_classes:
                # 计算从祖先到目标的相对层级数
                levels = i + 1
                if len(parts) > levels:
                    # 获取从祖先到目标的XPath部分
                    relative_part = "/".join(parts[-(levels+1):])
                    # 使用多个类名组合提高准确性
                    if len(meaningful_ancestor_classes) >= 2:
                        candidate = f"//{ancestor_tag}[contains(@class, '{meaningful_ancestor_classes[0]}') and contains(@class, '{meaningful_ancestor_classes[1]}')]//{relative_part}"
                    else:
                        candidate = f"//{ancestor_tag}[contains(@class, '{meaningful_ancestor_classes[0]}')]//{relative_part}"
                    candidates.append(candidate)
    
    # 3. 如果仍然没有候选，尝试基于body和类名创建后备路径
    if not candidates and target_classes:
        meaningful_classes = [
            c for c in target_classes 
            if c and len(c) > 2 and not c.isnumeric() 
            and not re.match(r"^(js-|is-|has-|active|selected|hidden|focused|container|wrapper|base|main|content|item|block|module|component)", c, re.I)
        ]
        
        if meaningful_classes:
            candidates.append(f"//body//{tag_name}[contains(@class, '{meaningful_classes[0]}')]")
    
    # 4. 最后的备用方案：尝试使用父元素的ID或类名
    if not candidates and ancestors:
        first_ancestor = ancestors[0] if ancestors else {}
        if first_ancestor.get('id'):
            levels = 1
            if len(parts) > levels:
                relative_part = "/".join(parts[-(levels+1):])
                candidates.append(f"//{first_ancestor.get('tag')}[@id='{first_ancestor.get('id')}']//{relative_part}")
        elif first_ancestor.get('classes'):
            meaningful_classes = [
                c for c in first_ancestor.get('classes') 
                if c and len(c) > 2 and not c.isnumeric() 
                and not re.match(r"^(js-|is-|has-|active|selected|hidden|focused|container|wrapper|base|main|content|item|block|module|component)", c, re.I)
            ]
            if meaningful_classes:
                levels = 1
                if len(parts) > levels:
                    relative_part = "/".join(parts[-(levels+1):])
                    candidates.append(f"//{first_ancestor.get('tag')}[contains(@class, '{meaningful_classes[0]}')]//{relative_part}")
    
    # 去重并返回结果
    return list(dict.fromkeys(c for c in candidates if c))

async def analyze_changes(all_changes: list, current_session_id: int):
    detector_logger.info(f"\n--- 分析会话 {current_session_id} 的 {len(all_changes)} 条 DOM 变化记录 ---")
    if not all_changes:
        detector_logger.info("没有检测到 DOM 变化。")
        return {}
    
    xpath_details = defaultdict(lambda: {
        'count': 0, 'types': Counter(), 'texts': [], 'tag_name': '',
        'target_id': '', 'target_classes': [], 'relative_candidates': [],
        'ancestors': []
    })
    
    for change in all_changes:
        abs_xpath = change['absXpath']
        details = xpath_details[abs_xpath]
        details['count'] += 1
        details['types'][change['type']] += 1
        if change['textSample']: 
            details['texts'].append(change['textSample'])
        if not details['tag_name']:
            details['tag_name'] = change['targetTagName']
            details['target_id'] = change['targetId']
            details['target_classes'] = change['targetClasses']
            details['ancestors'] = change.get('ancestors', [])
            details['relative_candidates'] = generate_relative_xpath_candidates(
                abs_xpath, 
                change['targetTagName'], 
                change['targetId'], 
                change['targetClasses'],
                change.get('ancestors', [])  # 新增：传递祖先信息
            )
    
    detector_logger.info("\n--- XPath 变化频率统计 (按变化次数排序) ---")
    sorted_xpaths_items = sorted(xpath_details.items(), key=lambda item: (item[1]['count'], -len(item[0])), reverse=True)
    processed_results = {}
    
    for abs_xpath, details in sorted_xpaths_items:
        if details['count'] < MIN_OCCURRENCES_FOR_CANDIDATE: 
            continue
            
        processed_results[abs_xpath] = details
        detector_logger.info(f"绝对 XPath: {abs_xpath}")
        detector_logger.info(f"  元素标签: {details['tag_name']}, ID: '{details['target_id']}', Classes: {details['target_classes'][:5]}")
        detector_logger.info(f"  发生变化次数: {details['count']}")
        
        types_str = ", ".join([f"{t}:{c}" for t, c in details['types'].items()])
        detector_logger.info(f"  变化类型: {types_str}")
        
        text_samples = list(set(details['texts']))
        if text_samples: 
            detector_logger.info(f"  关联文本片段 (最多显示3个不同样本): {text_samples[:3]}")
        
        if details['ancestors']:
            detector_logger.info(f"  祖先元素信息:")
            for i, ancestor in enumerate(details['ancestors'][:3]):
                classes = ancestor.get('classes', [])[:3]
                classes_display = ', '.join(classes) + ('...' if len(classes) > 3 else '')
                detector_logger.info(f"    层级 {i+1}: {ancestor.get('tag', '?')}, ID: '{ancestor.get('id', '')}', Classes: [{classes_display}]")
        
        if details['relative_candidates']:
            detector_logger.info(f"  启发式相对 XPath 候选:")
            for rel_xpath in details['relative_candidates'][:3]: 
                detector_logger.info(f"    - {rel_xpath}")
        else:
            detector_logger.info("  无法生成相对 XPath 候选（元素缺少标识属性）")
        
        detector_logger.info("-" * 20)
    
    parent_xpath_counts = Counter()
    for abs_xpath in processed_results.keys():
        parts = abs_xpath.split('/')
        if len(parts) > 2:
            parent_xpath = "/".join(parts[:-1])
            parent_xpath_counts[parent_xpath] += processed_results[abs_xpath]['count']
    
    detector_logger.info("\n--- 潜在父容器 (基于其下子元素总变化量) ---")
    most_active_parents = parent_xpath_counts.most_common(5)
    if most_active_parents:
        for parent_xpath, total_child_changes in most_active_parents:
             detector_logger.info(f"  父 XPath: {parent_xpath} (子元素总变化: {total_child_changes})")
    else:
        detector_logger.info("  未找到明显活跃的父容器。")
    
    detector_logger.info(f"--- 会话 {current_session_id} 分析结束 ---\n")
    return processed_results

def analyze_aggregated_changes(all_sessions_data: dict, num_total_sessions: int):
    detector_logger.info(f"\n--- 分析所有 {num_total_sessions} 个会话的聚合 DOM 变化数据 ---")
    if not all_sessions_data:
        detector_logger.info("没有来自任何会话的分析数据。")
        return
    
    aggregated_details = defaultdict(lambda: {
        'total_count': 0, 'session_appearances': 0, 'session_counts': Counter(),
        'texts': [], 'tag_name': '', 'target_id': '', 'target_classes': [], 
        'relative_candidates': [], 'ancestors': []
    })
    
    for session_id, session_results in all_sessions_data.items():
        for abs_xpath, details in session_results.items():
            agg_detail = aggregated_details[abs_xpath]
            agg_detail['total_count'] += details['count']
            agg_detail['session_appearances'] += 1
            agg_detail['session_counts'][session_id] = details['count']
            agg_detail['texts'].extend(details['texts'])
            
            if not agg_detail['tag_name']:
                agg_detail['tag_name'] = details['tag_name']
                agg_detail['target_id'] = details['target_id']
                agg_detail['target_classes'] = details['target_classes']
                agg_detail['relative_candidates'] = details['relative_candidates']
                agg_detail['ancestors'] = details['ancestors']
    
    sorted_aggregated_xpaths = sorted(
        aggregated_details.items(),
        key=lambda item: (item[1]['session_appearances'], item[1]['total_count'], -len(item[0])),
        reverse=True
    )
    
    detector_logger.info("\n--- 跨会话 XPath 变化一致性与频率 (按推荐度排序) ---")
    final_recommendations = []
    
    for abs_xpath, details in sorted_aggregated_xpaths:
        if details['session_appearances'] < max(1, num_total_sessions // 2) and num_total_sessions > 1: 
            continue
            
        if details['total_count'] < MIN_OCCURRENCES_FOR_CANDIDATE * details['session_appearances'] * 0.3: 
            continue
            
        final_recommendations.append((abs_xpath, details))
        detector_logger.info(f"绝对 XPath: {abs_xpath}")
        detector_logger.info(f"  元素标签: {details['tag_name']}, ID: '{details['target_id']}', Classes: {details['target_classes'][:5]}")
        detector_logger.info(f"  总变化次数: {details['total_count']}")
        detector_logger.info(f"  出现在 {details['session_appearances']}/{num_total_sessions} 个会话中。")
        
        if details['ancestors']:
            detector_logger.info(f"  祖先元素信息 (来自一个会话样本):")
            for i, ancestor in enumerate(details['ancestors'][:2]):
                classes = ancestor.get('classes', [])[:3]
                classes_display = ', '.join(classes) + ('...' if len(classes) > 3 else '')
                detector_logger.info(f"    层级 {i+1}: {ancestor.get('tag', '?')}, ID: '{ancestor.get('id', '')}', Classes: [{classes_display}]")
        
        text_samples = list(set(details['texts']))
        if text_samples: 
            detector_logger.info(f"  关联文本片段 (部分样本): {text_samples[:3]}")
        
        if details['relative_candidates']:
            detector_logger.info(f"  启发式相对 XPath 候选:")
            for rel_xpath in details['relative_candidates'][:3]: 
                detector_logger.info(f"    - {rel_xpath}")
        else:
            detector_logger.info("  无法生成相对 XPath 候选（元素缺少标识属性）")
        
        detector_logger.info("-" * 30)
    
    if not final_recommendations:
        detector_logger.info("未能从多次会话中找到足够一致或频繁的流式容器 XPath。")
    else:
        detector_logger.info("\n--- 总结：高度推荐的流式容器 XPath (请重点验证以下条目) ---")
        for i, (abs_xpath, details) in enumerate(final_recommendations[:5]):
            detector_logger.info(f"  推荐 {i+1}:")
            detector_logger.info(f"    绝对 XPath: {abs_xpath}")
            
            if details['relative_candidates']:
                 detector_logger.info(f"    相对候选 (选1-2个最可能的):")
                 for rel_xpath in details['relative_candidates'][:2]: 
                     detector_logger.info(f"      - {rel_xpath}")
            else:
                detector_logger.info(f"    (未能生成简洁的相对 XPath 候选 - 尝试使用绝对路径)")
                
            detector_logger.info(f"    (总变化: {details['total_count']}, 出现会话数: {details['session_appearances']})")
    
    detector_logger.info("\n提示: 请将以上建议的 XPath 在浏览器开发者工具中测试 ($x(\"xpath\"))，并结合 Playwright Trace (如果启用) 进行验证其是否准确捕获了流式文本。")

# --- 使用非阻塞输入 ---
async def get_user_input(prompt: str) -> str:
    """Gets user input in a non-blocking way by running input() in a separate thread."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)

async def run_observer_session(page: Page, context: BrowserContext, session_id: int) -> list:
    detector_logger.info(f"\n--- 会话 {session_id} ---")
    detector_logger.info(f"请在浏览器中为会话 {session_id} 手动输入 Prompt 并提交。")
    
    # 循环直到用户按下Enter键或's'键
    while True:
        user_action_prompt = (
            f"完成会话 {session_id} 的手动提交后，请按 Enter 键开始监控 DOM 变化 "
            f"(或输入 's' 跳过此会话, 'q' 退出整个脚本): "
        )
        input_value = await get_user_input(user_action_prompt)
        input_value = input_value.strip().lower()
        if input_value == "" or input_value == 's' or input_value == 'q':
            break
        else:
            detector_logger.info("无效输入，请按 Enter, 's', 或 'q'.")

    if input_value == 's':
        detector_logger.info(f"跳过会话 {session_id}。")
        return []
    if input_value == 'q':
        detector_logger.info(f"用户请求退出脚本。")
        return "quit"  # 返回特殊值表示退出

    detector_logger.info(f"会话 {session_id}: 正在页面注入 MutationObserver 并开始监听 {MONITOR_DURATION_SECONDS} 秒...")
    await page.evaluate(JS_MUTATION_OBSERVER_SCRIPT, {
        "minTextChangeLength": MIN_TEXT_CHANGE_LENGTH,
        "maxAncestors": MAX_ANCESTORS_TO_COLLECT  # 新增：传递祖先收集层级参数
    })

    session_dom_changes = []
    monitor_start_time = time.time()

    try:
        while time.time() - monitor_start_time < MONITOR_DURATION_SECONDS:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            changes_batch = await page.evaluate(JS_GET_CHANGES_SCRIPT)
            if changes_batch:
                detector_logger.debug(f"会话 {session_id}: 获取到 {len(changes_batch)} 条 DOM 变化记录。")
                for change in changes_batch:
                    change['sessionId'] = session_id
                session_dom_changes.extend(changes_batch)
        detector_logger.info(f"会话 {session_id}: 监控时间达到。总共记录 {len(session_dom_changes)} 条原始变化。")

    except KeyboardInterrupt:
        detector_logger.info(f"会话 {session_id}: 用户中断监控。")
        raise 
    finally:
        detector_logger.info(f"会话 {session_id}: 正在停止页面 MutationObserver...")
        await page.evaluate(JS_STOP_OBSERVER_SCRIPT)
    
    return session_dom_changes

async def main():
    all_sessions_analysis_data = {}
    session_count = 0

    async with async_playwright() as p:
        detector_logger.info("正在启动 Firefox...")
        browser = await p.firefox.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        tracing_started = False
        trace_path = None
        
        # 处理Playwright Tracing
        if ENABLE_PLAYWRIGHT_TRACING:
            TRACE_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            trace_path = TRACE_DIR / f"adv_detect_multisession_trace_{timestamp}.zip"
            try:
                await context.tracing.start(name="adv_detect_multisession", screenshots=True, snapshots=True, sources=True)
                tracing_started = True
                detector_logger.info(f"Playwright Tracing 已启动 (覆盖所有会话)。Trace 文件将保存到: {trace_path}")
            except Exception as e_trace_start:
                detector_logger.error(f"启动 Playwright Tracing 时出错: {e_trace_start}")
                tracing_started = False

        try:
            detector_logger.info(f"正在导航到: {TARGET_URL}")
            await page.goto(TARGET_URL, timeout=60000)
            detector_logger.info(f"页面加载完成。")

            while True:
                session_count += 1
                try:
                    session_result = await run_observer_session(page, context, session_count)
                    
                    # 检查是否要退出
                    if session_result == "quit":
                        detector_logger.info("用户选择退出整个脚本。")
                        break
                        
                    if session_result and isinstance(session_result, list) and len(session_result) > 0:
                        session_analysis_result = await analyze_changes(session_result, session_count)
                        if session_analysis_result:
                            all_sessions_analysis_data[session_count] = session_analysis_result
                    else:
                        detector_logger.info(f"会话 {session_count} 未记录到 DOM 变化。")
                except Exception as e:
                    detector_logger.error(f"执行会话 {session_count} 时发生错误: {e}", exc_info=True)
                
                # 询问是否继续
                if session_count > 0:
                    cont_prompt = "是否要进行下一次 Prompt 观察? (y/n, q退出): "
                    cont = await get_user_input(cont_prompt)
                    cont = cont.strip().lower()
                    if cont == 'q':
                        detector_logger.info("用户选择退出。")
                        break
                    if cont != 'y':
                        break
            
            # 聚合分析所有会话
            if all_sessions_analysis_data:
                analyze_aggregated_changes(all_sessions_analysis_data, session_count)
            elif session_count > 0:
                detector_logger.info("所有会话均未记录到足够用于分析的 DOM 变化。")

        except Exception as e:
            detector_logger.error(f"主流程发生错误: {e}", exc_info=True)
        finally:
            detector_logger.info("正在关闭浏览器...")
            
            # 停止并保存Trace文件
            if tracing_started:
                try:
                    await context.tracing.stop(path=str(trace_path))
                    detector_logger.info(f"Playwright Trace 已保存到: {trace_path}")
                    detector_logger.info(f"你可以使用 'playwright show-trace {trace_path}' 来查看。")
                except Exception as e_trace_stop:
                    detector_logger.error(f"保存 Playwright Trace 时出错: {e_trace_stop}")
            
            await browser.close()
            detector_logger.info("浏览器已关闭。脚本结束。")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        detector_logger.info("\n脚本被用户强制退出 (主程序级别)。")
    except SystemExit:
        detector_logger.info("\n脚本正常退出。")