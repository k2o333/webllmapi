import requests
import json
import argparse
import os
from typing import Optional

def call_chat_completion_api(
    model_id: str,
    prompt: str,
    api_key: str,
    stream: bool = False,
    api_base_url: str = "http://localhost:8000"
):
    """
    调用本地的 LLM 网页自动化 API。

    Args:
        model_id (str): 配置中 LLM 站点的 ID (例如 "wenxiaobai")，
                        或站点 ID/模型变体 ID (例如 "wenxiaobai/deepseek-v3")。
        prompt (str): 要发送给 LLM 的用户提示。
        api_key (str): 访问 API 的密钥 (与 .env 中的 API_KEY 对应)。
        stream (bool): 如果为 True，则启用流式响应。
        api_base_url (str): 本地 API 服务的基地址。
    """
    url = f"{api_base_url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": api_key
    }

    messages = [{"role": "user", "content": prompt}]

    data = {
        "model": model_id,
        "messages": messages,
        "stream": stream
    }

    print(f"\n--- 调用 API ---")
    print(f"URL: {url}")
    print(f"模型: {model_id} (流式: {stream})")
    print(f"提示: {prompt[:50]}...")
    print(f"----------------\n")

    try:
        if stream:
            # 流式请求
            full_response_content = ""
            with requests.post(url, headers=headers, json=data, stream=True) as response:
                response.raise_for_status() # 检查 HTTP 错误，如果状态码不是 2xx 则抛出异常

                print("流式响应:")
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        
                        if decoded_line.startswith("data:"):
                            json_data_str = decoded_line[len("data:"):].strip()
                            
                            if json_data_str == "[DONE]":
                                break # 流结束标记
                            
                            try:
                                chunk = json.loads(json_data_str)
                                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if content:
                                    print(content, end='', flush=True) # 实时打印内容
                                    full_response_content += content
                            except json.JSONDecodeError:
                                print(f"\n[API 响应解析错误] 无法解析 JSON 行: {json_data_str}")
                                # 可以根据需要记录更详细的错误
                print("\n\n流式响应结束。")
                print(f"总计接收字符: {len(full_response_content)}")

        else:
            # 非流式请求
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status() # 检查 HTTP 错误

            print("非流式响应:")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))

    except requests.exceptions.HTTPError as e:
        print(f"\n[HTTP 错误] 请求失败: {e}")
        if e.response is not None:
            print(f"服务器响应: {e.response.text}")
    except requests.exceptions.ConnectionError as e:
        print(f"\n[连接错误] 无法连接到服务器。请确保您的 FastAPI 服务已运行在 {api_base_url}。错误: {e}")
    except requests.exceptions.Timeout as e:
        print(f"\n[超时错误] 请求超时。错误: {e}")
    except requests.exceptions.RequestException as e:
        print(f"\n[未知请求错误] 发生未知请求错误: {e}")
    except Exception as e:
        print(f"\n[意外错误] 发生意外错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="调用本地 LLM 网页自动化 API")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="LLM 站点 ID 或 '站点ID/模型变体ID' (例如: 'wenxiaobai' 或 'wenxiaobai/deepseek-v3')")
    parser.add_argument("--prompt", "-p", type=str, required=True,
                        help="发送给 LLM 的用户提示")
    parser.add_argument("--stream", "-s", action="store_true",
                        help="启用流式响应 (默认为非流式)")
    parser.add_argument("--api-key", "-k", type=str,
                        help="API 密钥 (如果未提供，将尝试从环境变量 API_KEY 获取)")
    parser.add_argument("--url", "-u", type=str, default="http://localhost:8000",
                        help="本地 API 服务的基地址 (默认为 http://localhost:8000)")

    args = parser.parse_args()

    # 如果命令行未提供 API 密钥，则尝试从环境变量获取
    api_key_to_use = args.api_key
    if not api_key_to_use:
        api_key_to_use = os.getenv("API_KEY")
        if not api_key_to_use:
            print("错误: 未提供 API 密钥。请通过 --api-key 参数提供，或在 .env 文件中设置 API_KEY 环境变量。", file=os.sys.stderr)
            os.sys.exit(1)

    call_chat_completion_api(
        model_id=args.model,
        prompt=args.prompt,
        api_key=api_key_to_use,
        stream=args.stream,
        api_base_url=args.url
    )