import json
from openai import OpenAI

def main(model_name="DeepSeek-V3.1", thinking=False, use_json_schema=None):
    """
    model_name: 模型名称
    thinking: 是否开启思考模式（DeepSeek 用 thinking, Qwen3 用 enable_thinking）
    use_json_schema:
        True  -> 启用 JSON Schema
        False -> 不传 response_format
        None  -> 不传 response_format
    """
    niogpt_base_url = "http://0.0.0.0:30000/v1"
    niogpt_api_key = "sk-no-api-key-needed"
    client = OpenAI(
        api_key=niogpt_api_key,
        base_url=niogpt_base_url
    )

    # JSON Schema 示例
    json_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }

    # 根据模型类型映射思考开关 key
    if "qwen" in model_name.lower():
        chat_kwargs = {"enable_thinking": thinking}
    else:
        chat_kwargs = {"thinking": thinking}

    # 基础请求参数
    request_params = dict(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": "Give me the information of the capital of France in the JSON format.",
            }
        ],
        temperature=0,
        max_tokens=512,
        extra_body={"chat_template_kwargs": chat_kwargs}
    )

    # 只有 True 时才加 response_format
    if use_json_schema is True:
        request_params["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "foo", "schema": json_schema}
        }

    response = client.chat.completions.create(**request_params)

    # 输出 completion_tokens
    print("========== completion_tokens ==========")
    print(response.usage.completion_tokens)

    # 输出 content
    print("========== content ==========")
    for choice in response.choices:
        print(choice.message.content if choice.message.content is not None else "None")

    # 输出 reasoning_content
    print("========== reasoning_content ==========")
    for choice in response.choices:
        print(choice.message.reasoning_content if choice.message.reasoning_content is not None else "None")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    # 示例1：DeepSeek 开启思考且使用 JSON Schema
#     main(model_name="DeepSeek-V3.1", thinking=False, use_json_schema=False)

    # 示例2：Qwen3 不传 response_format
    main(model_name="Qwen3-1.7B", thinking=True, use_json_schema=True)

    # 示例3：同样不传 response_format（参数缺省）
    # True False
