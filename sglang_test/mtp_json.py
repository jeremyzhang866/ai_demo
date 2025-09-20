import json
from openai import OpenAI

def main(model_name="Qwen3-14B", thinking=False, use_json_schema=None):

    niogpt_base_url = "http://0.0.0.0:30000/v1"
    niogpt_api_key = "sk-no-api-key-needed"
    client = OpenAI(
        api_key=niogpt_api_key,
        base_url=niogpt_base_url
    )
    json_schema = {
        "type": "object",
        "properties": {
            "population": {"type": "integer"},
            "name": {"type": "string", "pattern": "^[\\w]+$"},
        },
        "required": ["name", "population"],
    }
    
    chat_kwargs = {"enable_thinking": thinking}

    # 基础请求参数
    request_params = dict(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": "show me the information of the capital of China in the JSON format.",
            }
        ],
        temperature=0,
        max_tokens=512,
        extra_body={"chat_template_kwargs": chat_kwargs}
    )
    if use_json_schema is True:
        request_params["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "foo", "schema": json_schema}
        }

    response = client.chat.completions.create(**request_params)

    print("========== completion_tokens ==========")
    print(response.usage.completion_tokens)

    print("========== content ==========")
    for choice in response.choices:
        print(choice.message.content if choice.message.content is not None else "None")

    print("========== reasoning_content ==========")
    for choice in response.choices:
        print(choice.message.reasoning_content if choice.message.reasoning_content is not None else "None")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main(model_name="Qwen3-14B", thinking=True, use_json_schema=True)
