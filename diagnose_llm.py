"""临时诊断脚本：复现 openai_gpt.py 里的 try/except fallback。"""
import os
from dotenv import load_dotenv
load_dotenv("./agentboard/.env")

import openai
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_base = os.environ["OPENAI_API_BASE"]

print(f"endpoint={openai.api_base}, sdk={openai.__version__}")

system_message = "You are a master in moving objects."
user_prompt = (
    "The robot has actions: pick, drop, move.\n"
    "Observation: Ball1 is at rooma. Robby is at rooma.\n"
    "Action: "
)
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt},
]

def show(label, **kwargs):
    print("\n===", label, "===")
    print("kwargs:", {k: v for k, v in kwargs.items() if k != "messages"})
    try:
        r = openai.ChatCompletion.create(messages=messages, **kwargs)
        print("usage:", dict(r["usage"]))
        choice = r["choices"][0]
        print("finish_reason:", choice.get("finish_reason"))
        content = choice["message"].get("content")
        print("content repr:", repr(content))
    except Exception as e:
        print("ERROR:", type(e).__name__, str(e)[:400])

# 用例 A：完全照代码走 — max_tokens=100 + stop="\n\n"
show("A: max_tokens=100, stop=\\n\\n",
     model="gpt-4-turbo", temperature=0, max_tokens=100, stop="\n\n")

# 用例 B：fallback 路径 — max_completion_tokens=100
show("B: max_completion_tokens=100, stop=\\n\\n",
     model="gpt-4-turbo", temperature=0, max_completion_tokens=100, stop="\n\n")

# 用例 C：去掉 stop
show("C: max_completion_tokens=100, no stop",
     model="gpt-4-turbo", temperature=0, max_completion_tokens=100)

# 用例 D：max_completion_tokens 调大
show("D: max_completion_tokens=1024, no stop",
     model="gpt-4-turbo", temperature=0, max_completion_tokens=1024)

# 用例 E：换 gpt-4o 试试
show("E: gpt-4o, max_tokens=100, stop=\\n\\n",
     model="gpt-4o", temperature=0, max_tokens=100, stop="\n\n")
