from openai import OpenAI
import numpy as np
import os
from datetime import datetime
import json

with open("deepseek_key") as file:
    deepseek_key = file.read()

client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com/")


class CangjieLearningAgent:
    def __init__(self, client, model_name="deepseek-chat", dump_dir="./learn_agent_dir"):
        self.metaprompt_learning = """【已掌握的知识】%s

【目标】掌握一种名为仓颉的计算机编程语言，可以通过它来完成基本的程序编写任务。

【学习方式】你将被提供一份关于仓颉语言的文字资料，你需要生成一份记忆，该记忆将被添加到【已掌握的知识】部分（该部分将帮助你解决面临的问题）。

资料："""
        self.metaprompt_asking = """【已掌握的知识】%s
        请你用仓颉语言编写以下代码，并确保该代码可以被编译运行："""
        self.memory = ""
        self.client = client
        self.model_name = model_name
        if dump_dir is not None and not os.path.isdir(dump_dir):
            os.mkdir(dump_dir)
        self.dump_dir = dump_dir

    def learn(self, text):
        reply, ppl = self._inference(text, self.metaprompt_learning % self.memory)
        self._add_memory(reply)

    def ask(self, question):
        reply, ppl = self._inference(question, self.metaprompt_asking)
        return reply, ppl

    def _add_memory(self, text):
        self.memory += "\n" + text

    def _inference(self, prompt, system_prompt):
        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            logprobs=True
        )
        ppl = np.exp([ChatCompletionTokenLogprob.logprob for ChatCompletionTokenLogprob in
                      chat_completion.choices[0].logprobs.content]).mean()
        reply = chat_completion.choices[0].message.content
        if self.dump_dir is not None:
            with open(self.dump_dir + "/" + self.get_now() + ".json", "w") as f:
                json.dump({
                    "model_name": self.model_name,
                    "system_prompt": system_prompt,
                    "user_prompt": prompt,
                    "reply": reply,
                    "ppl": ppl,
                }, f)
        return reply, ppl

    def get_now(self):
        return datetime.now().strftime('%d-%m-%y-%H-%M-%S-%f')


questions = [
    "输出`hello world`。"
]

agent = CangjieLearningAgent(client=client)

for book_path in (
        "cangjie/基本概念/标识符.md",
        "cangjie/基本概念/程序结构.md",
        "cangjie/基本概念/表达式.md",
        "cangjie/基本概念/函数.md"
):
    reply, _ = agent.ask(questions[0])
    print(reply, "before", book_path)
    with open(book_path, "r") as f:
        text = f.read()
    agent.learn(text)

# from openai import OpenAI
# import json
# import numpy as np
#
# # https://baijiahao.baidu.com/s?id=1740598555675692511
#
# def inference(prompt, model, system_prompt, client, is_json=True):
#     chat_completion = client.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": prompt}
#         ],
#         response_format=({"type": "json_object"} if is_json else {"type": "text"}),
#         temperature=0,
#         logprobs=True
#     )
#     ppl = np.exp([ChatCompletionTokenLogprob.logprob for ChatCompletionTokenLogprob in chat_completion.choices[0].logprobs.content]).mean()
#     reply = chat_completion.choices[0].message.content
#     return reply, ppl
#
# with open("deepseek_key") as file:
#     deepseek_key = file.read()
#
# client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com/")
#
# data = json.load(open("dataset/inequation.json", encoding="utf-8", mode="r"))
#
# # 回答问题
# system_prompt = "\n".join([data["prompt"]["question_template-0"],
#                            data["prompt"]["answer_template"]])
# prompt = data["question"][0][0]
# print("=" * 20, "回答问题")
# print("[SYS]", system_prompt)
# print("[USER]", prompt)
# res, ppl = inference(prompt, "deepseek-chat", system_prompt, client)
# print(res, ppl, end="\n\n\n\n")
# 背景 + 回答问题
# print("=" * 20, "背景 + 回答问题")
# system_prompt = "\n".join([data["prompt"]["question_template-1"].replace("<<reference>>", data["textbook"]),
#                            data["prompt"]["answer_template"],
#                            ])
# prompt = data["question"][0][0]
# print("[SYS]", system_prompt)
# print("[USER]", prompt)
# res, ppl = inference(prompt, "deepseek-chat", system_prompt, client)
# print(res, ppl, end="\n\n\n\n")
# # 背景 + 笔记 + 回答问题
# print("=" * 20, "背景 + 笔记 + 回答问题")
# system_prompt = "\n".join([data["prompt"]["notion_template"]])
# prompt = data["textbook"]
# print("[SYS]", system_prompt)
# print("[USER]", prompt)
# res, ppl = inference(prompt, "deepseek-chat", system_prompt, client, is_json=False)
# print("=" * 20, "[NOTION")
# print(res, ppl)
# print("=" * 20, "NOTION]")
# system_prompt = "\n".join([data["prompt"]["question_template-1"].replace("<<reference>>", res),
#                            data["prompt"]["answer_template"]])
# prompt = data["question"][0][0]
# print("[SYS]", system_prompt)
# print("[USER]", prompt)
# res, ppl = inference(prompt, "deepseek-chat", system_prompt, client)
# print(res, ppl)