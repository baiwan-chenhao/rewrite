from json import JSONDecodeError
from datetime import datetime

from openai import RateLimitError, BadRequestError, InternalServerError, PermissionDeniedError, APIConnectionError
import anthropic
import subprocess
import os
from datetime import datetime
import json
from abc import abstractmethod
from retry import retry

from forging_utils.coder.smartassert import smart_assert


def get_now():
    return datetime.now().strftime('%d-%m-%y-%H-%M-%S-%f')


@retry(exceptions=(RateLimitError, JSONDecodeError, InternalServerError, APIConnectionError), tries=2, delay=60, backoff=2, max_delay=120)
def openai_inference(client, prompt, system_prompt, model_name, prefix=None, is_coding=False):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    smart_assert(chat_completion.choices[0].finish_reason == "stop", chat_completion.choices[0].finish_reason)
    reply = chat_completion.choices[0].message.content
    if is_coding:
        reply = request_check(reply)
    return reply


@retry(exceptions=(RateLimitError, JSONDecodeError), tries=2, delay=60, backoff=2, max_delay=120)
def deepseek_inference(client, prompt, system_prompt, model_name, prefix=None):
    """
    :param prompt:
    :param system_prompt:
    :param model_name:
    :param prefix: 是否为对话续写。
    :return:
    """
    if prefix is None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        stop = None
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": prefix, "prefix": True},
        ]
        stop = ["```"]
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        stop=stop,
    )
    reply = chat_completion.choices[0].message.content
    return reply


class BaseMemery:
    def __init__(self, load_path=None):
        if load_path is None or not os.path.isfile(load_path):
            self.memory = None  # 子类根据各自的情况进行覆盖。
            self.memory_origin = []
            self.memory_ids = []
            self.measure = []  # 用来衡量学习效率
        else:
            with open(load_path, 'r', encoding="utf-8") as file:
                data = json.load(file)
                for k, v in data.items():
                    setattr(self, k, v)
        self.attrs = ["memory", "memory_origin", "measure"]
        # self.memory = data['memory']
        # self.memory_origin = data['memory_origin']
        # self.measure = data['measure']
        # self.memory_ids = data['memory_ids']

    def register(self, keys):
        assert isinstance(keys, list), "`keys` must be `list`"
        self.attrs.extend(keys)

    def memorise(self, text: str, text_origin: str, text_ids: str):
        self.memory_origin.append(text_origin)
        # self.memory_ids.append(text_ids)

    def has_memorised(self, text_ids: str):
        return False

    def save(self, path: str):
        dumps = dict()
        for key in self.attrs:
            dumps[key] = getattr(self, key, "")
        with open(path, 'w', encoding="utf-8") as file:
            json.dump(dumps, file, indent=4, ensure_ascii=False)

    @abstractmethod
    def size(self):
        pass

    def as_memorised(self, text):
        return str(self) + text

    @abstractmethod
    def __str__(self):
        pass


class Memory(BaseMemery):
    def __init__(self, load_path=None):
        """
        记忆采用分块储存的模式。还会储存记忆的源。
        :param capacity: 记忆的最大容量。
        """
        super().__init__(load_path)
        if self.memory is None:
            self.memory = list()

    def memorise(self, text: str, text_origin: str, text_ids: str):
        assert len(text_origin) > 0
        self.memory.append(text)
        self.memory_origin.append(text_origin)
        self.measure.append((len(text), len(text_origin)))
        self.memory_ids.append(text_ids)

    def has_memorised(self, text_ids):
        return text_ids in self.memory_ids

    def effectiveness(self):
        return list(map(lambda x: x[0] / x[1], self.measure))

    def size(self):
        return sum(map(len, self.memory))

    def __str__(self):
        return "".join(self.memory)

    def __len__(self):
        return sum(map(len, self.memory))

    def __getitem__(self, item):
        return self.memory[item]

    def __iter__(self):
        return iter(self.memory)





def enum_path(path, suffix="", prefix=""):
    if isinstance(path, dict):
        for k, v in path.items():
            if isinstance(v, dict):
                for i in enum_path(v):
                    yield prefix + k + "/" + i + suffix
            elif isinstance(v, list):
                for i in v:
                    if isinstance(i, str):
                        yield prefix + k + "/" + str(i) + suffix
                    elif isinstance(i, dict):
                        for j in enum_path(i):
                            yield prefix + k + "/" + j + suffix
                    else:
                        raise TypeError
            elif isinstance(v, str):
                yield prefix + k + "/" + v + suffix
            else:
                raise TypeError
    else:
        raise TypeError


@retry(exceptions=(RateLimitError, BadRequestError, InternalServerError), tries=2, delay=60, backoff=2, max_delay=120)
def qwen_inference(client, prompt, system_prompt, model_name, prefix=None, is_coding=False):
    """
    :param prompt:
    :param system_prompt:
    :param model_name:
    :param dump_dir:
    :param filename:
    :param prefix: 是否为对话续写。
    :param prefix_key: 对话续写的键。对deepseek为prefix,对千问是partial。
    :return:
    """
    if prefix is None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        stop = None
    else:
        # assert model_name[:4] == "qwen", "only support qwen model"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": prefix, "partial": True},
        ]
        if model_name[:4] == "qwen":
            stop = ["```"]
        else:
            stop = None
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        stop=stop,
    )
    reply = chat_completion.choices[0].message.content
    if is_coding:
        reply = stract_code(reply, prefix)
    return reply


@retry(exceptions=(PermissionDeniedError, APIConnectionError), tries=2, delay=60, backoff=2, max_delay=120)
def byte_inference(client, prompt, system_prompt, model_name, prefix=None, is_coding=False):
    msg = [{"role": "system", "content": system_prompt},
           {"role": "user", "content": prompt}]
    if prefix:
        msg.append({"role": "assistant", "content": prefix})
        stop = ["```"] if model_name[:6] == "doubao" else None
    else:
        stop = None
    model_name = {
        "deepseek-v3": "ep-20250214080055-97fjf",
        "deepseek-v3-compared": "ep-20250214080055-97fjf",

        "doubao-1.5-pro-32k-250115": "ep-20250214082336-n2bds",
        "doubao-pro-256k-241115_0214": "ep-20250214113122-8l8t7",
        "doubao-1.5-pro-256k-250115_0214": "ep-20250214135521-98987",
        "doubao-1.5-pro-256k-250115_0214-compared": "ep-20250214135521-98987",
        "deepseek-r1": "ep-20250215093546-fgfrp"
    }[model_name]
    completion = client.chat.completions.create(
        model=model_name,
        messages=msg,
        stop=stop,
        # temperature=0.2
    )
    choice = completion.choices[0]
    smart_assert(choice.finish_reason == "stop", choice.finish_reason)
    # print(choice.message.reasoning_content)
    reply = choice.message.content
    if is_coding:
        reply = request_check(reply)
    return reply


@retry(exceptions=(anthropic.InternalServerError, anthropic.APITimeoutError), tries=4, delay=60, backoff=2, max_delay=120)
def anthropic_inference(client, prompt, system_prompt, model_name, prefix=None, is_coding=False):
    msg = [
        {"role": "user", "content": prompt}
    ]
    stop = None
    # if prefix is not None:  Not support.
    #     msg.append({"role": "assistant", "content": prefix})
    #     stop = ["```"]

    completion = client.messages.create(
        system=system_prompt,
        max_tokens=4096,
        model=model_name,
        messages=msg,
        stop_sequences=stop
    )
    reply = completion.content[0].text
    assert completion.stop_reason in ["end_turn"], completion.stop_reason
    if is_coding:
        reply = request_check(reply)
    return reply


def stract_code(reply):
    import re
    match = re.search(pattern=f"```.*?\n(.*?)```", string=reply, flags=re.DOTALL)
    if match:
        reply = match.group(1)
    return reply


from evaluator import CJCodeEvaluator

cj_coder = CJCodeEvaluator()

def request_check(reply):
    if cj_coder.is_valid_code(reply):
        return reply
    if reply[:4] == "func" and reply.find("```") != -1:
        return reply[:reply.find("```")]
    if stract_code(reply) != reply:
        return stract_code(reply)
    reply = reply.replace("`", "")
    # never reach.
    # with open("tmp.txt", "w", encoding="utf-8") as f:
    #     f.write(reply)
    # os.system("notepad tmp.txt")
    # with open("tmp.txt", "r", encoding="utf-8") as f:
    #     reply = f.read()
    return reply





# class CCodeEvaluator(CodeEvaluator):
#     def run_code(self, code, warpped=True):
#         """
#         :param code:
#         :param warpped: 是否包裹代码。默认输入的`code`是
#         :return:
#         """
#         with open("CodeEvaluating.cpp", "w", encoding='utf-8') as f:
#             f.write(code)
#         args = [r"D:\app\mingw64\bin\gcc", "CodeEvaluating.cpp", "-oCodeEvaluating.exe"]
#         completed = self.run_with_timeout(args)
#         if os.path.isfile("CodeEvaluating.cpp"):
#             os.remove("CodeEvaluating.cpp")
#         if completed.returncode == 0:
#             completed1 = subprocess.run("CodeEvaluating.exe", capture_output=True, shell=False, timeout=self.timeout)
#             if completed1.returncode == 0:
#                 self.result = completed1.stdout.decode("gbk")
#                 self.failed = False
#             else:
#                 self.result = completed1.stderr.decode("gbk")
#                 self.failed = True
#         else:
#             self.failed = True
#             self.result = completed.stderr.decode("gbk")
#         if os.path.isfile("CodeEvaluating.exe"):
#             os.remove("CodeEvaluating.exe")
#
#     def __str__(self):
#         return f"result:{self.result}\nfailed:{self.failed}"


if __name__ == "__main__":
    # for n in range(1, 5):
    #     ls = [1,2,3,4]
    #     ls = ls[ls.index(n) + 1:] + ls[0:ls.index(n)]
    #     print(ls)
        py_coder = CJCodeEvaluator(timeout=4)
        py_coder.run("""
func performAction(action: () -> Unit) {
    println("Performing action...")
    action()
}

// 使用尾随 Lambda 调用函数

 main(){
 println("Fg")
 performAction {
    println("Action completed!")
}
 }
    """)
        print(py_coder.result)
