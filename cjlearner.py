import json
import re
import time

from anthropic import Anthropic
from openai import OpenAI

from coder.smartassert import smart_assert
from keys import OPENAI_KEY

from forging_utils import JsonSerialize
from abc import ABC, abstractmethod
import os
from jinja2 import Template, Environment, StrictUndefined
from evaluator import CJCodeEvaluator
from utils import stract_code, byte_inference, anthropic_inference, openai_inference
from agent_learning import learning_dict

focused_learning_book = [
    "dataset/cangjie开发指南/cangjie/基本概念/标识符.md",
    "dataset/cangjie开发指南/cangjie/基本概念/程序结构.md",
    "dataset/cangjie开发指南/cangjie/基本概念/表达式.md",
    "dataset/cangjie开发指南/cangjie/基本概念/函数.md",
    "dataset/cangjie开发指南/cangjie/基础数据类型/整数类型.md",
    "dataset/cangjie开发指南/cangjie/基础数据类型/浮点类型.md",
    "dataset/cangjie开发指南/cangjie/基础数据类型/布尔类型.md",
    "dataset/cangjie开发指南/cangjie/基础数据类型/字符类型.md",
    "dataset/cangjie开发指南/cangjie/基础数据类型/字符串类型.md",
    "dataset/cangjie开发指南/cangjie/基础数据类型/元组类型.md",
    "dataset/cangjie开发指南/cangjie/基础数据类型/数组类型.md",
    "dataset/cangjie开发指南/cangjie/基础数据类型/区间类型.md",
    "dataset/cangjie开发指南/cangjie/基础数据类型/Unit类型.md",
    "dataset/cangjie开发指南/cangjie/基础数据类型/Nothing类型.md",
    "dataset/cangjie开发指南/cangjie/函数/定义函数.md",
    "dataset/cangjie开发指南/cangjie/函数/调用函数.md",
    "dataset/cangjie开发指南/cangjie/函数/函数类型.md",
    "dataset/cangjie开发指南/cangjie/函数/嵌套函数.md",
    "dataset/cangjie开发指南/cangjie/函数/Lambda 表达式.md",
    "dataset/cangjie开发指南/cangjie/函数/闭包.md",
    "dataset/cangjie开发指南/cangjie/函数/函数调用语法糖.md",
    "dataset/cangjie开发指南/cangjie/函数/函数重载.md",
    "dataset/cangjie开发指南/cangjie/函数/操作符重载.md",
    "dataset/cangjie开发指南/cangjie/函数/const 函数和常量求值.md",
    "dataset/cangjie开发指南/cangjie/结构类型/定义struct类型.md",
    "dataset/cangjie开发指南/cangjie/结构类型/创建struct实例.md",
    "dataset/cangjie开发指南/cangjie/结构类型/mut函数.md",
    "dataset/cangjie开发指南/cangjie/枚举类型和模式匹配/枚举类型.md",
    "dataset/cangjie开发指南/cangjie/枚举类型和模式匹配/Option类型.md",
    "dataset/cangjie开发指南/cangjie/枚举类型和模式匹配/模式概述.md",
    "dataset/cangjie开发指南/cangjie/枚举类型和模式匹配/模式的 Refutability.md",
    "dataset/cangjie开发指南/cangjie/枚举类型和模式匹配/match表达式.md",
    "dataset/cangjie开发指南/cangjie/枚举类型和模式匹配/if-let表达式.md",
    "dataset/cangjie开发指南/cangjie/枚举类型和模式匹配/while-let表达式.md",
    "dataset/cangjie开发指南/cangjie/枚举类型和模式匹配/其他使用模式的地方.md",
    "dataset/cangjie开发指南/cangjie/类和接口/类.md",
    "dataset/cangjie开发指南/cangjie/类和接口/接口.md",
    "dataset/cangjie开发指南/cangjie/类和接口/属性.md",
    "dataset/cangjie开发指南/cangjie/类和接口/子类型关系.md",
    "dataset/cangjie开发指南/cangjie/类和接口/类型转换.md",
    "dataset/cangjie开发指南/cangjie/Collection类型/基本Collection类型概述.md",
    "dataset/cangjie开发指南/cangjie/Collection类型/ArrayList.md",
    "dataset/cangjie开发指南/cangjie/Collection类型/HashSet.md",
    "dataset/cangjie开发指南/cangjie/Collection类型/HashMap.md",
    "dataset/cangjie开发指南/API/std模块/collection包/LinkedList.md",
    "dataset/cangjie开发指南/cangjie/Collection类型/Iterable 和 Collections.md",
    "dataset/cangjie开发指南/cangjie/异常处理/定义异常.md",
    "dataset/cangjie开发指南/cangjie/异常处理/throw和异常处理.md",
    "dataset/cangjie开发指南/cangjie/异常处理/常见运行时异常.md",
    "dataset/cangjie开发指南/cangjie/异常处理/使用Option.md",
    "dataset/cangjie开发指南/仓颉语言结构体/String/String.md",
    "dataset/cangjie开发指南/仓颉语言结构体/String/String1.md",
    "dataset/cangjie开发指南/仓颉语言结构体/String/String2.md",
    "dataset/cangjie开发指南/仓颉语言结构体/Range/Range.md",
    "dataset/cangjie开发指南/cangjie/Collection类型/实例教程/ArrayList使用示例.md",
    "dataset/cangjie开发指南/cangjie/Collection类型/实例教程/HashMap使用示例.md",
    "dataset/cangjie开发指南/cangjie/Collection类型/实例教程/迭代器操作函数.md",
    "dataset/cangjie开发指南/仓颉语言结构体/String/StringBuilder.md",
    "dataset/cangjie开发指南/cangjie/Collection类型/实例教程/HashSet.md",
    "dataset/cangjie开发指南/API/std模块/math包/接口1.md",
    "dataset/cangjie开发指南/API/std模块/math包/接口2.md",
    "dataset/cangjie开发指南/API/std模块/math包/函数.md",
    "dataset/cangjie开发指南/仓颉语言结构体/Array/Array.md",
    "dataset/cangjie开发指南/仓颉语言结构体/Array/Array1.md",
    "dataset/cangjie开发指南/API/std模块/collection包/函数.md",
    "dataset/cangjie开发指南/API/std模块/collection包/函数1.md",
    "dataset/cangjie开发指南/cangjie/基础数据类型/类型转换.md",
    "dataset/cangjie开发指南/API/std模块/sort.md",
    "dataset/cangjie开发指南/API/std模块/collection包/ArrayList1.md",
    "dataset/cangjie开发指南/API/std模块/collection包/ArrayList2.md"
]


def path_replace(path, filename):
    path = os.path.join(os.path.split(path)[0], filename)
    if not os.path.isfile(path):
        raise FileNotFoundError
    return path


def path_join(path, filename):
    path = os.path.join(path, filename)
    if not os.path.isfile(path):
        print(path)
        raise FileNotFoundError
    return path


def prompt_load(filename, with_meta: bool = False, **kw):
    """
    load prompt from path/filename.
    :param filename:
    :param with_meta:
    :param kw:
    :return:
    """
    with open(filename, encoding="utf-8") as f:
        text = f.read()
    if with_meta:
        with open(path_replace(filename, "rewrite_meta.txt"), encoding="utf-8") as f:
            text = f.read() + text
    env = Environment(undefined=StrictUndefined)
    tmpl = env.from_string(text)
    # tmpl = Template(text)
    res = tmpl.render(**kw)
    # if "{{" in res and "}}" in res:
    #     print(res)
    #     raise Exception("there are unredered params")
    return res


class BaseMemory(JsonSerialize, ABC):
    def __init__(self, load_path, **added_keys):
        """
        load status from *load_path*.
        :param load_path:
        :param added_keys: keys added by subclass.
        """
        JsonSerialize.__init__(self, load_path)
        JsonSerialize.set_key(self, "memory", self.init_mem())  # 记忆
        JsonSerialize.set_key(self, "has_mem_list", list())
        JsonSerialize.set_key(self, "learned_len", 0)  # 已学习的总长度
        for k, v in added_keys.items():
            JsonSerialize.set_key(self, k, v)

    @abstractmethod
    def init_mem(self):
        pass

    @abstractmethod
    def __len__(self):
        """
        return the size of *mem_list*,used to measure memory usage.
        :return:
        """
        pass

    @abstractmethod
    def __str__(self):
        """
        used to transform the mem to prompt.
        :return:
        """
        pass

    def memorize(self, notepage, textpage, textid):
        """
        memorize the note *notepage* learned from *textpage*.
        the *textid* is used to identify the note.

        :param notepage:
        :param textpage:
        :param textid:
        :return:
        """
        if textid:
            if textid not in self.has_mem_list:
                self.has_mem_list.append(textid)
                self.learned_len += len(textpage)

    def has_memorized(self, textid):
        return textid in self.has_mem_list


class SegmentMemory(BaseMemory):
    freezed_items = ["基本输入输出", "循环语句", "函数定义", "基本数据类型", "标识符", "表达式", "struct数据类型",
                     "枚举类型与模式匹配", "类与接口", "泛型", "Collection类型", "String数据类型",
                     "std.math"]  # difficulty added

    def __init__(self, load_path, agent, prompt_dir,
                 relations=("相同", "相似", "无关"), note_classes_restraint=5, splitor=",", **kwargs):
        super().__init__(load_path, focus_mem=list(), path_analyis=list(), note_len=0, error_classify=0, **kwargs)
        self.agent = agent
        # prompt path
        self.prompt_dir = prompt_dir

        self.note_path_gen_path = path_join(prompt_dir, "note_path_gen.txt")
        self.note_merge_path = path_join(prompt_dir, "note_merge.txt")
        self.note_relation_gen_path = path_join(prompt_dir, "note_relation_gen.txt")
        self.rewrite_meta_path = path_join(prompt_dir, "rewrite_meta.txt")
        self.query_path = path_join(prompt_dir, "query.txt")

        self.memory: dict
        self.focus_mem: list
        self.path_analysis: list
        self.note_len: int = 0  # 笔记长度
        assert len(relations) == 3
        self.relations = relations
        self.note_classes_restraint = note_classes_restraint
        self.splitor = splitor
        self.reflection_path = "代码反思"
        self.prequeried_item = ["基本输入输出", "基本数据类型", "标识符", "表达式", self.reflection_path]

    def query(self, question, return_keys=False):
        keys = self.agent.openai_inference(prompt=prompt_load(self.query_path,
                                                              question=question, classes=self.classes_gen()))
        keys = list(set(keys.split(self.splitor) + self.prequeried_item))
        keys = [key.replace("\n", "") for key in keys]  # added for r1
        if return_keys:
            return keys
        self.agent.print_d(keys, title="Queried keys")
        res = ""
        for key in keys:
            res += f"<{key}>" + "".join(self.memory.get(key, [])) + f"</{key}>\n"
        return res

    def classes_gen(self):
        return self.splitor.join(self.memory.keys())

    def init_mem(self):
        return {k: list() for k in self.freezed_items}

    def load(self):
        super().load()
        for k in self.freezed_items:
            if k not in self.memory:
                self.memory[k] = list()

    def memorize(self, notepage, textpage, textid, path=None):
        BaseMemory.memorize(self, notepage, textpage, textid)
        if path is None:
            path = self.agent.openai_inference(
                prompt_load(self.note_path_gen_path, note=notepage, types=self.classes_gen())).replace("\n", "")
        res = path + " "
        if path in self.memory:
            has_memorised = False
            for p, item in enumerate(self.memory[path]):
                if notepage == item:
                    relation = self.relations[0]
                else:
                    relation = self.agent.openai_inference(prompt_load(self.note_relation_gen_path,
                                                                       note1=notepage, note2=item, with_meta=False))
                    relation = relation.replace("\n", "")
                if relation == self.relations[0]:  # 相同
                    has_memorised = True
                    res += "has same"
                    break
                elif relation == self.relations[1]:  # 相似
                    res += "merged"
                    new_note = self.agent.openai_inference(
                        prompt_load(self.note_merge_path, note1=notepage, note2=item, with_meta=False))
                    self.memory[path][p] = new_note
                    self.note_len = self.note_len + len(new_note) - len(item)
                    has_memorised = True
                    break
                elif relation == self.relations[2]:  # 不同
                    pass
                else:  # error will be viewed as difference.
                    self.agent.print_d("错误的分类结果", repr(relation))
                    self.error_classify += 1
            if not has_memorised:
                res += "appended"
                self.memory[path].append(notepage)
                self.note_len += len(notepage)
        else:
            res += "new"
            self.memory[path] = [notepage]
            self.note_len += len(notepage)

        if textid != self.reflection_path:
            self.path_analyis.append((textid, res))
        return res

    def __len__(self):
        return len(str(self))

    def compacity(self):
        return self.note_len / self.learned_len

    def __str__(self):
        return "".join(["".join(i) for i in self.memory.values()]) + "".join(self.focus_mem)


CODING_TIPS = """仓颉语言小知识

内置类型的最大值、最小值：type.Max, type.Min。如Int64.Max, Int64.Min。

三元运算符：Int64 a = if(true){1} else {2}

创建数组：
let a = Array<Int64>() // Created an empty Array whose element type is Int64
let b = Array<Int64>(a) // Use another Array to initialize b
let c = Array<Int64>(3, item: 0) // Created an Array whose element type is Int64, length is 3 and all elements are initialized as 0
let d = Array<Int64>(3, {i => i + 1}) // Created an Array whose element type is Int64, length is 3 and all elements are initialized by the initialization function
var dp = Array<Array<Int64>>(n + 1, {_ => Array<Int64>(m + 1, {_ => 0})})

从HashMap中取值：
Int64 v = if(map.contains(v)){map[v]}else{default}

堆栈的实现
使用LinkedList<T>类型
添加元素：append(element);弹出元素：popLast()。
ArrayList类型可以自由添加删除元素，Array的长度是不可变的。

for循环遍历:for(i in 0..5:2) //步长为2
{println(i)}//0 2 4"""


class CJLearner:
    def __init__(self, model_name, infra, dump_dir, load_path, debug=False, language="chinese",
                 memory=None, prompt_dir=None, external_mem=None,
                 # 消融实验参数
                 using_note_generating=True, using_note_querying=True
                 ):
        """
        :param client: used to request.
        :param dump_dir: used to record all the request info.
        :param load_path: used to load mem.
        :param debug: weather open debug mode.
        :param prompt_dir: dir of prompt.
        """
        assert using_note_generating or using_note_querying, "ablation only changes one."
        self.using_note_generating = using_note_generating
        self.using_note_querying = using_note_querying
        self.in_ablation = not (using_note_querying and using_note_generating)


        self.model_name = model_name
        if infra in ("byte", "openai", "ali"):
            self.client = OpenAI(**OPENAI_KEY[infra])
        elif infra == "anthropic":
            self.client = Anthropic(**OPENAI_KEY["anthropic"])
        else:
            raise Exception("wrong infra: " + infra)
        self.infra = infra
        self.dump_dir = dump_dir  # is implemented to a file.
        if os.path.isfile(self.dump_dir):
            with open(self.dump_dir, mode="rb") as file:
                self.dump_data = json.load(file)
        else:
            self.dump_data = dict()  # learning order is mentioned in index file. So dict() is enough.
        self.load_path = load_path

        self.debug = debug
        if memory is None:
            prompt_dir = f"prompts/{model_name}/{language}"
            self.memory = SegmentMemory(load_path=load_path, agent=self, prompt_dir=prompt_dir,
                                        requested_token_num=0,  # token used when learning.
                                        reflection_token_num=0  # reflection expend. learning expend is sub of them.
                                        )
        else:
            self.memory = memory
            assert prompt_dir is not None, "`prompt_dir` must be provided with `memory`"
            self.prompt_dir = prompt_dir
        self.external_mem = external_mem
        if os.path.isfile(load_path):
            self.memory.load()
        self.note_gen_path = path_join(prompt_dir, "note_gen.txt")
        self.code_gen_filename = path_join(prompt_dir, "code_gen.txt")
        self.code_reflection_filename = path_join(prompt_dir, "code_reflection.txt")
        self.rewrite_meta = path_join(prompt_dir, "rewrite_meta.txt")

        self.evaluator = CJCodeEvaluator()
        self.token_count = 0  # token used in a period time. Zero it using `token_count_zero` func.

    def token_count_zero(self):
        self.token_count = 0

    def learn(self, textpage, textid, path):
        if self.memory.has_memorized(textid):
            print("has learned", textid)
            return
        if self.using_note_generating:
            notepage = self.openai_inference(
                prompt_load(self.note_gen_path, text=textpage, memory="", with_meta=True))
        else:
            notepage = textpage
        self.print_d(self.memory.memorize(notepage, textpage, textid, path=path))
        self.save()

    def openai_inference(self, prompt, sys_prompt="You are a helpful assistant.", prefix=None, is_coding=False):
        """
        client, prompt, sys_prompt --> response
        :param prompt:
        :param sys_prompt:
        :param prefix: used in code gen. code gen always gen.
        :param cached: is chached. if True and prefix is None, will used cached response.
        :return:
        """
        key = self.model_name + prompt + sys_prompt
        if not is_coding and key in self.dump_data:  # has learned.
            reply = self.dump_data[key]
        else:  # learning and coding
            reply = self.inference(prompt=prompt, system_prompt=sys_prompt, prefix=prefix, is_coding=is_coding)
            self.dump_data[key] = reply
            if not prefix:
                self.memory.requested_token_num += len(prompt) + len(sys_prompt) + len(reply)
        self.token_count += len(prompt) + len(sys_prompt) + len(reply)
        return reply

    def focus_learned_check(self):
        will_assert = False
        unlearning_list = list()
        for i in focused_learning_book:
            if not self.memory.has_memorized(i):
                unlearning_list.append(i)
                will_assert = True
        if will_assert:
            for i in unlearning_list:
                print(i)
        assert not will_assert

    def code_gen(self, task, is_cangjie=True, try_number=3, tips=CODING_TIPS, using_external_mem=False,
                 return_keys=False, times=None):
        # times用于测试框架多次执行，无实际作用。
        self.focus_learned_check()
        time_start = time.perf_counter()

        if return_keys:  # internal analysis for memory
            assert is_cangjie
            return self.memory.query(f"请用仓颉语言编写如下任务：" + task, return_keys=True)

        def add_parentheses_to_code(code):
            # code = re.sub(r'for\s+(\w+\s+in\s+[^{]+)', r'for (\1)', code)
            for_pattern = re.compile(r'for\s+([^\{]+)\s*\{')
            def wrap_for(match):
                condition = match.group(1).strip()
                if not condition.startswith('(') and not condition.endswith(')'):
                    return f'for({condition}) {{'
                elif condition.startswith("(") and not condition.endswith(")"):
                    return f"for {condition}) {{"
                elif not condition.startswith("(") and condition.endswith(")"):
                    return f"for ({condition} {{"
                return f'for {condition} {{'
            code = for_pattern.sub(wrap_for, code)

            code = re.sub(r'if\s+([^{]+)', r'if (\1)', code)
            code = re.sub(r'while\s+([^{]+)', r'while (\1)', code)

            # 1..<n --> 1..n
            pattern = re.compile(r"in\s+(\d+|\w+)\.\.<(\w+)")
            code = pattern.sub(r"in \1..\2", code)

            # 字符串比较
            pattern = re.compile(r"==\s*'([^']*)'")
            def replace_match(match):
                char_content = match.group(1)  # 获取字符内容
                return f"== UInt8(UInt32(r'{char_content}'))"  # 替换为目标形式
            code = pattern.sub(replace_match, code)

            code = code.replace("]!", "]")  # !好像没在代码中用过
            code = code.replace(")!", ")")
            return code

        if is_cangjie:
            assert "仓颉" in task, "用仓颉编程时请在task中指明。"
        else:
            try_number = 1
        assert try_number > 0
        evaluator = self.evaluator
        self.token_count_zero()
        code, origin_code = "", ""
        retrieve_token = 0  # mem token used.
        tried_times = None  # try times.
        reflection = list()
        results_ret = list()
        if is_cangjie:
            if using_external_mem:
                with open(using_external_mem, encoding="utf-8", mode="r") as f:
                    mem = f.read()
            else:
                if self.using_note_querying:
                    mem = self.memory.query(f"请用仓颉语言编写如下任务：" + task)
                else:
                    mem = str(self.memory)
        else:
            mem = ""
        for i in range(try_number):
            tried_times = i + 1
            retrieve_token += len(mem)
            if is_cangjie:
                prompt = prompt_load(self.code_gen_filename, with_meta=True, memory=mem, task=task)
                prompt += tips
            else:
                prompt = task
            code = self.openai_inference(prompt=prompt,
                                         prefix="```cangjie\n" if is_cangjie else "",
                                         is_coding=True)
            origin_code = code
            if is_cangjie:
                code = add_parentheses_to_code(code)
            self.print_d(code, title="code")
            if not is_cangjie:
                break
            # ==== reflection
            evaluator.run(code)
            if evaluator.failed:
                results = evaluator.result
                results_ret.append(results)
                self.print_d(results, title="result")
                # mem = self.memory.query(f"请用{language}语言编写如下任务：" + task)
                # retrieve_token += len(mem)
                # if "is not a member of" in results:
                #     api_retrieve = "在仓颉语言中，请给我一个%s的示例代码" %
                code_reflection = self.openai_inference(
                    prompt=prompt_load(self.code_reflection_filename, task=task, code=code, results=results),
                    sys_prompt=prompt_load(self.rewrite_meta, memory=mem))
                reflection.append(code_reflection)
                # self.memory.reflection_token_num += len(code_reflection)
                self.print_d(i, code_reflection, title="reflection")
                # self.memory.memorize(code_reflection, code_reflection, "None", path=self.memory.reflection_path)
                mem += code_reflection
            else:
                break
        if not evaluator.failed and tried_times > 1:
            self.memory.reflection_token_num += len(code_reflection)
            self.memory.memorize(code_reflection, code_reflection, "None", path=self.memory.reflection_path)
            self.memory.memorize(code, code, "None", path=self.memory.reflection_path)
            self.print_d("saving===========================@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            self.save()
        res = {
            "code": code,  # gened code.
            "origin_code": origin_code,
            "retrieve_token": retrieve_token,  # len of retrieved mem.
            "token_used": self.token_count,  # all the token used when solve this task.
            "tried_times": tried_times,  # tried times solving this code.
            "reflection": reflection,
            "results": results_ret,
            "time_used": time.perf_counter() - time_start
        }
        self.token_count_zero()
        return res

    def talk(self, hello):
        mem = self.memory.query(hello)
        prompt = prompt_load(self.rewrite_meta, memory=mem) + hello
        return self.openai_inference(prompt)

    def save(self):
        self.memory.save()
        with open(self.dump_dir, mode="w") as file:
            json.dump(self.dump_data, file)

    def print_d(self, *d, end="\n", seq=" ", title=None):
        if title is not None and self.debug:
            print('\033[1;32m' + f"<{title}>\n" + '\033[0m', end="")
        if self.debug:
            print('\033[1;32m' + seq.join(map(str, d)) + '\033[0m', end=end)
        if title is not None and self.debug:
            print('\033[1;32m' + f"\n</{title}>" + '\033[0m', end="")

    def inference(self, prompt, system_prompt, prefix, is_coding=False):
        raise NotImplemented
        # return {
        #     "byte": byte_inference
        # }[self.infra](self.client, prompt, system_prompt, self.model_name, prefix)


class CJLearnerForDeepseekV3(CJLearner):
    model_name = "deepseek-v3"

    def __init__(self, infra, dump_dir, load_path, debug=False, language="chinese", external_mem=None,
                 using_note_generating=True, using_note_querying=True):
        CJLearner.__init__(self, self.model_name, infra, dump_dir, load_path, debug, language,
                           external_mem=external_mem,
                           using_note_generating=using_note_generating, using_note_querying=using_note_querying)

    def learn_book(self):
        learning_dict(self, book_path="dataset/cangjie开发指南/", index=book_index_simple)
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "API": {
                "std模块": ["sort"]
            }
        }, path="数组排序"),
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "API": {
                "std模块": {
                    "math包": ["接口1", "接口2", "函数"]
                }
            }
        }, path="std.math"),
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "API": {
                "std模块": {
                    "collection包": ["函数", "函数1", "ArrayList1", "ArrayList2", "LinkedList"]
                }
            }
        }, path="Collection类型"),
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "仓颉语言结构体": {
                "Array": ["Array", "Array1"]
            }
        }, path="Array"),
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "仓颉语言结构体": {
                "String": ["String", "String1", "String2", "StringBuilder"]
            }
        }, path="String数据类型"),
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "仓颉语言结构体": {
                "Range": ["Range"]
            }
        }),
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "cangjie": {
                "Collection类型": {
                    "实例教程": ["ArrayList使用示例", "HashMap使用示例", "迭代器操作函数", "HashSet"]
                }
            }
        }),

    def chat(self):
        hello = input(">>> ")
        msg = list()
        if hello:
            mem = self.memory.query(hello)
            prompt = prompt_load(self.rewrite_meta, memory=mem) + hello
            msg.append({
                "role": "system",
                "content": "You are a helpful assistant."
            })
            msg.append({
                "role": "user",
                "content": prompt
            })
            while hello:
                reply = self.openai_chat_inference(msg)
                print(reply)
                msg.append({
                    "role": "assistant",
                    "content": reply
                })
                hello = input(">>> ")
                msg.append({
                    "role": "user",
                    "content": hello
                })

    def openai_chat_inference(self, msg):
        model_name = "ep-20250214080055-97fjf"
        completion = self.client.chat.completions.create(
            model=model_name,
            messages=msg,
        )
        choice = completion.choices[0]
        smart_assert(choice.finish_reason == "stop", choice.finish_reason)
        # print(choice.message.reasoning_content)
        reply = choice.message.content
        return reply


    def inference(self, prompt, system_prompt, prefix, is_coding=False):
        if self.infra == "byte":
            return byte_inference(self.client, prompt, system_prompt, self.model_name, prefix, is_coding)
        else:
            raise Exception("not supported")


class CJLearnerForDeepseekR1(CJLearner):
    """
    r1有循环语句学习困难症。就是学不会循环语句。
    """
    model_name = "deepseek-r1"

    def __init__(self, infra, dump_dir, load_path, debug=False, language="chinese", external_mem=None):
        CJLearner.__init__(self, self.model_name, infra, dump_dir, load_path, debug, language,
                           external_mem=external_mem)


class CJLearnerForClaude3_haiku(CJLearner):
    """
    cluade分类分不明白，见log。需要强制分类。
    sonnet不能理解学的是仓颉语言（a)，指名道姓让他再学一次。
    """
    model_name = "claude-3-5-sonnet-20240620"  # "claude-3-haiku-20240307"

    def __init__(self, dump_dir, load_path, debug=False, language="chinese", infra=None, external_mem=None):
        prompt_dir = f"prompts/cluade-haiku/{language}"
        memory = SegmentMemory(load_path=load_path, agent=self, prompt_dir=prompt_dir,
                               requested_token_num=0,  # token used when learning.
                               reflection_token_num=0  # reflection expend. learning expend is sub of them.
                               )
        CJLearner.__init__(self, self.model_name, infra, dump_dir, load_path, debug, language,
                           memory=memory, prompt_dir=prompt_dir, external_mem=external_mem)


    def inference(self, prompt, system_prompt, prefix, is_coding=False):
        return anthropic_inference(self.client, prompt, system_prompt, self.model_name, prefix, is_coding=is_coding)

    def learn_book(self):
        learning_dict(self, book_path="dataset/cangjie开发指南/", index=book_index_simple)
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "仓颉语言结构体": {
                "String": ["String", "String1", "String2", "StringBuilder"]
            }
        })
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "仓颉语言结构体": {
                "Range": ["Range"]
            }
        }, path="Range")
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "API": {
                "std模块": {
                    "math包": ["接口1", "接口2", "函数"]
                }
            }
        }, path="std.math")
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={"cangjie": {
            "Collection类型": [
                {
                    "实例教程": ["ArrayList使用示例", "HashMap使用示例", "迭代器操作函数", "HashSet"]
                }
            ],
        }}, path="Collection类型")
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "仓颉语言结构体": {
                "Array": ["Array", "Array1"]
            }
        }, path="Array")
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "API": {
                "std模块": [
                    "sort"
                ]
            }
        }, path="数组排序")
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "API": {
                "std模块": [
                    {
                        "collection包": ["函数", "函数1", "ArrayList1", "ArrayList2", "LinkedList"]
                    }
                ],
            }
        }, path="Collection类型")


book_index_simple = {  # 0114
    "cangjie": {
        "基本概念": ["标识符", "程序结构", "表达式", "函数"],
        "基础数据类型": ["整数类型", "浮点类型", "布尔类型", "字符类型", "字符串类型", "元组类型", "数组类型",
                         "区间类型", "Unit类型", "Nothing类型", "类型转换"],
        "函数": ["定义函数", "调用函数", "函数类型", "嵌套函数", "Lambda 表达式", "闭包", "函数调用语法糖",
                 "函数重载",
                 "操作符重载", "const 函数和常量求值"],
        "结构类型": ["定义struct类型", "创建struct实例", "mut函数"],
        "枚举类型和模式匹配": ["枚举类型", "Option类型", "模式概述", "模式的 Refutability", "match表达式",
                               "if-let表达式", "while-let表达式", "其他使用模式的地方"],
        "类和接口": ["类", "接口", "属性", "子类型关系", "类型转换"],
        # "泛型"
        # 扩展
        "Collection类型": ["基本Collection类型概述", "ArrayList", "HashSet", "HashMap", "Iterable 和 Collections"],
        # 包
        "异常处理": ["定义异常", "throw和异常处理", "常见运行时异常", "使用Option"]
    }
}


class CJLearnerForDoubao(CJLearner):
    """
    doubao可能还是注意力不集中。collections学过了，但是还是不会用。手动添加到反思里也不会用。
    中文比英文优秀。
    函数符重载会放错位置。
    """
    model_name = "doubao-1.5-pro-256k-250115_0214"

    def __init__(self, infra, dump_dir, load_path, debug=False, language="chinese", external_mem=None):
        prompt_dir = f"prompts/doubao-1_5-pro/{language}"
        memory = SegmentMemory(load_path=load_path, agent=self, prompt_dir=prompt_dir,
                               requested_token_num=0,  # token used when learning.
                               reflection_token_num=0  # reflection expend. learning expend is sub of them.
                               )
        CJLearner.__init__(self, self.model_name, infra, dump_dir, load_path, debug, language,
                           memory=memory, prompt_dir=prompt_dir,
                           external_mem=external_mem)

    def learn_book(self):
        learning_dict(self, book_path="dataset/cangjie开发指南/", index=book_index_simple)

        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "仓颉语言结构体": {
                "String": ["String", "String1", "String2", "StringBuilder"]
            }
        })
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "仓颉语言结构体": {
                "Range": ["Range"]
            }
        }, path="Range")
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "API": {
                "std模块": {
                    "math包": ["接口1", "接口2", "函数"]
                }
            }
        }, path="std.math")
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={"cangjie": {
            "Collection类型": [
                {
                    "实例教程": ["ArrayList使用示例", "HashMap使用示例", "迭代器操作函数", "HashSet"]
                }],
        }}, path="Collection类型")
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "仓颉语言结构体": {
                "Array": ["Array", "Array1"]
            }
        }, path="Array")
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "API": {
                "std模块": {
                    "collection包": ["函数", "函数1", "ArrayList1", "ArrayList2", "LinkedList"]
                }
            }
        }, path="Collection类型")
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "API": {
                "std模块": ["sort"]
            }
        }, path="数组排序")

    def learn(self, textpage, textid, path):
        if self.memory.has_memorized(textid):
            print("has learned", textid)
            return
        notepage = self.openai_inference(
            prompt_load(self.note_gen_path, text=textpage, memory=self.memory, with_meta=True))
        self.print_d(self.memory.memorize(notepage, textpage, textid, path=path))
        self.save()

    def inference(self, prompt, system_prompt, prefix, is_coding=False):
        if self.infra == "byte":
            return byte_inference(self.client, prompt, system_prompt, self.model_name, prefix, is_coding=is_coding)
        else:
            raise Exception("not supported")


class CJLearnerForGPT4O(CJLearner):
    """
    prompt copied from V3.
    """
    model_name = "gpt-4o-2024-08-06"

    def __init__(self, infra, dump_dir, load_path, debug=False, language="chinese", external_mem=None):
        prompt_dir = f"prompts/gpt-4o/{language}"
        memory = SegmentMemory(load_path=load_path, agent=self, prompt_dir=prompt_dir,
                               requested_token_num=0,  # token used when learning.
                               reflection_token_num=0  # reflection expend. learning expend is sub of them.
                               )
        CJLearner.__init__(self, self.model_name, infra, dump_dir, load_path, debug, language,
                           memory=memory, prompt_dir=prompt_dir, external_mem=external_mem)

    def learn_book(self):
        learning_dict(self, book_path="dataset/cangjie开发指南/", index=book_index_simple)

        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "仓颉语言结构体": {
                "String": ["String", "String1", "String2", "StringBuilder"]
            }
        })
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "仓颉语言结构体": {
                "Range": ["Range"]
            }
        }, path="Range")
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "API": {
                "std模块": {
                    "math包": ["接口1", "接口2", "函数"]
                }
            }
        }, path="std.math")
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={"cangjie": {
            "Collection类型": [
                {
                    "实例教程": ["ArrayList使用示例", "HashMap使用示例", "迭代器操作函数", "HashSet"]
                }],
        }}, path="Collection类型")
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "仓颉语言结构体": {
                "Array": ["Array", "Array1"]
            }
        }, path="Array")
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "API": {
                "std模块": {
                    "collection包": ["函数", "函数1", "ArrayList1", "ArrayList2", "LinkedList"]
                }
            }
        }, path="Collection类型")
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "API": {
                "std模块": ["sort"]
            }
        }, path="数组排序")

    def inference(self, prompt, system_prompt, prefix, is_coding=False):
        return openai_inference(self.client, prompt, system_prompt, self.model_name, prefix, is_coding=is_coding)


class CJLearnerForQwen(CJLearner):
    model_name = "qwen-max-2025-01-25"

    def __init__(self, infra, dump_dir, load_path, debug=False, language="chinese", external_mem=None):
        prompt_dir = f"prompts/0114_qwen/{language}"
        memory = SegmentMemory(load_path=load_path, agent=self, prompt_dir=prompt_dir,
                               requested_token_num=0,  # token used when learning.
                               reflection_token_num=0  # reflection expend. learning expend is sub of them.
                               )
        CJLearner.__init__(self, self.model_name, infra, dump_dir, load_path, debug, language,
                           memory=memory, prompt_dir=prompt_dir,
                           external_mem=external_mem)

    def learn_book(self):
        learning_dict(self, book_path="dataset/cangjie开发指南/", index=book_index_simple)

        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "仓颉语言结构体": {
                "String": ["String", "String1", "String2", "StringBuilder"]
            }
        })
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "仓颉语言结构体": {
                "Range": ["Range"]
            }
        }, path="Range")
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "API": {
                "std模块": {
                    "math包": ["接口1", "接口2", "函数"]
                }
            }
        }, path="std.math")
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={"cangjie": {
            "Collection类型": [
                {
                    "实例教程": ["ArrayList使用示例", "HashMap使用示例", "迭代器操作函数", "HashSet"]
                }],
        }}, path="Collection类型")
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "仓颉语言结构体": {
                "Array": ["Array", "Array1"]
            }
        }, path="Array")
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "API": {
                "std模块": {
                    "collection包": ["函数", "函数1", "ArrayList1", "ArrayList2", "LinkedList"]
                }
            }
        }, path="Collection类型")
        learning_dict(self, book_path="dataset/cangjie开发指南/", index={
            "API": {
                "std模块": ["sort"]
            }
        }, path="数组排序")

    def inference(self, prompt, system_prompt, prefix, is_coding=False):
        return openai_inference(self.client, prompt, system_prompt, self.model_name, prefix, is_coding=is_coding)