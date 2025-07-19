import os
import json
from pprint import pprint

from tqdm import tqdm

from openai import OpenAI

from utils import enum_path
from keys import OPENAI_KEY
from glob import glob


def learning(agent, book_path):
    assert agent.load_path is not None  # 确保结果可以被保存下来
    index_path = os.path.join(book_path, 'index.json')
    assert os.path.exists(index_path)
    with open(index_path, 'r', encoding="utf-8") as f:
        index = json.load(f)
    # check all are exist.
    has_not_exist = False
    for filename in enum_path(path=index, suffix=".md", prefix=book_path):
        if not os.path.isfile(filename):
            print("not found", filename)
            has_not_exist = True
    if has_not_exist:
        return
    for filename in tqdm(list(enum_path(path=index, suffix=".md", prefix=book_path))):
        print("learning", filename)
        if os.path.isfile(filename):
            with open(filename, "r", encoding="utf-8") as f:
                text = f.read()
            agent.learn(text, filename)
        else:
            print("not found", filename)
    agent.save()


def learning_dict(agent, book_path, index: dict, path=None):
    assert agent.load_path is not None  # 确保结果可以被保存下来
    # check all are exist.
    has_not_exist = False
    for filename in enum_path(path=index, suffix=".md", prefix=book_path):
        if not os.path.isfile(filename):
            print("not found", filename)
            has_not_exist = True
    if has_not_exist:
        raise FileNotFoundError
    for filename in tqdm(list(enum_path(path=index, suffix=".md", prefix=book_path))):
        print("learning", filename)
        if os.path.isfile(filename):
            with open(filename, "r", encoding="utf-8") as f:
                text = f.read()
            agent.learn(text, filename, path=path)
        else:
            print("not found", filename)
    agent.save()


def learning_glob(agent, book_path):
    assert agent.load_path is not None  # 确保结果可以被保存下来

    for filename in tqdm(list(glob(book_path, recursive=True))):
        print("learning", filename)
        if os.path.isfile(filename):
            with open(filename, "r", encoding="utf-8") as f:
                text = f.read()
            agent.learn(text, filename)
        else:
            print("not found", filename)
    agent.save()


def reduce_parameters(model_name, language, infra, is_learning, debug=False):
    client = OpenAI(**OPENAI_KEY[infra])
    saving_dir = os.path.join("experiment", language, infra, model_name)
    if is_learning:  # learning mode
        os.makedirs(saving_dir, exist_ok=True)
    else:  # testing mode
        if not os.path.exists(saving_dir):
            print("that model haven't been teached.")
            return
        return saving_dir
    dump_dir = os.path.join(saving_dir, "dump.json")
    load_path = os.path.join(saving_dir, "load.json")
    prompt_dir = "prompts/V1/" + language
    res = {
        "client": client,
        "model_name": model_name,
        "dump_dir": dump_dir,
        "load_path": load_path,
        "prompt_dir": prompt_dir,
        "infra": infra
    }
    if debug:
        pprint(res)
    return res


book_index_0114 = {
    "cangjie": {
        "基本概念": ["标识符", "程序结构", "表达式", "函数"],
        "基础数据类型": ["整数类型", "浮点类型", "布尔类型", "字符类型", "字符串类型", "元组类型", "数组类型",
                         "区间类型", "Unit类型", "Nothing类型"],
        "函数": ["定义函数", "调用函数", "函数类型", "嵌套函数", "Lambda 表达式", "闭包", "函数调用语法糖", "函数重载",
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

book_index_adding = {
    "cangjie": {
        "基本概念": ["标识符", "程序结构", "表达式", "函数"],
        "基础数据类型": ["整数类型", "浮点类型", "布尔类型", "字符类型", "字符串类型", "元组类型", "数组类型",
                         "区间类型", "Unit类型", "Nothing类型", "类型转换"],
        "函数": ["定义函数", "调用函数", "函数类型", "嵌套函数", "Lambda 表达式", "闭包", "函数调用语法糖", "函数重载",
                 "操作符重载", "const 函数和常量求值"],
        "结构类型": ["定义struct类型", "创建struct实例", "mut函数"],
        "枚举类型和模式匹配": ["枚举类型", "Option类型", "模式概述", "模式的 Refutability", "match表达式",
                               "if-let表达式", "while-let表达式", "其他使用模式的地方"],
        "类和接口": ["类", "接口", "属性", "子类型关系", "类型转换"],
        # "泛型"
        # 扩展
        "Collection类型": ["基本Collection类型概述", "ArrayList", "HashSet", "HashMap", "Iterable 和 Collections",
                           {
                               "实例教程": ["ArrayList使用示例", "HashMap使用示例", "迭代器操作函数", "HashSet"]
                           }],
        # 包
        "异常处理": ["定义异常", "throw和异常处理", "常见运行时异常", "使用Option"]
    },
    # 下面的是困难额外学的
    "仓颉语言结构体": {
        "String": ["String", "String1", "String2", "StringBuilder"],
        "Range": ["Range"],
        "Array": ["Array", "Array1"]
    },
    "API": {
        "std模块": {
            "math包": ["接口1", "接口2", "函数"]
        }
    }
}


if __name__ == "__main__":
    from cjlearner import (CJLearnerForClaude3_haiku, CJLearnerForDeepseekV3, CJLearnerForDoubao,
                           CJLearnerForGPT4O, CJLearnerForQwen)
    from ablation_models import AblationLM_wo_NoteGenerating_Deepseek
    from forging_utils.coder.smartassert import smart_assert
    smart_assert.disable()

    CLASS = AblationLM_wo_NoteGenerating_Deepseek

    exp_dir = f"exps/en/byte/{CLASS.fingerprint()}"
    os.makedirs(exp_dir, exist_ok=True)
    dump_path = os.path.join(exp_dir, "dump.json")
    load_path = os.path.join(exp_dir, "load.json")

    model = CLASS("byte", dump_path, load_path)
    model.learn_book()

    # language, infra, CLASS = "en", "anthropic", CJLearnerForClaude3_haiku
    # language, infra, CLASS = "en", "byte", CJLearnerForDeepseekV3
    # language, infra, CLASS = "en", "byte", CJLearnerForDoubao
    # language, infra, CLASS = "en", "openai", CJLearnerForGPT4O
    # language, infra, CLASS = "en", "ali", CJLearnerForQwen
    #
    # exp_dir = f"exps/{language}/{infra}/{CLASS.model_name}"
    # os.makedirs(exp_dir, exist_ok=True)
    # dump_path = os.path.join(exp_dir, "dump.json")
    # load_path = os.path.join(exp_dir, "load.json")
    # rewrite = CLASS(infra=infra, dump_dir=dump_path, load_path=load_path, debug=True, language=language)
    # rewrite.learn_book()
    # reply = rewrite.code_gen("编写一个英文小写字母到1-26的映射的程序")
    # print(reply["code"])
    # print(reply)
    # learning_dict(rewrite, book_path="dataset/cangjie开发指南/", index=book_index_adding)
    # learning_dict(agent=rewrite, book_path="dataset/cangjie开发指南/", index={
    #     "cangjie": {
    #         "基本概念": ["标识符", "程序结构", "表达式", "函数"],
    #         "基础数据类型": ["整数类型", "浮点类型", "布尔类型", "字符类型", "字符串类型", "元组类型", "数组类型",
    #                          "区间类型", "Unit类型", "Nothing类型"],
    #         "函数": ["定义函数", "调用函数"]
    #     }
    # })
    # learning_dict(agent=rewrite, book_path="dataset/cangjie开发指南/", index={
    #     "cangjie": {
    #         "函数": ["函数类型", "嵌套函数", "Lambda 表达式", "闭包", "函数调用语法糖", "函数重载", "操作符重载",
    #                  "const 函数和常量求值"],
    #         "结构类型": ["定义struct类型", "创建struct实例", "mut函数"],
    #         "枚举类型和模式匹配": ["枚举类型", "Option类型", "模式概述", "模式的 Refutability", "match表达式",
    #                                "if-let表达式", "while-let表达式", "其他使用模式的地方"],
    #         "类和接口": ["类", "接口", "属性", "子类型关系", "类型转换"],
    #         # "泛型"
    #         # 扩展
    #         "Collection类型": ["基本Collection类型概述", "ArrayList", "HashSet", "HashMap", "Iterable 和 Collections"],
    #         # 包
    #         "异常处理": ["定义异常", "throw和异常处理", "常见运行时异常", "使用Option"]
    #     }
    # })
