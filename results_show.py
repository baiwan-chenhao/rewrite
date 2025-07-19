import json
import os.path
from typing import Optional
from collections import Counter
from enum import Enum

import tabulate

from leetcode_gen_base import LANG_TYPE, get_lan_name
from cjlearner import focused_learning_book, prompt_load
from leetcode_testing import get_models, model2name

import numpy as np


def abort(msg):
    print(msg)
    quit()


def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$ """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def check_fingerprint(test_file, solution_idx):
    fingerprint = None
    for dumping in test_file:
        for idx in solution_idx:
            if fingerprint is None:
                fingerprint = dumping[idx]["param_fingerprint"]
            else:
                if fingerprint != dumping[idx]["param_fingerprint"]:
                    abort("wrong fingerprint")


def get_test_file(exp_dir, coding_lan, n_at_passk=3):
    assert exp_dir
    # if not exp_dir:
    #     exp_dir = f"exps/{language}/{infra}/{CLASS.model_name}"
    file_names = [f"{i}-{coding_lan}-[{PARAMS}].json" for i in range(n_at_passk)]
    test_file = [json.load(open(os.path.join(exp_dir, name), encoding="utf-8", mode="r")) for name in file_names]
    return test_file


def get_score_of_files(exp_dir, coding_lan, solutions, n_at_passk=3):
    test_file = get_test_file(exp_dir, coding_lan, n_at_passk)
    solution_idx = [s.idx for s in solutions()]
    for file in test_file:
        score = list()
        for k in solution_idx:
            score.append(file[k]["score"])
            assert score[-1] != -1
        print(sum(score) / len(score), score)


def show_failed_code(solutions, which):
    test_file = get_test_file(which)[-1]
    solution_idx = [s.idx for s in solutions()]
    for p, idx in enumerate(solution_idx):
        if test_file[idx]["failed"]:
            print("=" * 20 + "solution" + str(p + 1) + " " + idx + "\n", test_file[idx]["result"], "\n\n",
                  test_file[idx]["code"], end="\n")
            print(test_file[idx]["origin_code"], end="\n\n\n\n\n")


def compute_pass_at_k(exp_dir, coding_lan, solutions, n_at_passk=3, k=1, tried_times=3, exclude=None,
                      only_for_notfailed=False):
    if exclude is None:
        exclude = list()
    test_file = get_test_file(exp_dir, coding_lan, n_at_passk)
    solution_idx = [s.idx for s in solutions() if s.idx not in exclude]
    num = len(solution_idx)
    check_fingerprint(test_file, solution_idx)
    pass_ = 0
    for idx in solution_idx:
        passed = sum([test_file[i][idx]["score"] == 1 and
                      test_file[i][idx]["tried_times"] <= tried_times for i in range(n_at_passk)])
        pass_ += pass_at_k(n_at_passk, passed, k)
    return pass_ / num


def compute_pass_at_k_soft(exp_dir, coding_lan, solutions, n_at_passk=3, k=1, tried_times=3, exclude=None):
    if exclude is None:
        exclude = list()
    test_file = get_test_file(exp_dir, coding_lan, n_at_passk)
    solution_idx = [s.idx for s in solutions() if s.idx not in exclude]
    num = len(solution_idx)
    check_fingerprint(test_file, solution_idx)

    pass_ = 0
    for idx in solution_idx:
        passed = sum([not test_file[i][idx]["failed"] and
                      test_file[i][idx]["tried_times"] <= tried_times for i in range(n_at_passk)])
        pass_ += pass_at_k(n_at_passk, passed, k)
    return pass_ / num


def compute_pass_at_k_modified(exp_dir, coding_lan, solutions, n_at_passk=3, k=1, tried_times=3):
    test_file = get_test_file(exp_dir, coding_lan, n_at_passk)
    solution_idx = [s.idx for s in solutions()]
    num = 0
    check_fingerprint(test_file, solution_idx)

    pass_ = 0
    for idx in solution_idx:
        all_grammar_correct = sum([not test_file[i][idx]["failed"] and
                      test_file[i][idx]["tried_times"] <= tried_times for i in range(n_at_passk)]) == n_at_passk
        if all_grammar_correct:
            passed = sum([test_file[i][idx]["score"] == 1 and
                          test_file[i][idx]["tried_times"] <= tried_times for i in range(n_at_passk)])
            pass_ += pass_at_k(n_at_passk, passed, k)
            num += 1
    return (pass_ / num) if num > 0 else 0


from leetcode_gen_simple import solutions as simple_solutions
from leetcode_gen_difficut import solutions as difficut_solutions
from leetcode_gen_week import solutions as week_solutions
from leetcode_testing import models, compare_with_baseline_commented, compare_with_baseline, base_test

n_at_passk = 3
PARAMS = ""


# show_failed_code(1)


def baseline_results(solutions, language="en", return_tulpe=False, tried_times=3):
    result = dict()
    excludes = ["14", "115", "123", "154", "188"]
    for model in models:
        exp_dir = compare_with_baseline(model, language, only_return=True, solutions=solutions)
        if return_tulpe:
            r = (
                compute_pass_at_k(exp_dir, LANG_TYPE.CANGJIE, solutions, tried_times=tried_times, exclude=excludes),
                compute_pass_at_k_soft(exp_dir, LANG_TYPE.CANGJIE, solutions, tried_times=tried_times, exclude=excludes)
            )
        else:
            r = "%.4f(%.4f)" % (
                compute_pass_at_k(exp_dir, LANG_TYPE.CANGJIE, solutions, tried_times=tried_times, exclude=excludes),
                compute_pass_at_k_soft(exp_dir, LANG_TYPE.CANGJIE, solutions, tried_times=tried_times, exclude=excludes)
            )
        result[model] = r

    return result


def baseline_commented_results(solutions, language="en", return_tulpe=False):
    result = dict()
    excludes = ["14", "115", "123", "154", "188"]
    for model in models:
        exp_dir = compare_with_baseline_commented(model, language, only_return=True, solutions=solutions)
        if return_tulpe:
            r = (
                compute_pass_at_k(exp_dir, LANG_TYPE.CANGJIE, solutions, exclude=excludes),
                compute_pass_at_k_soft(exp_dir, LANG_TYPE.CANGJIE, solutions, exclude=excludes)
            )
        else:
            r = "%.4f(%.4f)" % (
                compute_pass_at_k(exp_dir, LANG_TYPE.CANGJIE, solutions, exclude=excludes),
                compute_pass_at_k_soft(exp_dir, LANG_TYPE.CANGJIE, solutions, exclude=excludes)
            )
        result[model] = r

    return result


def across_languange_results(solutions, language="en", lans: Optional[list] = None, return_tulpe=False, models_=models):
    result = dict()
    if lans is None:
        lans = LANG_TYPE
    for lan in lans:
        lan_name = get_lan_name(lan)
        if lan_name not in result:
            result[lan_name] = dict()
        for model in models_:
            exp_dir = base_test(model, language, lan, solutions, only_return=True)
            if return_tulpe:
                r = (
                    compute_pass_at_k(exp_dir, lan, solutions),
                    compute_pass_at_k_soft(exp_dir, lan, solutions)
                )
            else:
                r = "%.4f(%.4f)" % (
                    compute_pass_at_k(exp_dir, lan, solutions),
                    compute_pass_at_k_soft(exp_dir, lan, solutions)
                )
            result[lan_name][model] = r
    return result


def get_document_len():
    book_len = 0
    for book in focused_learning_book:
        with open(book, encoding="utf-8", mode="r") as f:
            book_len += len(f.read())
    return book_len


def measure_learning_capability(solutions, model_dir, tried_times=3, max_tried_times=3):
    document_len = get_document_len()
    solution_idx = [s.idx for s in solutions()]

    ratio = tried_times / max_tried_times

    # model_dir = "exps/en/byte/deepseek-v3-wo_NoteGen"
    load_file = os.path.join(model_dir, "load.json")
    dump_file = os.path.join(model_dir, "dump.json")
    load_file = json.load(open(load_file, encoding="utf-8", mode="r"))
    dump_file = json.load(open(dump_file, encoding="utf-8", mode="r"))
    note_len = load_file["note_len"]
    test_file = get_test_file(model_dir, LANG_TYPE.CANGJIE)
    token_used = 0
    task_count = 0
    for file in test_file:
        if solutions:
            for k in solution_idx:
                token_used += file[k]["token_used"] * ratio
                task_count += 1

    pass_k = compute_pass_at_k(model_dir, LANG_TYPE.CANGJIE, solutions, tried_times=tried_times)
    pass_k_soft = compute_pass_at_k_soft(model_dir, LANG_TYPE.CANGJIE, solutions, tried_times=tried_times)

    def get_size(file):
        size = 0
        for k, v in file.items():
            size += len(k) + len(v)
        return size

    note_compression_ratio = note_len / document_len
    token_using_in_learning = get_size(dump_file)
    inference_cost = int(token_used / task_count)

    return {
        "note_compression_ratio $\\downarrow$": "%.4f" % note_compression_ratio,
        "token_using_in_learning $\\downarrow$": "%d" % token_using_in_learning,
        "token_using_in_coding $\\downarrow$": "%d" % inference_cost,
        "pass@k $\\uparrow$": "%.4f(%.4f)" % (pass_k, pass_k_soft)
    }


def ability_merge_results(model_noter, solution):
    # solution = week_solutions
    results = list()
    for model_coder in models:
        if model_coder == model_noter:
            exp_dir = base_test(model_coder, "en", LANG_TYPE.CANGJIE, solution, only_return=True)
        else:
            infra, _ = get_models(model_coder)
            exp_dir = f"exps/en/{infra}/{model2name[model_noter]}@{model2name[model_coder]}"
        hard = compute_pass_at_k(exp_dir, LANG_TYPE.CANGJIE, solution)
        hard = float("%.4f" % hard)
        soft = compute_pass_at_k_soft(exp_dir, LANG_TYPE.CANGJIE, solution)
        soft = float("%.4f" % soft)
        results.append((hard, soft))
    return results

    # from itertools import combinations
    # results = list()
    # for model in models:
    #     results.append(ability_merge_results(model))
    # for m1, m2 in combinations(range(len(models)), 2):
    #     if results[m1][m1][0] > results[m2][m2][0]:
    #         coder = m1
    #     elif results[m1][m1][0] < results[m2][m2][0]:
    #         coder = m2
    #     else:
    #         coder = -1
    #     if results[m1][m1][1] > results[m2][m2][1]:
    #         noter = m1
    #     elif results[m1][m1][1] < results[m2][m2][1]:
    #         noter = m2
    #     else:
    #         noter = -1
    #     if coder == noter:
    #         print(models[m1], models[m2], "is same.")
    #         continue
    #     if results[noter][coder][0] > results[coder][coder][0] > results[noter][noter][0] > results[coder][noter][0]:
    #         print(f"({models[noter]}.note + {models[coder]}.code).hard({results[noter][coder][0]}) > {models[coder]}.hard({results[coder][coder][0]}) > {models[noter]}.hard({results[noter][noter][0]}) > "
    #               f"({models[coder]}.note + {models[coder]}.code).hard({results[coder][noter][0]}) yes", coder, noter)
    #     else:
    #         # if max(results[noter][coder][0], results[coder][noter][0]) > results[coder][coder][0]
    #         print("noter", models[noter], "coder", models[coder], f"{results[noter][coder][0]} > {results[coder][coder][0]} > {results[noter][noter][0]} > "
    #               f"{results[coder][noter][0]}", "hard")
    #
    #     if results[noter][coder][1] > results[noter][noter][1] > results[coder][coder][1] > results[coder][noter][1]:
    #         print(f"({models[noter]}.note + {models[coder]}.code).soft({results[noter][coder][1]}) > {models[noter]}.soft({results[noter][noter][1]}) > {models[coder]}.soft({results[coder][coder][1]}) > "
    #               f"({models[coder]}.note + {models[noter]}.code).soft({results[coder][noter][1]}) yes", coder, noter)
    #     else:
    #         print("noter", models[noter], "coder", models[coder], f"{results[noter][coder][1]} > {results[noter][noter][1]} > {results[coder][coder][1]} > "
    #               f"{results[coder][noter][1]}", "soft")


def segemnt_analysis(solution, model):
    infra, CLASS = get_models(model)
    exp_dir = f"exps/en/byte/{CLASS.model_name}"
    dump_path = os.path.join(exp_dir, "dump.json")
    load_path = os.path.join(exp_dir, "load.json")
    rewrite = CLASS(infra=infra, dump_dir=dump_path, load_path=load_path, debug=True, language="en")
    buckets = rewrite.code_gen(solution.des, return_keys=True)

    with open(load_path, encoding="utf-8", mode="r") as file:
        data = json.load(file)
    # note_sum_actual = sum([sum(map(len, i)) for i in data["memory"].values()])
    with open(dump_path, encoding="utf-8", mode="r") as file:
        dump = json.load(file)

    data = data["path_analyis"]
    lens_vanilla = [0]
    lens_noting = [0]
    lens_rewrite = [0]

    for k, i in data:
        bucket_name, op = i.split(" ")
        with open(k, mode="r", encoding="utf-8") as file:
            text = file.read()
        prompt = prompt_load("prompts/deepseek-v3/en/note_gen.txt", text=text, memory="", with_meta=True)
        key = "deepseek-v3" + prompt + "You are a helpful assistant."
        note = dump[key]
        lens_vanilla.append(lens_vanilla[-1] + len(text))
        lens_noting.append(lens_noting[-1] + len(note))
        if bucket_name in buckets:
            if op == "appended":
                lens_rewrite.append(lens_rewrite[-1] + len(note))
            elif op == "merged":
                # merged 信息没保存，进行估算。所有的merge操作共减少了k = sum(note_i) - actual_notes 个token。
                # 平均一个merge减少 k / n 个。 (56596 - 52999) / 70 = 51 以deepseek估计
                lens_rewrite.append(lens_rewrite[-1] + len(note) - 51)
            elif op == "new":
                lens_rewrite.append(lens_rewrite[-1] + len(note))
            else:
                raise NotImplemented
        else:
            lens_rewrite.append(lens_rewrite[-1])
    return lens_vanilla, lens_noting, lens_rewrite


def iteratalbe_merge(*iterables):
    def func():
        for it in iterables:
            for i in it():
                yield i

    return func


def scaling_of_tried_times(model, language, solutions, exp_dir=None, max_tried_times=3):
    if exp_dir is None:
        exp_dir = base_test(model, language, LANG_TYPE.CANGJIE, solutions, only_return=True)
    res_soft = list()
    res_hard = list()
    for i in range(1, max_tried_times + 1):
        res_soft.append(compute_pass_at_k_soft(exp_dir, LANG_TYPE.CANGJIE, solutions, tried_times=i))
        res_hard.append(compute_pass_at_k(exp_dir, LANG_TYPE.CANGJIE, solutions, tried_times=i))
    return res_soft, res_hard


class Result_type(Enum):
    AC = 0  # 测试样例全部通过
    Semantic = 1  # 编译通过，测试样例未全部通过
    API = 2  # 访问了不存在的成员。
    Grammer = 3  # 语法错误
    Timeout = 4  # 超时
    Param = 5  # 参数传递
    Undefine = 6  # 未定义变量


result_types = [str(Result_type(k)).split(".")[-1] for k in range(len(Result_type))]

def results_analysis(model, language, solutions, results_id=-1, exp_dir=None):
    API_indicator = "is not a member of ",
    Grammer_indicator = "expected", "rune literal may only contain one character"
    Param_indicator = "named parameter",
    Undefine_indicator = "undeclared identifier"

    if exp_dir is None:
        exp_dir = base_test(model, language, LANG_TYPE.CANGJIE, solutions, only_return=True)

    def contains(text: str, ts):
        for t in ts:
            if t in text:
                return True
        return False

    def result2type(item, results_id):
        res = None
        if not item["failed"]:
            if item["score"] == 1.0:
                res = Result_type.AC
            else:
                res = Result_type.Semantic
        else:
            if item["result"] == "timeout":
                res = Result_type.Timeout
                print(item)
            elif contains(item["results"][results_id], API_indicator):
                res = Result_type.API
            elif contains(item["results"][results_id], Grammer_indicator):
                res = Result_type.Grammer
            elif contains(item["results"][results_id], Param_indicator):
                res = Result_type.Param
            elif contains(item["results"][results_id], Undefine_indicator):
                res = Result_type.Undefine
        return res

    idxs = [s.idx for s in solutions()]
    ress = []
    for data in get_test_file(exp_dir, LANG_TYPE.CANGJIE, n_at_passk=3):
        for idx in idxs:
            item = data[idx]
            res = result2type(item, results_id)
            if res is None:
                print(item["results"][results_id])
            else:
                ress.append(res)

    ret = dict()
    total = len(ress)
    for k, v in Counter(ress).items():
        ret[str(k).split(".")[-1]] = "%.2f" % (v / total)
    for k in Result_type:
        if str(k).split(".")[-1] not in ret:
            ret[str(k).split(".")[-1]] = 0.0
    return ret

if __name__ == "__main__":


    from leetcode_gen_week2 import solutions as solution_week2
    from leetcode_gen_week1 import solutions as solution_week1
    from leetcode_gen_week3 import solutions as solution_week3
    from leetcode_gen_week4 import solutions as solution_week4
    from leetcode_gen_simple import Solution1, solutions as solution_simple
    from leetcode_gen_difficut import Solution1, solutions as solution_difficut
    from drawing_utils import Tabular, TableStyle

    weekall_solutions = iteratalbe_merge(week_solutions, solution_week1, solution_week2, solution_week3,
                                         solution_week4)
    print(scaling_of_tried_times("deepseek", "en", solutions=weekall_solutions, exp_dir=None, max_tried_times=3))

    # print(ability_merge_results("claude", solution_week3))
    # print(ability_merge_results("claude", solution_week4))
    # print(compute_pass_at_k_modified("exps/en/byte/deepseek-v3", LANG_TYPE.CANGJIE, weekall_solutions))
    # print(compute_pass_at_k_modified("exps/en/byte/deepseek-v3", LANG_TYPE.PYTHON, weekall_solutions))
    # print(results_analysis("", "", solution_week1, exp_dir="exps/en/byte/claude-3-5-sonnet-20240620@deepseek-v3"))
    # print(results_analysis("", "", solution_week1, exp_dir="exps/en/anthropic/deepseek-v3@claude-3-5-sonnet-20240620"))
    # get_score_of_files("exps/en/byte/deepseek-v3", LANG_TYPE.CANGJIE, solution_week, n_at_passk=3)
    #
    # # print(results_analysis("deepseek", "en", simple_solutions))
    #
    # PARAMS = "'try_number@5',_'times@2'"

    # get_score_of_files("exps/en/byte/deepseek-v3", LANG_TYPE.CANGJIE, solution_week, n_at_passk=3)
    # PARAMS = "'try_number@5'"
    # for s in (simple_solutions, difficut_solutions, week_solutions):
    #     print(scaling_of_tried_times("deepseek", "en", s,
    #                              exp_dir="exps/en/byte/deepseek-v3/",
    #                              max_tried_times=5))
    # PARAMS = "'try_number@6'"
    # print(scaling_of_tried_times("deepseek", "en", week_solutions,
    #                              exp_dir="exps/en/byte/deepseek-v3/",
    #                              max_tried_times=5))

    # file_path = "exps/en/byte/deepseek-v3/2-LANG_TYPE.CANGJIE-[].json"
    # print(results_analysis(file_path, simple_solutions))
    #
    # results = {'Python': {'deepseek': '0.0000(0.0000)', 'doubao': '0.3167(0.9000)', 'gpt-4o': '0.0000(0.0000)', 'qwen': '0.0000(0.0000)', 'claude': '0.3667(0.9333)'},
    #            'Cangjie': {'deepseek': '0.1167(0.2167)', 'doubao': '0.1000(0.1333)', 'gpt-4o': '0.1000(0.1333)', 'qwen': '0.1000(0.1000)', 'claude': '0.0667(0.2833)'},
    #            'Scala': {'deepseek': '0.0000(0.0000)', 'doubao': '0.1000(0.3167)', 'gpt-4o': '0.0000(0.0000)', 'qwen': '0.0000(0.0000)', 'claude': '0.1000(0.4000)'},
    #            'Erlang': {'deepseek': '0.0000(0.0000)', 'doubao': '0.0000(0.0000)', 'gpt-4o': '0.0000(0.0000)', 'qwen': '0.0000(0.0000)', 'claude': '0.1833(0.8167)'}}
    # print(compute_pass_at_k_soft("exps/en/anthropic/claude-3-5-sonnet-20240620", LANG_TYPE.PYTHON, solution_week2))
    # results = across_languange_results(solution_week2, "en")
    # tabular = Tabular().from_dicts(list(results.values()), list(results.keys()), head="language")
    # tabular.print()

    # print(segemnt_analysis(Solution1(), "deepseek"))

    # print(compute_pass_at_k("exps/en/byte/deepseek-v3", LANG_TYPE.CANGJIE, solution_week2))
    # print(compute_pass_at_k_soft("exps/en/byte/deepseek-v3", LANG_TYPE.CANGJIE, solution_week2))
    # test_file = get_test_file("exps/en/byte/deepseek-v3", LANG_TYPE.CANGJIE, 3)
    # solution_idx = [s.idx for s in week_solutions()]
    # num = len(solution_idx)
    # check_fingerprint(test_file, solution_idx)
    # pass_ = list()
    # p = 0
    # for idx in solution_idx:
    #     passed = sum([test_file[i][idx]["score"] == 1 and
    #                   test_file[i][idx]["tried_times"] <= 3 for i in range(n_at_passk)])
    #     pass_.append(passed)
    #     p += pass_at_k(3, passed, 1)
    # print(p / num)
    # print(pass_)
    # exp_dir = base_test("deepseek", "en", LANG_TYPE.ERLANG, simple_solutions, only_return=True)
    # r = "%.4f(%.4f)" % (
    #     compute_pass_at_k(exp_dir, LANG_TYPE.ERLANG, simple_solutions),
    #     compute_pass_at_k_soft(exp_dir, LANG_TYPE.ERLANG, simple_solutions)
    # )
    # print(r)

    # inputs = list()
    # labels = list()
    # for i in range(len(models)):
    #     for j in range(len(models) - 1):
    #         if j == i:
    #             continue
    #         inputs.append((results[i][i][0], results[i][i][1], results[j][j][0], results[j][j][1]))
    #         labels.append(results[i][j][0])
    #         # inputs.append((results[j][j][0], results[j][j][1], results[i][i][0], results[i][i][1]))
    #         # labels.append(results[j][i][0])
    #
    # print(inputs)
    # print(labels)
    # import numpy as np
    # import statsmodels.api as sm
    #
    # # 示例数据
    # X = np.array(inputs)
    # y = np.array(labels)
    #
    # # 添加截距项
    # X = sm.add_constant(X)
    #
    # # 建立模型
    # model = sm.OLS(y, X)
    #
    # # 拟合模型
    # results = model.fit()
    #
    # # 输出结果
    # print(results.summary())
    # inputs_train = inputs[:int(0.8 * len(inputs))]
    # inputs_test = inputs[int(0.8 * len(inputs)):]
    # labels_train = labels[:int(0.8 * len(labels))]
    # labels_test = labels[int(0.8 * len(labels)):]
    #
    #
    # import numpy as np
    # from sklearn.linear_model import SGDRegressor
    # from sklearn.pipeline import make_pipeline
    # from sklearn.preprocessing import StandardScaler
    #
    # # Always scale the input. The most convenient way is to use a pipeline.
    # reg = SGDRegressor(max_iter=100, tol=1e-3)
    # reg.fit(inputs_train, labels_train)
    # print(reg.score(inputs_test, labels_test))

# print(baseline_results(week_solutions))
# print([s.idx for s in difficut_solutions()])
# exp_dir = compare_with_baseline_commented("deepseek", "en", only_return=True, solutions=difficut_solutions)
# print(get_score_of_files(exp_dir, difficut_solutions))
# print(get_score_of_files("exps/en/byte/deepseek-v3", difficut_solutions))

# print(compute_pass_at_k(difficut_solutions, "exps/en/byte/deepseek-v3"),
#       compute_pass_at_k_soft(difficut_solutions, "exps/en/byte/deepseek-v3"))
