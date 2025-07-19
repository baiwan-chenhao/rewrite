import os.path
from itertools import permutations
from math import log

from matplotlib.ticker import MaxNLocator

from leetcode_gen_simple import solutions as simple_solutions, Solution1
from leetcode_gen_difficut import solutions as difficut_solutions
from leetcode_gen_week import solutions as week_solutions
from leetcode_gen_week1 import solutions as week1_solutions
from leetcode_gen_week2 import solutions as week2_solutions
from leetcode_gen_week3 import solutions as week3_solutions
from leetcode_gen_week4 import solutions as week4_solutions
from leetcode_gen_base import LANG_TYPE, get_lan_name

from results_show import (across_languange_results, baseline_results, base_test,
                          baseline_commented_results, measure_learning_capability,
                          ability_merge_results, segemnt_analysis, results_analysis, result_types)
import results_show
from leetcode_testing import models

import numpy as np
import matplotlib.pyplot as plt
from drawing_utils import drawing_tabel, merge_tabulars, remark_max, transpose, add_hat, Tabular, TableStyle


def drawing_across_language():
    caption = "Result across language"

    tabulars = list()
    dataset_names = "simple-30", "difficult-30", "week-20"
    col_num = len(models) + 1

    for p, solution in enumerate((simple_solutions, difficut_solutions, week_solutions)):
        align = "c" * col_num
        rows = [["Language"] + models]
        results = across_languange_results(solution, "en")
        for coding_lang in results:
            scores = [results[coding_lang][model] for model in models]
            rows.append([coding_lang] + scores)
        rows = [" & ".join(map(lambda x: "%s" % x, row)) for row in rows]
        content = "\\\\ \\hline \n".join(rows)
        tabular = r"""
\begin{tabular}{%s}
    \multicolumn{%d}{c}{%s} \\ \hline
    %s
  \end{tabular}
""" % (align, col_num, dataset_names[p], content)
        tabulars.append(tabular)

    tabel = r"""
\begin{table*}
  \centering
  %s
  \caption{%s}
  \label{tab:compare_baseline}
\end{table*}
    """ % ("\n".join(tabulars), caption)
    print(tabel)


def drawing_compared_with_baseline():
    caption = "The results of comparing with baselines in pass@$1$-hard(pass@$1$-soft)."
    label = "compare_baseline"

    tabulars = list()
    dataset_names = "Simple", "Hard", "Week"
    hat_row = ["method"] + models
    hat_col = ["rewrite", "5-shot", "5-shot\&c"]

    tabular_heads = ["Simple", "Hard", "week"]

    cangjie_lan_name = get_lan_name(LANG_TYPE.CANGJIE)

    for p, solution in enumerate((simple_solutions, difficut_solutions, weekall_solutions)):
        results0, results1 = list(), list()
        results_rewrite = across_languange_results(solution, "en", lans=[LANG_TYPE.CANGJIE], return_tulpe=True)
        # result[lan_name][model]
        results_baseline = baseline_results(solution, "en", return_tulpe=True)
        # result[model]
        results_baseline_commented = baseline_commented_results(solution, "en", return_tulpe=True)
        # result[model]

        for model in models:
            results0.append(
                [
                    results_rewrite[cangjie_lan_name][model][0],
                    results_baseline[model][0],
                    results_baseline_commented[model][0]
                ]
            )
            results1.append(
                [
                    results_rewrite[cangjie_lan_name][model][1],
                    results_baseline[model][1],
                    results_baseline_commented[model][1]
                ]
            )
        results0 = remark_max(results0, lambda x: r"\textbf{%.4f}" % x, lambda x: r"%.4f" % x)
        results1 = remark_max(results1, lambda x: r"\textbf{%.4f}" % x, lambda x: r"%.4f" % x)
        result = results0  # merge_tabulars(
            # results0, results1, lambda x, y: "%s(%s)" % (x, y)
        # )
        result = transpose(result)
        # hat_row, hat_col = hat_col, hat_row
        # hat_row.append("pass@1")
        result = add_hat(result, hat_row, hat_col)
        tabulars.append(result)
    print(drawing_tabel(tabulars, caption, label,
                  tabular_heads=tabular_heads, style=TableStyle.Booktabs))


def drawing_across_languanges(filename=None):
    markers = ".", "v", "^", "s", "+"  # lan
    titles = "Simple", "Hard", "Week"
    assert len(LANG_TYPE) <= len(markers)
    plt.figure(figsize=(18, 6))
    size = 14
    plt.rcParams.update({'font.size': size})
    # plt.xticks(fontsize=size)
    # plt.yticks(fontsize=size)
    for idx, solution in enumerate((simple_solutions, difficut_solutions, weekall_solutions)):
        plt.subplot(1, 3, idx + 1)
        print(idx)
        result = across_languange_results(solution, language="en", return_tulpe=True) # r[lan][model]
        for p, lan in enumerate(LANG_TYPE):
            marker = markers[p]
            x = list()
            y = list()
            for j, model in enumerate(models):
                x.append(model)
                y.append(result[get_lan_name(lan)][model][0])
            plt.scatter(x, y, marker=marker, label=get_lan_name(lan))
            plt.title(titles[idx], size=size)
            # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # if idx == 0:, label='Cangjie of the best fusion learner'
        #     plt.axhline(y=0.4778, color='green', linestyle='--', label='learning outcome of the hybrid strategy')
        if idx == 2:
            plt.axhline(y=0.2367, color='green', linestyle='--')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if filename:
        assert "." not in filename, "saving in pdf, do not provide suffix."
        f = plt.gcf()
        f.savefig(os.path.join(base_dir, filename + ".pdf"))
        f.clear()
    else:
        plt.show()
    # if filename:
    #
    #     f = plt.gcf()
    #     f.savefig(os.path.join(base_dir, filename + "-hard.pdf"))
    #     f.clear()
    # else:
    #     plt.show()


def drawing_ablations():
    solutions = simple_solutions
    whichs = "exps/en/byte/deepseek-v3", "exps/en/byte/deepseek-v3-wo_NoteGen", "exps/en/byte/deepseek-v3-wo_Atten"
    labels = ["rewrite", "w/o noting", "w/o attention"]
    dicts = list()
    for p, which in enumerate(whichs):
        r = measure_learning_capability(solutions, which)
        dicts.append(r)

    dicts.append(measure_learning_capability(solutions, whichs[0], tried_times=1))
    labels.append("w/o reflection")
    tabular = Tabular().from_dicts(dicts, labels, head="method")
    tabular.print_tabel("Ablation", "ablation", style=TableStyle.Booktabs, control_code=7)
    tabular.print()


def drawing_measures():
    solutions = simple_solutions
    rs = list()
    for model in models:
        exp_dir = base_test(model, "en", LANG_TYPE.CANGJIE, solutions, only_return=True)
        r = measure_learning_capability(solutions, exp_dir)
        rs.append(r)
    tabular = Tabular().from_dicts(rs, models, head="model")
    tabular.print_tabel("capability", "capability")
    tabular.print()


def drawing_ability_merge(solution, filename=None, cmap="viridis"):
    # ^ noter
    # |
    # ----> coder
    hard_scores = list()
    soft_scores = list()
    for noter in models:
        scores = ability_merge_results(noter, solution)
        hard_scores.append(list(map(lambda x: x[0], scores)))
        soft_scores.append(list(map(lambda x: x[1], scores)))
    hard_scores = np.array(hard_scores)
    soft_scores = np.array(soft_scores)

    plt.imshow(hard_scores, cmap=cmap, interpolation="nearest")  # RdYlGn
    plt.colorbar()
    plt.xticks(np.arange(len(models)), models)  # 设置列标签
    plt.yticks(np.arange(len(models)), models)
    for i in range(hard_scores.shape[0]):  # 遍历行
        for j in range(hard_scores.shape[1]):  # 遍历列
            plt.text(j, i, f"{hard_scores[i, j]:.4f}",  # 保留1位小数
                     ha="center", va="center",  # 水平和垂直居中
                     color="white" if hard_scores[i, j] > np.max(hard_scores) / 2 else "black")  # 自动
    if filename:
        assert "." not in filename, "saving in pdf, do not provide suffix."
        f = plt.gcf()
        f.savefig(os.path.join(base_dir, filename + "-hard.pdf"))
        f.clear()
    else:
        plt.show()

    plt.imshow(soft_scores, cmap=cmap, interpolation="nearest")
    plt.colorbar()
    plt.xticks(np.arange(len(models)), models)  # 设置列标签
    plt.yticks(np.arange(len(models)), models)
    # 在热力图的每个单元格中添加数值标签
    for i in range(soft_scores.shape[0]):  # 遍历行
        for j in range(soft_scores.shape[1]):  # 遍历列
            plt.text(j, i, f"{soft_scores[i, j]:.4f}",  # 保留1位小数
                     ha="center", va="center",  # 水平和垂直居中
                     color="black")  # 自动
    if filename:
        f = plt.gcf()
        f.savefig(os.path.join(base_dir, filename + "-soft.pdf"))
        f.clear()
    else:
        plt.show()

def drawing_segment_analysis(*solutions, filename=None):
    for solution in solutions:
        lens_vanilla, lens_noting, lens_rewrite = segemnt_analysis(solution, "deepseek")
        max_len = lens_vanilla[-1]
        lens_vanilla = np.array(lens_vanilla) / max_len
        lens_noting = np.array(lens_noting) / max_len
        lens_rewrite = np.array(lens_rewrite) / max_len
        x = lens_vanilla
        plt.plot(x, lens_rewrite, label="rewrite of " + str(solution.idx))
    plt.plot(x, lens_vanilla, label="vanilla")
    plt.plot(x, lens_noting, label="w/o attention")
    plt.plot(x, np.log10(lens_vanilla + 1), label="log10(x+1)")
    plt.legend()
    if filename:
        assert "." not in filename, "saving in pdf, do not provide suffix."
        f = plt.gcf()
        f.savefig(os.path.join(base_dir, filename + ".pdf"))
        f.clear()
    else:
        plt.show()


def drawing_code_analysis_table(model):
    language = "en"
    ss = (simple_solutions, difficut_solutions, week_solutions)
    names = "simple", "hard", "week0"
    res = dict()
    for idx, s in enumerate(ss):
        res[names[idx]] = results_analysis(model, language, s)
    tabular = Tabular().from_dicts(list(res.values()), list(res.keys()), head="type", keys=result_types, var2title=False)
    tabular.transpose()
    tabular.print_tabel("code result analysis", "code_result", style=TableStyle.Booktabs)


def drawing_scaling_of_reflection(filename=None):
    results_show.PARAMS = "'try_number@5'"
    colors = "c", "g", "m"
    plt.figure(figsize=(6, 6))
    # plt.subplot(2, 1, 1)
    soft, hard = results_show.scaling_of_tried_times("deepseek", "en", simple_solutions,
                                                         exp_dir="exps/en/byte/deepseek-v3/",
                                                         max_tried_times=5)
    # plt.plot(np.arange(1, 6), soft, label="Simple-soft", linestyle="--", color=colors[0]), color=colors[0]
    plt.plot(np.arange(1, 6), hard, label="Simple", linestyle="-")

    soft, hard = results_show.scaling_of_tried_times("deepseek", "en", difficut_solutions,
                                                     exp_dir="exps/en/byte/deepseek-v3/",
                                                     max_tried_times=5)
    # plt.plot(np.arange(1, 6), soft, label="Hard-soft", linestyle="--", color=colors[1]), color=colors[1]
    plt.plot(np.arange(1, 6), hard, label="Hard", linestyle="-")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    # plt.subplot(2, 1, 2)
    soft, hard = results_show.scaling_of_tried_times("deepseek", "en", week_solutions,
                                                     exp_dir="exps/en/byte/deepseek-v3/",
                                                     max_tried_times=5)
    # plt.plot(np.arange(1, 6), soft, label="Week0-soft", linestyle="--", color=colors[2]), color=colors[2]
    plt.plot(np.arange(1, 6), hard, label="Week0", linestyle="-")
    results_show.PARAMS = "'try_number@4'"
    soft, hard = results_show.scaling_of_tried_times("deepseek", "en", week_solutions,
                                                     exp_dir="exps/en/byte/deepseek-v3/",
                                                     max_tried_times=4)
    # plt.plot(np.arange(1, 5), soft, label="Week0-soft", linestyle="--", color=colors[1]), color=colors[1]
    plt.plot(np.arange(1, 5), hard, label="Week0", linestyle="-")

    results_show.PARAMS = "'try_number@3',_'times@1'"

    soft, hard = results_show.scaling_of_tried_times("deepseek", "en", week_solutions,
                                                     exp_dir="exps/en/byte/deepseek-v3/",
                                                     max_tried_times=3)
    # plt.plot(np.arange(1, 4), soft, label="Week0-soft", linestyle="--", color=colors[0]), color=colors[0]
    plt.plot(np.arange(1, 4), hard, label="Week0", linestyle="-")

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    if filename:
        assert "." not in filename, "saving in pdf, do not provide suffix."
        f = plt.gcf()
        f.savefig(os.path.join(base_dir, filename + ".pdf"))
        f.clear()
    else:
        plt.show()


base_dir = r"C:\Users\xzh\Desktop"

# from leetcode_gen_simple import Solution1, Solution2, Solution3
# drawing_segment_analysis(Solution1(), Solution2(), Solution3(), filename="segment_scaling")
# drawing_segment_analysis(Solution3(), "segment_scaling3")
# drawing_code_analysis_table("deepseek")"reflection_scaling"
# drawing_scaling_of_reflection()
weekall_solutions = results_show.iteratalbe_merge(week_solutions, week1_solutions, week2_solutions, week3_solutions, week4_solutions)

hard_scores = list()
soft_scores = list()
for noter in models:
    scores = ability_merge_results(noter, weekall_solutions)
    hard_scores.append(list(map(lambda x: x[0], scores)))
    soft_scores.append(list(map(lambda x: x[1], scores)))

def pair_classify(scores):
    def symbol(x, y):
        if x < y:
            return "+"
        elif x > y:
            return "-"
        else:
            return "="

    len_scores = len(scores)
    for i, j in permutations(range(len_scores), 2):
        if j <= i:
            continue
        nums = scores[i][i], scores[i][j], scores[j][i], scores[j][j]
        s1= symbol(nums[0], nums[2]) + symbol(nums[1], nums[3])
        print(s1)

def drawing_langs():
    solutions = simple_solutions
    whichs = "exps/en/byte/deepseek-v3", "exps/en/byte/deepseek-v3-wo_NoteGen", "exps/en/byte/deepseek-v3-wo_Atten"
    labels = ["rewrite", "w/o noting", "w/o attention"]
    dicts = list()
    for p, which in enumerate(whichs):
        r = measure_learning_capability(solutions, which)
        dicts.append(r)

    dicts.append(measure_learning_capability(solutions, whichs[0], tried_times=1))
    labels.append("w/o reflection")
    tabular = Tabular().from_dicts(dicts, labels, head="method")
    tabular.print_tabel("Ablation", "ablation", style=TableStyle.Booktabs, control_code=7)
    tabular.print()


drawing_scaling_of_reflection("reflection_scaling-v2")
# drawing_compared_with_baseline()
# drawing_across_languanges("across_lans_big")
# caption = "The results of comparing with baselines in pass@$1$-hard(pass@$1$-soft)."
# label = "compare_baseline"
#
# tabulars = list()
# dataset_names = "Simple", "Hard", "Week"
# hat_col = models
# hat_row = ["pass@1", "rewrite", "5-shot", "5-shot\&c"]
#
# tabular_heads = ["Simple", "Hard", "Week"]
#
# cangjie_lan_name = get_lan_name(LANG_TYPE.CANGJIE)
#
# for p, solution in enumerate((simple_solutions, difficut_solutions, weekall_solutions)):
#     results = list()
#     results_rewrite = across_languange_results(solution, "en", lans=[LANG_TYPE.CANGJIE], return_tulpe=True)
#     results_baseline = baseline_results(solution, "en", return_tulpe=True)
#     results_baseline_commented = baseline_commented_results(solution, "en", return_tulpe=True)
#
#     for model in models:
#         results.append(
#             [
#                 results_rewrite[cangjie_lan_name][model][0],
#                 results_baseline[model][0],
#                 results_baseline_commented[model][0]
#             ]
#         )
#     results = remark_max(results, lambda x: r"\textbf{%.4f}" % x, lambda x: r"%.4f" % x)
#     print()
#     result = add_hat(results, hat_row, hat_col)
#     tabulars.append(result)
#
# print(tabulars[0])
# t = Tabular(tabulars)
# t.print()
# t.print_tabel()
# print(pair_classify(hard_scores))
# pair_classify(soft_scores)
# drawing_across_languanges("across_lans")
# drawing_compared_with_baseline()
# drawing_ability_merge(weekall_solutions, cmap="Reds", filename="merge_results")
# results = across_languange_results(weekall_solutions, "en")
# tabular = Tabular().from_dicts(list(results.values()), list(results.keys()), head="language")
# tabular.print_tabel("The results of different languages in pass@-1-hard(pass@1-soft)", "scores_languages", style=TableStyle.Booktabs, control_code=7)
# # drawing_tabel([[[1, 2], [3, 4]]], caption="tabel")
# print(merge_tabulars(
#     [[10, 20], [30, 40]],
#     remark_max([[1, 2], [3, 4]], lambda x: "$%s$" % str(x)),
#     lambda x, y: "%d(%s)" % (x, y),
# )
# )
# drawing_compared_with_baseline()