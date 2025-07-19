import os
import sys

from tqdm import tqdm

from coder.sender import notify

from cjlearner import (CJLearnerForDoubao, CJLearnerForGPT4O, CJLearnerForQwen,
                       CJLearnerForDeepseekV3, CJLearnerForClaude3_haiku)
from comparedlm import ComparedQwen, ComparedDoubao, ComparedCluade, ComparedDeepseekv3, ComparedGPT4O
from ablation_models import AblationLM_wo_NoteGenerating_Deepseek, AblationLM_wo_Attention_Deepseek
from leetcode_gen_base import LANG_TYPE
from leetcode_gen_simple import solutions as simple_solutions, solutions_test as simple_solutions_test
from leetcode_gen_difficut import solutions as difficit_solutions
from leetcode_gen_week import solutions as week_solutions


models = ["deepseek", "doubao", "gpt-4o", "qwen", "claude"]
model_names = ["deepseek-v3", "doubao-1.5-pro-256k-250115_0214", "gpt-4o-2024-08-06", "qwen-max-2025-01-25",
               "claude-3-5-sonnet-20240620"]
INFO = True

model2name = {models[i]: model_names[i] for i in range(len(models))}


def get_models(model):
    if model == "deepseek":
        infra, CLASS = "byte", CJLearnerForDeepseekV3
    elif model == "doubao":
        infra, CLASS = "byte", CJLearnerForDoubao
    elif model == "gpt-4o":
        infra, CLASS = "openai", CJLearnerForGPT4O
    elif model == "claude":
        infra, CLASS = "anthropic", CJLearnerForClaude3_haiku
    elif model == "qwen":
        infra, CLASS = "ali", CJLearnerForQwen
    else:
        print(model)
        raise KeyError
    return infra, CLASS


def get_compared_models(model):
    if model == "deepseek":
        infra, CLASS = "byte", ComparedDeepseekv3
    elif model == "doubao":
        infra, CLASS = "byte", ComparedDoubao
    elif model == "gpt-4o":
        infra, CLASS = "openai", ComparedGPT4O
    elif model == "claude":
        infra, CLASS = "anthropic", ComparedCluade
    elif model == "qwen":
        infra, CLASS = "ali", ComparedQwen
    else:
        raise KeyError
    return infra, CLASS


def info(*d, **kw):
    if INFO:
        print(*d, **kw)


# def For(iters, head=None, tail=None):
#     iteers = list(iters)
#     for


def running_test(rewrite, language_coding, exp_dir, solutions, its=3, retest=False, **kw):
    scores = list()
    for it in range(its):
        tqdmor = tqdm(list(solutions()),
                      desc=rewrite.model_name + " " + language_coding.name + " " + str(it),
                      leave=True, dynamic_ncols=True)
        for s in tqdmor:
            info("testing ", s)
            scores.append(s.test(rewrite, language_coding, exp_dir, iter=it, retest=retest, **kw))
            info("score===========", scores[-1])
            tqdmor.set_postfix({
                "idx": s.idx,
                "score": scores[-1]
            })
    info(scores, sum(scores))
    notify("runing done.")


def base_test(model, language, coding_language, solutions, its=3, only_return=False, retest=False, **kw):
    infra, CLASS = get_models(model)
    exp_dir = f"exps/{language}/{infra}/{CLASS.model_name}"
    if only_return:
        return exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    dump_path = os.path.join(exp_dir, "dump.json")
    load_path = os.path.join(exp_dir, "load.json")
    rewrite = CLASS(infra=infra, dump_dir=dump_path, load_path=load_path, debug=True, language=language)
    running_test(rewrite, coding_language, exp_dir, solutions, its=its, retest=retest, **kw)


def compare_with_baseline(model, language, solutions, only_return=False, retest=False):
    infra, CLASS = get_compared_models(model)

    rewrite = CLASS(prompt_filename="prompts/few_shots_prompt.txt", infra=infra)

    exp_dir = f"exps/{language}/{infra}/{rewrite.fingerprint()}"
    if only_return:
        assert os.path.isdir(exp_dir), exp_dir
        return exp_dir
    print("compare with basiline")
    os.makedirs(exp_dir, exist_ok=True)

    running_test(rewrite, LANG_TYPE.CANGJIE, exp_dir, solutions, retest=retest)


def compare_with_baseline_commented(model, language, solutions, only_return=False):
    infra, CLASS = get_compared_models(model)

    rewrite = CLASS(prompt_filename="prompts/few_shots_prompt&.txt", infra=infra)

    exp_dir = f"exps/{language}/{infra}/{rewrite.fingerprint()}"
    if only_return:
        assert os.path.isdir(exp_dir), exp_dir
        return exp_dir
    print("compare with commented baseline")
    os.makedirs(exp_dir, exist_ok=True)

    running_test(rewrite, LANG_TYPE.CANGJIE, exp_dir, solutions)


def ablation_wo_note_generating(solutions, only_return=False, return_model=False):
    CLASS = AblationLM_wo_NoteGenerating_Deepseek

    exp_dir = f"exps/en/byte/{CLASS.fingerprint()}"
    if only_return:
        return exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    dump_path = os.path.join(exp_dir, "dump.json")
    load_path = os.path.join(exp_dir, "load.json")

    model = CLASS("byte", dump_path, load_path)
    if return_model:
        return model
    else:
        running_test(model, LANG_TYPE.CANGJIE, exp_dir, solutions)


def ablation_wo_attention(solutions, only_return=False, return_model=False):
    CLASS = AblationLM_wo_Attention_Deepseek

    exp_dir = f"exps/en/byte/{CLASS.fingerprint()}"
    if only_return:
        return exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    dump_path = os.path.join(exp_dir, "dump.json")
    load_path = os.path.join(exp_dir, "load.json")

    model = CLASS("byte", dump_path, load_path)
    if return_model:
        return model
    else:
        running_test(model, LANG_TYPE.CANGJIE, exp_dir, solutions)


def model_fusion(noter_name, language, coding_language, solutions, its=3):
    def remove_and_shift(ls, n):
        return ls[ls.index(n) + 1:] + ls[0:ls.index(n)]

    infra_noter, CLASS_noter = get_models(noter_name)
    models_to_fusion = list(models)
    models_to_fusion = remove_and_shift(models_to_fusion, noter_name)
    for model in models_to_fusion:
        infra_coder, CLASS_coder = get_models(model)
        exp_dir_noter = f"exps/{language}/{infra_noter}/{CLASS_noter.model_name}"
        exp_dir_coder = f"exps/{language}/{infra_coder}/{CLASS_coder.model_name}"
        exp_dir = f"exps/{language}/{infra_coder}/{CLASS_noter.model_name}@{CLASS_coder.model_name}"
        dump_path_noter = os.path.join(exp_dir_noter, "dump.json")
        load_path_noter = os.path.join(exp_dir_noter, "load.json")
        dump_path_coder = os.path.join(exp_dir_coder, "dump.json")
        rewrite = CLASS_coder(infra=infra_coder, dump_dir=dump_path_coder, load_path=load_path_noter, language=language)
        running_test(rewrite, coding_language, exp_dir, solutions, its=its)


def chat_deepseek():
    infra, CLASS = get_models("deepseek")
    language = "en"
    exp_dir = f"exps/{language}/{infra}/{CLASS.model_name}"
    os.makedirs(exp_dir, exist_ok=True)
    dump_path = os.path.join(exp_dir, "dump.json")
    load_path = os.path.join(exp_dir, "load.json")
    rewrite = CLASS(infra=infra, dump_dir=dump_path, load_path=load_path, debug=True, language=language)
    rewrite.chat()

if __name__ == "__main__":
    from forging_utils.coder.smartassert import smart_assert
    smart_assert.disable()
    from leetcode_gen_week1 import solutions as week1_solutions
    from leetcode_gen_week2 import solutions as week2_solutions
    from leetcode_gen_week3 import solutions as week3_solutions
    from leetcode_gen_week4 import solutions as week4_solutions

    chat_deepseek()
    # model_name = models[3]
    # print("model baseline", model_name)
    # compare_with_baseline(model_name, "en", week3_solutions)
    # deepseek&claude
    # week1: (0.0833, 0.3333) -> (0.1, 0.4)
    # week2: (0.0, 0.25) -> (0.0667, 0.4)
    # week3: (0.0667, 0.4) -> (0.1167, 0.6667)
    # week4: (0.0333, 0.2833) -> (0.1167, 0.5167)
    # claude&deepseek
    # week1: (0.1167, 0.2667) -> (0.21, 0.4)
    # week2: (0.2333, 0.3833) -> (0.2333, 0.45)
    # week3: (0.1, 0.35) -> (0.2833, 0.6833)
    # week4: (0.0833, 0.3) -> (0.15, 0.5167)
    # infra_noter, CLASS_noter = get_models("claude")
    # models_to_fusion = list(models)
    # infra_coder, CLASS_coder = get_models("deepseek")
    # language = "en"
    # exp_dir_noter = f"exps/{language}/{infra_noter}/{CLASS_noter.model_name}"
    # exp_dir_coder = f"exps/{language}/{infra_coder}/{CLASS_coder.model_name}"
    # exp_dir = f"exps/{language}/{infra_coder}/{CLASS_noter.model_name}@{CLASS_coder.model_name}"
    # dump_path_noter = os.path.join(exp_dir_noter, "dump.json")
    # load_path_noter = os.path.join(exp_dir_noter, "load.json")
    # dump_path_coder = os.path.join(exp_dir_coder, "dump.json")
    # rewrite = CLASS_coder(infra=infra_coder, dump_dir=dump_path_coder, load_path=load_path_noter, language=language,
    #                       debug=True)
    # running_test(rewrite, LANG_TYPE.CANGJIE, exp_dir, week3_solutions, its=3, retest=True)
    # running_test(rewrite, LANG_TYPE.CANGJIE, exp_dir, week4_solutions, its=3, retest=True)

    # compare_with_baseline_commented(model_name, "en", week4_solutions)
    # for model in models:
    #     for lan in LANG_TYPE:
    #         base_test(model, "en", lan, week4_solutions)
    # base_test("claude", "en", LANG_TYPE.CANGJIE, week_solutions, retest=True)
    # model_fusion(model_name, "en", LANG_TYPE.CANGJIE, week3_solutions)
    # for model in models:
    #     base_test(model, "en", LANG_TYPE.ERLANG, week_solutions, its=3)
        # compare_with_baseline_commented(model, "en", solutions=week_solutions)
    # for model in models:
    #     compare_with_baseline(model, "en", solutions=difficit_solutions)
    # compare_with_baseline("doubao", "en", difficit_solutions)
    # base_test("deepseek", "en", LANG_TYPE.C, simple_solutions_test, its=1)
    # print(ablation_wo_attention(simple_solutions))