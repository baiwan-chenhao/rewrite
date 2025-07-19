import os.path
import time

from anthropic import Anthropic
from openai import OpenAI

from cjlearner import prompt_load
from utils import byte_inference, openai_inference, anthropic_inference
from keys import OPENAI_KEY
from leetcode_gen_base import to_valid_filename


class ComparedLM:
    model_name = None

    def __init__(self, prompt_filename, infra):
        self.prompt = prompt_filename
        self.infra = infra
        assert self.model_name is not None
        if infra in ("byte", "openai", "ali"):
            self.client = OpenAI(**OPENAI_KEY[infra])
        elif infra == "anthropic":
            self.client = Anthropic(**OPENAI_KEY["anthropic"])
        else:
            raise Exception("wrong infra: " + infra)

    def code_gen(self, task, **kwargs):
        prompt = prompt_load(self.prompt, task=task)
        time_s = time.perf_counter()
        code = self.inference(prompt, "you are a helpful assistant", "```cangjie\n")
        res = {
            "code": code,
            "origin_code": "",
            "retrieve_token": 0,
            "token_used": 0,
            "tried_times": 1,
            "reflection": [],
            "results": [],
            "time_used" : time.perf_counter() - time_s
        }
        return res

    def fingerprint(self):
        return to_valid_filename('-'.join([self.model_name, "compared", os.path.split(self.prompt)[-1]]))

    def inference(self, prompt, system_prompt, prefix):
        raise NotImplemented


class ComparedDeepseekv3(ComparedLM):
    model_name = "deepseek-v3"

    def __init__(self, prompt_filename, infra="byte"):
        super().__init__(prompt_filename, infra)

    def inference(self, prompt, system_prompt, prefix):
        if self.infra == "byte":
            return byte_inference(self.client, prompt, system_prompt, self.model_name, prefix, is_coding=True)
        else:
            raise Exception("not supported")


class ComparedDoubao(ComparedLM):
    model_name = "doubao-1.5-pro-256k-250115_0214"

    def __init__(self, prompt_filename, infra="byte"):
        super().__init__(prompt_filename, infra)

    def inference(self, prompt, system_prompt, prefix):
        if self.infra == "byte":
            return byte_inference(self.client, prompt, system_prompt, self.model_name, prefix, is_coding=True)
        else:
            raise Exception("not supported")


class ComparedGPT4O(ComparedLM):
    model_name = "gpt-4o-2024-08-06"

    def __init__(self, prompt_filename, infra="openai"):
        super().__init__(prompt_filename, infra)

    def inference(self, prompt, system_prompt, prefix):
        if self.infra == "openai":
            return openai_inference(self.client, prompt, system_prompt, self.model_name, prefix, is_coding=True)
        else:
            raise Exception("not supported")


class ComparedCluade(ComparedLM):
    model_name = "claude-3-5-sonnet-20240620"

    def __init__(self, prompt_filename, infra="anthropic"):
        super().__init__(prompt_filename, infra)

    def inference(self, prompt, system_prompt, prefix):
        if self.infra == "anthropic":
            return anthropic_inference(self.client, prompt, system_prompt, self.model_name, prefix, is_coding=True)
        else:
            raise Exception("not supported")


class ComparedQwen(ComparedLM):
    model_name = "qwen-max-2025-01-25"

    def __init__(self, prompt_filename, infra="anthropic"):
        super().__init__(prompt_filename, infra)

    def inference(self, prompt, system_prompt, prefix):
        if self.infra == "ali":
            return openai_inference(self.client, prompt, system_prompt, self.model_name, prefix, is_coding=True)
        else:
            raise Exception("not supported")