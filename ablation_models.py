import os.path

from anthropic import Anthropic
from openai import OpenAI

from cjlearner import prompt_load, CJLearnerForDeepseekV3
from utils import byte_inference, openai_inference, anthropic_inference
from keys import OPENAI_KEY
from leetcode_gen_base import to_valid_filename


class AblationLM_wo_NoteGenerating:
    model_name = None

    @classmethod
    def fingerprint(cls):
        assert cls.model_name is not None
        return to_valid_filename('-'.join([cls.model_name, "wo_NoteGen"]))

    def inference(self, prompt, system_prompt, prefix):
        raise NotImplemented
    
    
class AblationLM_wo_NoteGenerating_Deepseek(AblationLM_wo_NoteGenerating, CJLearnerForDeepseekV3):
    model_name = "deepseek-v3"

    def __init__(self, infra, dump_dir, load_path):
        CJLearnerForDeepseekV3.__init__(self, infra, dump_dir, load_path, language="en",
                                        using_note_generating=False)

    def inference(self, prompt, system_prompt, prefix):
        return CJLearnerForDeepseekV3.inference(self, prompt, system_prompt, prefix)


class AblationLM_wo_Attention:
    model_name = None

    @classmethod
    def fingerprint(cls):
        assert cls.model_name is not None
        return to_valid_filename('-'.join([cls.model_name, "wo_Atten"]))

    def inference(self, prompt, system_prompt, prefix):
        raise NotImplemented


class AblationLM_wo_Attention_Deepseek(AblationLM_wo_Attention, CJLearnerForDeepseekV3):
    model_name = "deepseek-v3"

    def __init__(self, infra, dump_dir, load_path):
        CJLearnerForDeepseekV3.__init__(self, infra, dump_dir, load_path, language="en",
                                        using_note_querying=False)

    def inference(self, prompt, system_prompt, prefix):
        return CJLearnerForDeepseekV3.inference(self, prompt, system_prompt, prefix)