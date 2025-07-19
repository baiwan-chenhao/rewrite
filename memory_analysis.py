import json
from cjlearner import prompt_load, path_join
from tqdm import tqdm

file_path = "exps/en/byte/deepseek-v3/load.json"
dump_path = "exps/en/byte/deepseek-v3/dump.json"


with open(file_path, encoding="utf-8", mode="r") as file:
    data = json.load(file)
# print(sum([sum(map(len, i)) for i in data["memory"].values()]))
with open(dump_path, encoding="utf-8", mode="r") as file:
    dump = json.load(file)

model_name = "deepseek-v3"
language = "en"
prompt_dir = f"prompts/{model_name}/{language}"
note_gen_path = path_join(prompt_dir, "note_gen.txt")

for k, v in tqdm(dump.items()):
    if v == "func greet(name: String): String {\n    return \"Hello, \" + name\n}\n":
        print(k)

# for path in data["has_mem_list"]:
#     print(path)
#     with open(path, mode="r", encoding="utf-8") as file:
#         textpage = file.read()
#     prompt = prompt_load(note_gen_path, text=textpage, memory="", with_meta=True)
#     sys_prompt = "You are a helpful assistant."
#     key = model_name + prompt + sys_prompt
#     note = dump[key]
#     print(note)
#     break