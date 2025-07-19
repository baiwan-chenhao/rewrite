import copy
import json
import os
from enum import Enum

from typing import get_origin, get_args, List

from evaluator import (CJCodeEvaluator, PYCodeEvaluator, ScalaCodeEvaluator, ErlangCodeEvaluator,
                       JavaCodeEvaluator, CCodeEvaluator)
from random import randint
import random
import string
import re
import inspect

NUM = 100  # 总样本数
NUM_1, NUM_2 = 20, 5  # 生成list时，list的种类个数和一种list的个数
assert NUM == NUM_1 * NUM_2
INT_MIN, INT_MAX = 0, 10000  # 生成int时的最小值最大值
LEN_LIST_MIN = 10  # 生成list时，list的长度最小值

param_fingerprint = ".".join(map(str, [NUM_1, NUM_2, INT_MIN, INT_MAX, LEN_LIST_MIN]))

random.seed(0)


# todo 增加参数水印了。但是参数水印改变后的测试逻辑还没编写。水印变了后，只有修改assert就可以了，不必再生成代码了。


def gen_lists_int(int_min=INT_MIN, int_max=INT_MAX, is_sorted=False, len_list_k=1, len_list_min=LEN_LIST_MIN,
                  len_list_max=LEN_LIST_MIN + NUM_1):
    for len_list in range(len_list_min, len_list_max):
        len_list = len_list * len_list_k
        for j in range(NUM_2):
            ints = [randint(int_min, int_max) for i in range(len_list)]
            if is_sorted:
                ints.sort()
            yield ints


def gen_matirx_int(int_min=INT_MIN, int_max=INT_MAX, n_min=1, n_max=100, m_min=1, m_max=100, is_same=False):
    for _ in range(NUM):
        matrix = list()
        m = random.randint(m_min, m_max)
        if is_same:
            n = m
        else:
            n = random.randint(n_min, n_max)
        for i in range(n):
            matrix.append([random.randint(int_min, int_max) for _ in range(m)])
        yield matrix


def gen_int(int_min=INT_MIN, int_max=INT_MAX):
    for i in range(NUM):
        yield randint(int_min, int_max)


def gen_lists_str(vocab=string.ascii_letters, length=20):
    for len_list in range(LEN_LIST_MIN, LEN_LIST_MIN + NUM_1):
        for j in range(NUM_2):
            yield ["".join(random.choices(vocab, k=length)) for i in range(len_list)]


def gen_str(vocab=string.ascii_letters, length=20, adding=""):
    for i in range(NUM):
        yield "".join(random.choices(vocab, k=length)) + adding


def intToRoman(num: int) -> str:
    Rval, Rstr, ans, i = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1], ["M", "CM", "D", "CD", "C", "XC", "L",
                                                                                  "XL", "X", "IX", "V", "IV",
                                                                                  "I"], "", 0
    while num:
        if x := num // Rval[i]:
            ans += Rstr[i] * x
            num %= Rval[i]
        else:
            i += 1
    return ans


def argv2cjparam(param):
    if isinstance(param, bool):
        return "true" if param else "false"
    elif isinstance(param, int):
        return repr(param)
    elif isinstance(param, str):
        return repr(param)
    elif isinstance(param, float):
        return repr(param)
    elif isinstance(param, list):
        if len(param) == 0:
            return "[]"
        elif isinstance(param[0], int):
            return "[%s]" % ",".join(map(repr, param))
        elif isinstance(param[0], str):
            return "[%s]" % ",".join(map(repr, param))
        elif isinstance(param[0], list):  # [[]]
            return "[%s]" % ",".join(map(argv2cjparam, param))
        else:
            raise Exception("不受支持的类型" + str(param))
    else:
        raise Exception("不受支持的类型" + str(param))


def argv2pyparam(param):
    if isinstance(param, bool):
        return "True" if param else "False"
    elif isinstance(param, int):
        return repr(param)
    elif isinstance(param, str):
        return repr(param)
    elif isinstance(param, float):
        return repr(param)
    elif isinstance(param, list):
        if len(param) == 0:
            return "[]"
        elif isinstance(param[0], int):
            return "[%s]" % ",".join(map(repr, param))
        elif isinstance(param[0], str):
            return "[%s]" % ",".join(map(repr, param))
        else:
            raise Exception("不受支持的类型" + str(param))
    else:
        raise Exception("不受支持的类型" + str(param))


def argv2cjparams(inputs):
    params = list()
    for param in inputs:
        params.append(argv2cjparam(param))
    return ", ".join(params)


def argv2pyparams(inputs):
    params = list()
    for param in inputs:
        params.append(argv2pyparam(param))
    return ", ".join(params)


class LANG_TYPE(Enum):
    PYTHON = 0
    CANGJIE = 1
    SCALA = 2
    ERLANG = 3
    JAVA = 4
    # C = 4  todo : 在参数、类型转换时加上就可以了。


def get_lan_name(lan: LANG_TYPE):
    return str(lan).split(".")[-1].capitalize()


def get_true_value(solution):
    for x in zip(*solution.gen()):
        p = copy.deepcopy(x)
        yield solution.solve(*p), x


def to_valid_filename(s):
    # 替换掉非法字符为下划线 _
    s = re.sub(r'[\\/*?:"<>|]', '_', s)
    # 去除首尾的空白字符
    s = s.strip()
    # 替换连续的空格为单个下划线
    s = re.sub(r'\s+', '_', s)
    return s


class LanguageHandle:
    is_inline_testing_code = False  # 控制solve函数是在main函数内部还是在最顶层。

    def __init__(self, func_head_wrapper, func_param_wrapper, language, coding_prompt, TRUE, FALSE,
                 func_name="solve", func_splitor=", "):
        """
        :param func_head_wrapper: 包含3个%s占位符，分别表示函数名、函数参数列表和函数返回类型。用于生成函数签名。
        :param func_param_wrapper: 参数的形式，包含两个%s占位符，分别表示参数名和参数类型。 int a=1 #  b:int = 2
        :param func_name: 函数签名中的函数名。
        :param func_splitor: 函数参数列表的分隔符，默认为英文逗号：","。
        """
        self.func_head_wrapper = func_head_wrapper
        self.func_param_wrapper = func_param_wrapper
        self.func_name = func_name
        self.func_splitor = func_splitor
        self.language = language
        self.coding_prompt = coding_prompt
        self.TRUE = TRUE
        self.FALSE = FALSE
        self.EXCEPTION = "exception"

    def param_convert(self, param):
        """
        把python中的值转换为目标语言的形式。
        :param param:
        :return:
        """
        raise NotImplementedError

    def params_convert(self, argvs):
        params = list()
        for argv in argvs:
            params.append(self.param_convert(argv))
        return self.func_splitor.join(params)

    def typing_convert(self, typing):
        """
        把python中的类型转换为目标语言的形式。
        :param typing:
        :return:
        """
        raise NotImplementedError

    def print_wrapper(self, output):
        raise NotImplementedError

    def codestarter_wrapper(self, code):
        raise NotImplementedError

    def try_catch_wrapper(self, code):
        raise NotImplementedError

    def get_evaluator(self):
        raise NotImplementedError

    def add_commit(self, code, commit):
        raise NotImplementedError

    def array_param_reduce(self, param, warpper, splitor):
        return warpper % splitor.join(map(self.param_convert, param))

    def func_signature_gen(self, callable):
        """
        将python的callable转化为对应语言的函数签名。
        :param callable:
        :return:
        """
        signature = inspect.signature(callable)
        params = list()
        for name, param in signature.parameters.items():
            params.append((name, self.typing_convert(param.annotation)))
        param_list = self.func_splitor.join([self.func_param_wrapper % param for param in params])
        func_signature = self.func_head_wrapper % (self.func_name,
                                                   param_list,
                                                   self.typing_convert(signature.return_annotation))
        return func_signature


class ScalaLanguageHandle(LanguageHandle):
    def __init__(self):
        super().__init__(func_head_wrapper="def %s(%s): %s", func_param_wrapper="%s: %s",
                         language="scala",
                         coding_prompt="请用scala语言编写以下代码，你只需要编写既定函数即可，不要写class。",
                         TRUE="true", FALSE="false")
        self.is_inline_testing_code = True

    def get_evaluator(self):
        return ScalaCodeEvaluator(timeout=30)

    def param_convert(self, param):
        # 参数转换
        if isinstance(param, bool):
            return "true" if param else "false"
        elif isinstance(param, (int, str, float)):
            return repr(param)
        elif isinstance(param, list):  # Array(1,2) [1,2] [1,2]
            return self.array_param_reduce(param, "Array(%s)", ",")
        else:
            raise Exception("不受支持的类型" + str(param))

    def typing_convert(self, typing):
        # 类型转换
        def get_item(i):
            if len(i) == 1:
                return i[0]
            else:
                raise Exception("not 1-len array.")

        origin = get_origin(typing)
        if origin is None:
            if typing == bool:
                return "Boolean"
            elif typing == int:
                return "Int"
            elif typing == str:
                return "String"
            elif typing == float:
                return "Float"
            else:
                raise Exception("不受支持的类型" + str(typing))
        else:
            if origin == list:
                return "Array[%s]" % self.typing_convert(get_item(get_args(typing)))
            else:
                raise Exception("不受支持的类型" + str(typing))

    def print_wrapper(self, output):
        return "println(%s)" % output

    def codestarter_wrapper(self, code):
        return """
object HelloWorld {
    def main(args: Array[String]): Unit = {
        %s
    }
}
""" % code

    def try_catch_wrapper(self, code):
        return """
        try {
            %s
        } catch {
            case e: Exception => {
                println("%s")
            }
        }
""" % (code, self.EXCEPTION)

    def add_commit(self, code, commit):
        return "\n".join(["//" + i for i in commit.splitlines()]) + "\n" + code

    def compare_var(self, a, b):
        return ""


class ErlangLanguageHandle(LanguageHandle):
    def __init__(self):
        super().__init__(func_head_wrapper="-spec %s(%s) -> %s",
                         func_param_wrapper="%s :: %s",
                         language="Erlang",
                         coding_prompt="请用erlang语言编写以下代码。你只需要写出对应函数即可，不要写出main函数",
                         TRUE="true", FALSE="false")

    def param_convert(self, param):
        if isinstance(param, bool):
            return "true" if param else "false"
        elif isinstance(param, (int, str, float)):
            return repr(param)
        elif isinstance(param, list):
            return self.array_param_reduce(param, "[%s]", ",")
        else:
            raise Exception("不受支持的类型" + str(param))

    def typing_convert(self, typing):
        def get_item(i):
            if len(i) == 1:
                return i[0]
            else:
                raise Exception("not 1-len array.")

        origin = get_origin(typing)
        if origin is None:
            if typing == bool:
                return "boolean()"
            elif typing == int:
                return "integer()"
            elif typing == str:
                return "unicode:unicode_binary()"
            elif typing == float:
                return "float()"
            else:
                raise Exception("不受支持的类型" + str(typing))
        else:
            if origin == list:
                return "[%s]" % self.typing_convert(get_item(get_args(typing)))
            else:
                raise Exception("不受支持的类型" + str(typing))

    def print_wrapper(self, output):
        return 'io:fwrite("~w", [%s])' % output

    def codestarter_wrapper(self, code):
        return "main(_) -> \n %s 0." % code

    def try_catch_wrapper(self, code):
        return 'try\n    %s\ncatch\n    _:_ ->\n        io:format("%s")\nend,' % (code, self.EXCEPTION)

    def get_evaluator(self):
        return ErlangCodeEvaluator(timeout=30)

    def add_commit(self, code, commit):
        return "\n".join(["%%" + i for i in commit.splitlines()]) + "\n" + code


class OrderdFormatString:
    def __init__(self, text : str):
        """
        insert str into text orderly.
        "%2s %1s" % ("a", "b") == "b a"
        :param text:
        """
        self.text = text

    def __mod__(self, other : list[str]):
        text = str(self.text)
        for p, o in enumerate(other):
            text = text.replace("%%%ds" % p, o)
        return text


# class CLanguageHandle(LanguageHandle):
#     is_inline_testing_code = False
#
#     def __init__(self):
#         raise Exception("not support for figerpoint")
#         super().__init__(func_head_wrapper=OrderdFormatString("%2s %0s(%1s)"),
#                          func_param_wrapper=OrderdFormatString("%1s %0s"),
#                          language="c",
#                          coding_prompt="请用c语言编写以下代码。你只需要写出对应函数即可，不要写出main函数。你需要生成需要的头文件引用。",
#                          TRUE="true", FALSE="false")
#
#     def param_convert(self, param):
#         if isinstance(param, bool):
#             return "0" if param else "1"
#         elif isinstance(param, (int, str, float)):
#             return repr(param)
#         elif isinstance(param, list):
#             return self.array_param_reduce(param, "[%s]", ",")
#         else:
#             raise Exception("不受支持的类型" + str(param))
#
#     def typing_convert(self, typing):
#         def get_item(i):
#             if len(i) == 1:
#                 return i[0]
#             else:
#                 raise Exception("not 1-len array.")
#
#         origin = get_origin(typing)
#         if origin is None:
#             if typing == bool:
#                 return "bool"
#             elif typing == int:
#                 return "int"
#             elif typing == str:
#                 return "char*"
#             elif typing == float:
#                 return "float"
#             else:
#                 raise Exception("不受支持的类型" + str(typing))
#         else:
#             if origin == list:
#                 return "[%s]" % self.typing_convert(get_item(get_args(typing)))
#             else:
#                 raise Exception("不受支持的类型" + str(typing))
#
#     def print_wrapper(self, output):
#         return 'printf("%%d”, %s);' % output
#
#     def codestarter_wrapper(self, code):
#         return "main(){    \n %s \n}" % code
#
#     def try_catch_wrapper(self, code):
#         return code  # C 没有try工具。
#
#     def get_evaluator(self):
#         return NotImplemented
#         # return CCodeEvaluator(timeout=4)
#
#     def add_commit(self, code, commit):
#         return "\n".join(["//" + i for i in commit.splitlines()]) + "\n" + code


class JavaLanguageHandle(LanguageHandle):
    def __init__(self):
        super().__init__(func_head_wrapper="public static %s %s(%s)",
                         func_param_wrapper="%s %s",
                         language="java",
                         coding_prompt="请用Java语言编写以下代码，你只需要编写既定函数即可，不要写class。",
                         TRUE="true", FALSE="false")
        self.is_inline_testing_code = False

    def get_evaluator(self):
        return JavaCodeEvaluator(timeout=30)

    def param_convert(self, param):
        if isinstance(param, bool):
            return "true" if param else "false"
        elif isinstance(param, (int, float)):
            return repr(param)
        elif isinstance(param, str):
            return f"\"{param}\""
        elif isinstance(param, list):
            if all(isinstance(i, int) for i in param):
                return self.array_param_reduce(param, "new int[]{%s}", ",")
            elif all(isinstance(i, str) for i in param):
                return self.array_param_reduce(param, "new String[]{%s}", ",")
            elif all(isinstance(i, list) for i in param) and all(isinstance(i, list) and len(i) > 0 and all(isinstance(j, int) for j in i) for i in param):
                sub_arrays = [self.array_param_reduce(sub, "new int[]{%s}", ",") for sub in param]
                return f"new int[][]{{{','.join(sub_arrays)}}}"
            else:
                raise Exception("Unsupported list type or multi-dimensional array.")
        else:
            raise Exception(f"Unsupported parameter type: {type(param)}")

    def typing_convert(self, typing):
        if typing == bool:
            return "boolean"
        elif typing == int:
            return "int"
        elif typing == str:
            return "String"
        elif typing == float:
            return "double"
        elif get_origin(typing) == list:
            inner_type = get_args(typing)[0]
            if inner_type == int:
                return "int[]"
            elif inner_type == str:
                return "String[]"
            elif get_origin(inner_type) == list and get_args(inner_type)[0] == int:
                return "int[][]"
            else:
                raise Exception(f"Unsupported list type: {inner_type}")
        else:
            raise Exception(f"Unsupported type: {typing}")

    def print_wrapper(self, output):
        return f"System.out.println({output});"

    def codestarter_wrapper(self, code):
        return f"""
import java.util.*;
        
public class CodeEvaluating {{
    <<solution_code>>
    public static void main(String[] args) {{
        {code}
    }}
}}
"""

    def try_catch_wrapper(self, code):
        return f"""
try {{
    {code}
}} catch (Exception e) {{
    System.out.println("{self.EXCEPTION}");
}}
"""

    def add_commit(self, code, commit):
        return "\n".join([f"// {line}" for line in commit.splitlines()]) + "\n" + code

    def array_param_reduce(self, param, warpper, splitor):
        # 使用 param_convert 确保字符串带双引号
        return warpper % splitor.join(map(self.param_convert, param))


class CLanguageHandle(LanguageHandle):
    def __init__(self):
        super().__init__(
            func_head_wrapper="%s %s(%s)",  # e.g., "int solve(int a, char* b)"
            func_param_wrapper="%s %s",  # e.g., "int a"
            language="c",
            coding_prompt="请用C语言编写以下代码，你只需要编写既定函数即可",
            TRUE="1",  # C uses int for boolean: 1 for true
            FALSE="0",  # 0 for false
            func_splitor=", "  # Parameter separator
        )

    def get_evaluator(self):
        return CCodeEvaluator(timeout=30)

    def param_convert(self, param):
        if isinstance(param, bool):
            return "1" if param else "0"
        elif isinstance(param, int):
            return str(param)
        elif isinstance(param, float):
            return repr(param)
        elif isinstance(param, str):
            return f"\"{param}\""
        elif isinstance(param, list):
            if all(isinstance(i, int) for i in param):
                return self.array_param_reduce(param, "{%s}", ",")
            elif all(isinstance(i, str) for i in param):
                return self.array_param_reduce(param, "{%s}", ",")
            elif all(isinstance(i, list) for i in param) and all(
                    isinstance(i, list) and len(i) > 0 and all(isinstance(j, int) for j in i) for i in param):
                sub_arrays = [self.array_param_reduce(sub, "{%s}", ",") for sub in param]
                return f"{{{','.join(sub_arrays)}}}"
            else:
                raise Exception("Unsupported list type or multi-dimensional array.")
        else:
            raise Exception(f"Unsupported parameter type: {type(param)}")

    def typing_convert(self, typing):
        if typing == bool:
            return "int"  # C uses int for boolean
        elif typing == int:
            return "int"
        elif typing == float:
            return "double"
        elif typing == str:
            return "char*"
        elif get_origin(typing) == list:
            inner_type = get_args(typing)[0]
            if inner_type == int:
                return "int*"
            elif inner_type == str:
                return "char**"
            elif get_origin(inner_type) == list and get_args(inner_type)[0] == int:
                return "int**"
            elif get_origin(inner_type) == list and get_args(inner_type)[0] == str:
                return "char**"
            else:
                raise Exception(f"Unsupported list type: {inner_type}")
        else:
            raise Exception(f"Unsupported type: {typing}")

    def print_wrapper(self, output):
        # Choose format specifier based on output type
        if isinstance(output, str) and output.startswith('"'):
            return f'printf("%s\\n", {output});'
        elif isinstance(output, str) and output in ["1", "0"]:  # Boolean as int
            return f'printf("%d\\n", {output});'
        elif isinstance(output, str) and output.isdigit():  # Integer
            return f'printf("%d\\n", {output});'
        elif isinstance(output, str) and (
                (output.replace(".", "").isdigit() and output.count(".") == 1) or
                (output.startswith("-") and output[1:].replace(".", "").isdigit() and output[1:].count(".") == 1)
        ):
            return f'printf("%f\\n", {output});'  # Float/double
        else:
            return f'printf("%d\\n", {output});'  # Default to int

    def codestarter_wrapper(self, code):
        return code

    def try_catch_wrapper(self, code):
        # C has no try-catch
        pass

    def add_commit(self, code, commit):
        return "\n".join([f"// {line}" for line in commit.splitlines()]) + "\n" + code

    def array_param_reduce(self, param, warpper, splitor):
        # Use param_convert to ensure proper formatting (e.g., strings with quotes)
        return warpper % splitor.join(map(self.param_convert, param))

    def func_signature_gen(self, callable):
        signature = inspect.signature(callable)
        params = list()

        for name, param in signature.parameters.items():
            if name == 'self':
                continue

            if get_origin(param.annotation) == list:
                inner_type = get_args(param.annotation)[0]

                if get_origin(inner_type) == list:
                    params.append((name, self.typing_convert(param.annotation)))
                    params.append((f'{name}Size', 'int'))  # 行数
                    params.append((f'{name}ColSize', 'int*'))  # 每行的列数数组
                else:
                    params.append((name, self.typing_convert(param.annotation)))
                    params.append((f'{name}Size', 'int'))

            else:
                params.append((name, self.typing_convert(param.annotation)))

        param_list = self.func_splitor.join([self.func_param_wrapper % (typ, name) for name, typ in params])
        func_signature = self.func_head_wrapper % (
            self.typing_convert(signature.return_annotation),
            self.func_name,
            param_list
        )
        return func_signature

    # 比较两个数
    def compare_numbers(self, a, b):
        code = []

        # Check for unsupported boolean types
        if isinstance(a, bool) or isinstance(b, bool):
            raise Exception(f"Unsupported type: bool")

        # Handle different types
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            # Numeric comparison
            code.append(f"double x = {self.param_convert(a)};")
            code.append(f"double y = {self.param_convert(b)};")
            code.append('printf("%d\\n", x == y);')

        elif isinstance(a, str) and isinstance(b, str):
            # String comparison
            code.append(f'char* x = {self.param_convert(a)};')
            code.append(f'char* y = {self.param_convert(b)};')
            code.append('#include <string.h>')
            code.append('printf("%d\\n", strcmp(x, y) == 0);')

        elif isinstance(a, list) and isinstance(b, list) and all(isinstance(i, int) for i in a) and all(
                isinstance(i, int) for i in b):
            # Integer array comparison
            code.append(f'int x[] = {self.param_convert(a)};')
            code.append(f'int y[] = {self.param_convert(b)};')
            code.append(f'int len_x = {len(a)};')
            code.append(f'int len_y = {len(b)};')
            code.append('int equal = len_x == len_y;')
            code.append('if (equal) {')
            code.append('    for (int i = 0; i < len_x; i++) {')
            code.append('        if (x[i] != y[i]) {')
            code.append('            equal = 0;')
            code.append('            break;')
            code.append('        }')
            code.append('    }')
            code.append('}')
            code.append('printf("%d\\n", equal);')

        elif isinstance(a, list) and isinstance(b, list) and all(isinstance(i, str) for i in a) and all(
                isinstance(i, str) for i in b):
            # String array comparison
            code.append(f'char* x[] = {self.param_convert(a)};')
            code.append(f'char* y[] = {self.param_convert(b)};')
            code.append(f'int len_x = {len(a)};')
            code.append(f'int len_y = {len(b)};')
            code.append('#include <string.h>')
            code.append('int equal = len_x == len_y;')
            code.append('if (equal) {')
            code.append('    for (int i = 0; i < len_x; i++) {')
            code.append('        if (strcmp(x[i], y[i]) != 0) {')
            code.append('            equal = 0;')
            code.append('            break;')
            code.append('        }')
            code.append('    }')
            code.append('}')
            code.append('printf("%d\\n", equal);')

        elif isinstance(a, list) and isinstance(b, list) and all(isinstance(i, list) for i in a) and all(
                isinstance(i, list) for i in b) and all(isinstance(j, int) for i in a for j in i) and all(
                isinstance(j, int) for i in b for j in i):
            # 2D integer array comparison
            code.append(f'int x[][100] = {self.param_convert(a)};')
            code.append(f'int y[][100] = {self.param_convert(b)};')
            code.append(f'int rows_x = {len(a)};')
            code.append(f'int rows_y = {len(b)};')
            code.append(f'int cols_x = {len(a[0]) if a else 0};')
            code.append(f'int cols_y = {len(b[0]) if b else 0};')
            code.append('int equal = rows_x == rows_y && cols_x == cols_y;')
            code.append('if (equal) {')
            code.append('    for (int i = 0; i < rows_x; i++) {')
            code.append('        for (int j = 0; j < cols_x; j++) {')
            code.append('            if (x[i][j] != y[i][j]) {')
            code.append('                equal = 0;')
            code.append('                break;')
            code.append('            }')
            code.append('        }')
            code.append('        if (!equal) break;')
            code.append('    }')
            code.append('}')
            code.append('printf("%d\\n", equal);')

        else:
            raise Exception(f"Unsupported type combination: {type(a)}, {type(b)}")

        # Wrap the code
        return f"""
    #include <stdio.h>
    int main() {{
        {''.join(code)}
        return 0;
    }}
    """

class BenchTesting:
    idx = None
    des = None
    python = None
    cangjie = None
    code_gen_prompt = "你需要按照这个函数签名给出答案："
    return_list = "retrieve_token", "token_used", "reflection", "code", "tried_times", "origin_code", "time_used"
    uncheck = False  # 是否强制要求100个样例。特殊题目不应强制。

    def check_return_list(self, ret: dict):
        not_support_list = list()
        for k in self.return_list:
            if k not in ret:
                not_support_list.append(k)
        if len(not_support_list) > 0:
            raise Exception("not return", not_support_list)

    def get_info(self, lang_type, saving_dir, iter=None, **kwargs):
        """
        初始化。配置正确返回True，配置错误返回False。
        :param lang_type:
        :param saving_dir:
        :param iter:
        :param kwargs:
        :return:
        """
        assert self.idx is not None and isinstance(self.idx, str)
        assert self.des is not None
        assert self.python is not None
        assert self.cangjie is not None

        language = None
        system_prompt = None
        evaluator = None
        if lang_type == LANG_TYPE.PYTHON:
            language = "python"
            system_prompt = f"请用Python语言编写以下代码。"
            evaluator = PYCodeEvaluator()
            self.TRUE = "True"
            self.FALSE = "False"
        elif lang_type == LANG_TYPE.CANGJIE:
            system_prompt = f"请用仓颉语言编写以下代码。"
            language = "cangjie"
            self.TRUE = "true"
            self.FALSE = "false"
            evaluator = CJCodeEvaluator()
            # check func head.[:self.cangjie.find(")")]
            func_header = self.cangjie + "return " + argv2cjparam(next(get_true_value(self))[0]) + "\n}\n main(){}"
            evaluator.run(func_header)
            if evaluator.failed and not evaluator.is_timeouted:
                self.info(evaluator.result)
                self.info("have wrong func head.break.", self.cangjie, func_header)
                return False
        else:
            if lang_type == LANG_TYPE.SCALA:
                languagehandle = ScalaLanguageHandle()
            elif lang_type == LANG_TYPE.ERLANG:
                languagehandle = ErlangLanguageHandle()
            elif lang_type == LANG_TYPE.JAVA:
                languagehandle = JavaLanguageHandle()
            else:
                raise Exception("wrong lang_type" + str(lang_type))

            language = languagehandle.language
            system_prompt = languagehandle.coding_prompt
            self.TRUE = languagehandle.TRUE
            self.FALSE = languagehandle.FALSE
            evaluator = languagehandle.get_evaluator()
            self.languagehandle = languagehandle

        self.evaluator = evaluator
        assert language is not None and system_prompt is not None
        system_prompt += "你只需要补全为你提供的函数，请不要输出其他任何内容。不要给出解释。"
        self.code_gen_prompt += system_prompt
        if iter is not None:
            assert isinstance(iter, int)
            figerprint = "{}-{}-{}".format(iter, lang_type, [f"{k}@{v}" for k, v in kwargs.items()])
        else:
            figerprint = "{}-{}".format(lang_type, [f"{k}@{v}" for k, v in kwargs.items()])

        figerprint = to_valid_filename(figerprint)
        self.dump_path = os.path.join(saving_dir, figerprint + ".json")
        code_dir = os.path.join(saving_dir, figerprint)
        self.code_path = os.path.join(code_dir, "%s.cj" % self.idx)
        os.makedirs(code_dir, exist_ok=True)
        if os.path.isfile(self.dump_path):
            with open(self.dump_path, "r", encoding="utf-8") as f:
                self.dumping = json.load(f)
        else:
            self.dumping = dict()
        return True

    def show_results(self, lang_type: LANG_TYPE, saving_dir: str, iter=None, **kwargs):
        """
        展示结果。
        :param agent:
        :param lang_type:
        :param saving_dir:
        :param iter:
        :param kwargs:
        :return:
        """
        self.get_info(lang_type, saving_dir, iter=iter, **kwargs)
        keys = [int(k) for k in self.dumping.keys()]
        keys.sort()
        failed = list()
        scores = list()
        for p, k in enumerate(keys):
            scores.append(self.dumping[str(k)]["score"])
            failed.append(self.dumping[str(k)]["failed"])
            print("Solution" + str(p + 1), k, scores[-1], failed[-1])
        for p, k in enumerate(keys):
            if failed[p]:
                print("Solution" + str(p + 1), k, self.dumping[str(k)]["result"])

    def gen_test_code(self, agent, lang_type: LANG_TYPE, saving_dir: str, iter=None, retest=False, **kwargs):
        if not self.get_info(lang_type, saving_dir, iter=iter, **kwargs):
            return

        # if self.idx in self.dumping and "score" in self.dumping[self.idx]:  # has score.
        #     if not rewrite:  # just go away
        #         self.info("already tested, break.", self.dumping[self.idx]["score"])
        #         return
        """
        若 满分则跳过；若不满分，rewrite则重新计算。
        测试条件：
        1. 没有当前数据集指纹对应的分数的
        2. failed 且 retest 的
        """
        # has_tested
        has_tested = self.idx in self.dumping and "param_fingerprint" in self.dumping[self.idx] and \
                     self.dumping[self.idx]["param_fingerprint"] == param_fingerprint and \
                     "score" in self.dumping[self.idx] and self.dumping[self.idx]["score"] != -1
        # has_tested = has_tested and not retest
        if has_tested and not retest:
            not_timeout = self.dumping[self.idx]["result"] != "timeout"
            if not_timeout:
                self.info("already tested, break.", self.dumping[self.idx]["score"])
                return
        # if rewrite:  # 改pass@k之前的
        #     if self.idx in self.dumping and not self.dumping[self.idx]["failed"]:
        #         self.info("already tested, break.", self.dumping[self.idx]["score"])
        #         return
        # else:
        #     if self.idx in self.dumping and "score" in self.dumping[self.idx]:
        #         self.info("already tested, break.", self.dumping[self.idx]["score"])
        #         return

        if self.idx not in self.dumping:
            self.dumping[self.idx] = dict()

        solution_template = ""
        if lang_type == LANG_TYPE.PYTHON:
            solution_template = self.python.replace("self, ", "")
        elif lang_type == LANG_TYPE.CANGJIE:
            solution_template = self.cangjie
        else:
            solution_template = self.languagehandle.func_signature_gen(self.solve)

        # generating code.
        code = None
        if os.path.isfile(self.code_path) and not retest:
            with open(self.code_path, "r", encoding="utf-8") as f:
                code = f.read()
        if self.idx in self.dumping and "code" in self.dumping[self.idx]:
            if retest and self.dumping[self.idx]["failed"]:
                pass
            else:
                code = self.dumping[self.idx]["code"]
        if code is None:
            self.info("generating code...")
            res = agent.code_gen(self.des + self.code_gen_prompt + solution_template,
                                 is_cangjie=lang_type == LANG_TYPE.CANGJIE,
                                 **kwargs)
            self.check_return_list(res)
            code: str = res["code"]
            self.dumping[self.idx].update(res)
        # remove main func
        code = replace_main_func(code, replace="")
        # generating testing code
        if lang_type in (LANG_TYPE.CANGJIE, LANG_TYPE.PYTHON):
            all_code = code + "\n" + self.gen_testing_code(lang_type)
        elif lang_type == LANG_TYPE.JAVA:
            all_code = self.gen_testing_code(lang_type).replace("<<solution_code>>", code)
        elif not self.languagehandle.is_inline_testing_code:
            all_code = code + "\n" + self.gen_testing_code(lang_type)
        else:
            all_code = self.gen_testing_code(lang_type) % code
        with open(self.code_path, encoding="utf-8", mode="w") as f:
            f.write(all_code)

        # if param_fingerprint == self.dumping[self.idx]["param_fingerprint"]:
        #     if os.path.isfile(self.code_path):
        #         with open(self.code_path, encoding="utf-8", mode='r') as file:
        #             all_code = file.read()
        #         return all_code
        #     else:
        #         code = self.dumping[self.idx]["code"]
        #         all_code =
        # else:
        self.save()
        return all_code

    def gen_testing_code(self, lang_type):
        if lang_type == LANG_TYPE.CANGJIE:
            testing_code = f"\n// {param_fingerprint}\n\nmain(){{\n"
            for label, args in get_true_value(self):
                params = argv2cjparams(args)  # 转化成仓颉语言的参数
                test_line = "println(solve(%s)==%s)" % (params, argv2cjparam(label))
                test_line = "    try{%s} catch (e: Exception){println('exception')}\n" % test_line
                testing_code += test_line
            testing_code += "}"
        elif lang_type == LANG_TYPE.PYTHON:
            testing_code = f"\n# {param_fingerprint}\n\n\n"
            for label, args in get_true_value(self):
                params = args
                testing_code += "try:\n    print(solve(*%s)==%s)\nexcept Exception:\n    print('exception')\n" % (
                repr(params), repr(label))
        else:
            if self.languagehandle.is_inline_testing_code:
                testing_code = self.languagehandle.add_commit("%s", param_fingerprint)
            else:
                testing_code = self.languagehandle.add_commit("", param_fingerprint)
            for label, args in get_true_value(self):
                params = self.languagehandle.params_convert(args)
                label = self.languagehandle.param_convert(label)
                testing_code += self.languagehandle.try_catch_wrapper(
                    self.languagehandle.print_wrapper(
                        "solve(%s)==%s" % (params, label)
                    )
                )

            testing_code = self.languagehandle.codestarter_wrapper(testing_code)

        return testing_code

    def test(self, agent, lang_type: LANG_TYPE, saving_dir: str, iter=None, retest=False, **kwargs):
        """
        执行该测试样例的测试。支持预填充时执行这个，不支持时执行gen_test_code.
        :param agent:
        :param lang_type:
        :param saving_dir: where the agent dumping is in.
        :return:
        """
        all_code = self.gen_test_code(agent, lang_type, saving_dir, iter=iter, retest=retest, **kwargs)
        if all_code is None:
            if hasattr(self, "dumping"):
                return self.dumping[self.idx]["score"]
            else:
                return -1

        evaluator = self.evaluator
        evaluator.run(all_code)
        if evaluator.failed:
            true_c = 0
            false_c = 0
            exception_c = 0
            failed = True
            score = 0
        else:
            true_c = evaluator.result.count(self.TRUE)
            false_c = evaluator.result.count(self.FALSE)
            exception_c = evaluator.result.count(self.languagehandle.EXCEPTION if hasattr(self, "languagehandle") else "exception")
            if not (self.uncheck or true_c + false_c + exception_c == NUM):
                self.info(evaluator.result)
            assert self.uncheck or true_c + false_c + exception_c >= NUM, f"{true_c} + {false_c} + {exception_c} != {NUM}"
            failed = False
            score = true_c / (true_c + false_c + exception_c)

        self.dumping[self.idx].update({
            "true_c": true_c,
            "false_c": false_c,
            "exception_c": exception_c,
            "score": score,
            "failed": failed,
            "result": evaluator.result,
            "degree": self.degree,
            "param_fingerprint": param_fingerprint
        })
        # if "rewrite_times" in self.dumping[self.idx]:  # ought to be `tried times`. 增加了iter参数，就不必储存这个了。
        #     self.dumping[self.idx]["rewrite_times"] += 1
        # else:
        #     self.dumping[self.idx]["rewrite_times"] = 1
        self.save()
        return score

    def info(self, *d, end="\n", seq=" "):
        print(self.idx, '\033[1;34m' + seq.join(map(str, d)) + end + '\033[0m')

    def save(self):
        with open(self.dump_path, "w", encoding="utf-8") as f:
            json.dump(self.dumping, f, indent=4, ensure_ascii=False)


def replace_main_func(code: str, replace: str) -> str:
    code = re.sub(pattern=r"main\(\)\{.*?\}", repl=replace, string=code, flags=re.DOTALL)
    return code


if __name__ == "__main__":
    scala = CLanguageHandle()


    def test(nums: List[int]) -> str:
        pass


    print(scala.func_signature_gen(test))
