import os
import random
import string
from collections import defaultdict
from typing import List

from tqdm import tqdm

from coder.sender import notify
from leetcode_gen_base import BenchTesting, gen_int, gen_lists_str, gen_lists_int, intToRoman, gen_str, NUM, \
    LEN_LIST_MIN, to_valid_filename


class Solution1(BenchTesting):
    def solve(self, nums: List[int], target: int) -> List[int]:
        hashtable = dict()
        for i, num in enumerate(nums):
            if target - num in hashtable:
                return [hashtable[target - num], i]
            hashtable[nums[i]] = i
        return []

    cangjie = "func solve(nums: Array<Int64>, target: Int64): Array<Int64> {\n"
    python = "def solve(nums: List[int], target: int) -> List[int]:\n"
    des = "给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。\n\n你可以假设每种输入只会对应一个答案，并且你不能使用两次相同的元素。\n\n你可以按任意顺序返回答案。\n\n \n\n示例 1：\n\n输入：nums = [2,7,11,15], target = 9\n输出：[0,1]\n解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。\n示例 2：\n\n输入：nums = [3,2,4], target = 6\n输出：[1,2]\n示例 3：\n\n输入：nums = [3,3], target = 6\n输出：[0,1]"
    degree = 0
    idx = "1"

    def gen(self):
        return gen_lists_int(), gen_int()


class Solution2(BenchTesting):
    def solve(self, x: int) -> bool:
        return str(x) == str(x)[::-1]

    cangjie = "func solve(x: Int64): Bool {\n"
    python = "def solve(self, x: int) -> bool:\n"
    degree = 0
    idx = "9"
    des = "给你一个整数 x ，如果 x 是一个回文整数，返回 true ；否则，返回 false 。\n\n回文数\n是指正序（从左向右）和倒序（从右向左）读都是一样的整数。\n\n例如，121 是回文，而 123 不是。\n \n\n示例 1：\n\n输入：x = 121\n输出：true\n示例 2：\n\n输入：x = -121\n输出：false\n解释：从左向右读, 为 -121 。 从右向左读, 为 121- 。因此它不是一个回文数。\n示例 3：\n\n输入：x = 10\n输出：false\n解释：从右向左读, 为 01 。因此它不是一个回文数。"

    def gen(self):
        return tuple([gen_int()])


class Solution3(BenchTesting):
    def solve(self, s: str) -> int:
        d = {'I': 1, 'IV': 3, 'V': 5, 'IX': 8, 'X': 10, 'XL': 30, 'L': 50, 'XC': 80, 'C': 100, 'CD': 300, 'D': 500,
             'CM': 800, 'M': 1000}
        return sum([d.get(s[max(i - 1, 0):i + 1], d[n]) for i, n in enumerate(s)])

    cangjie = "func solve(x: String): Int64 {\n"
    python = "def solve(self, s: str) -> int:\n"
    degree = 0
    idx = "13"
    des = "罗马数字包含以下七种字符: I， V， X， L，C，D 和 M。\n\n字符          数值\nI             1\nV             5\nX             10\nL             50\nC             100\nD             500\nM             1000\n例如， 罗马数字 2 写做 II ，即为两个并列的 1 。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。\n\n通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：\n\nI 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。\nX 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 \nC 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。\n给定一个罗马数字，将其转换成整数。\n\n \n\n示例 1:\n\n输入: s = \"III\"\n输出: 3\n示例 2:\n\n输入: s = \"IV\"\n输出: 4\n示例 3:\n\n输入: s = \"IX\"\n输出: 9\n示例 4:\n\n输入: s = \"LVIII\"\n输出: 58\n解释: L = 50, V= 5, III = 3.\n示例 5:\n\n输入: s = \"MCMXCIV\"\n输出: 1994\n解释: M = 1000, CM = 900, XC = 90, IV = 4."

    def gen(self):
        return tuple([[intToRoman(x) for x in gen_int()]])


class Solution4(BenchTesting):
    def solve(self, strs: List[str]) -> str:
        if not strs:
            return ""

        prefix, count = strs[0], len(strs)
        for i in range(1, count):
            prefix = self.lcp(prefix, strs[i])
            if not prefix:
                break

        return prefix

    cangjie = "func solve(strs: Array<String>): String {\n"
    python = "def solve(self, s: str) -> int:\n"
    degree = 0
    idx = "14"
    des = "编写一个函数来查找字符串数组中的最长公共前缀。\n\n如果不存在公共前缀，返回空字符串 \"\"。\n\n \n\n示例 1：\n\n输入：strs = [\"flower\",\"flow\",\"flight\"]\n输出：\"fl\"\n示例 2：\n\n输入：strs = [\"dog\",\"racecar\",\"car\"]\n输出：\"\"\n解释：输入不存在公共前缀。"

    def lcp(self, str1, str2):
        length, index = min(len(str1), len(str2)), 0
        while index < length and str1[index] == str2[index]:
            index += 1
        return str1[:index]

    def gen(self):
        return tuple([gen_lists_str()])


class Solution5(BenchTesting):
    cangjie = "func solve(s: String): Bool {\n"
    python = "def solve(self, s: str) -> bool:"
    degree = 0
    idx = "20"
    des = "给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。\n\n有效字符串需满足：\n\n左括号必须用相同类型的右括号闭合。\n左括号必须以正确的顺序闭合。\n每个右括号都有一个对应的相同类型的左括号。\n \n\n示例 1：\n\n输入：s = \"()\"\n\n输出：true\n\n示例 2：\n\n输入：s = \"()[]{}\"\n\n输出：true\n\n示例 3：\n\n输入：s = \"(]\"\n\n输出：false\n\n示例 4：\n\n输入：s = \"([])\"\n\n输出：true"

    def solve(self, s: str) -> bool:
        dic = {')': '(', ']': '[', '}': '{'}
        stack = []
        for i in s:
            if stack and i in dic:
                if stack[-1] == dic[i]:
                    stack.pop()
                else:
                    return False
            else:
                stack.append(i)

        return not stack

    def gen(self):
        return tuple([gen_str(vocab="(){}[]")])


class Solution6(BenchTesting):
    cangjie = "func solve(haystack: String, needle: String): Int64 {\n"
    python = "def solve(self, haystack: str, needle: str) -> int:"
    idx = "28"
    degree = "0"
    des = "给你两个字符串 haystack 和 needle ，请你在 haystack 字符串中找出 needle 字符串的第一个匹配项的下标（下标从 0 开始）。如果 needle 不是 haystack 的一部分，则返回  -1 。\n\n \n\n示例 1：\n\n输入：haystack = \"sadbutsad\", needle = \"sad\"\n输出：0\n解释：\"sad\" 在下标 0 和 6 处匹配。\n第一个匹配项的下标是 0 ，所以返回 0 。\n示例 2：\n\n输入：haystack = \"leetcode\", needle = \"leeto\"\n输出：-1\n解释：\"leeto\" 没有在 \"leetcode\" 中出现，所以返回 -1 。"

    def solve(self, haystack: str, needle: str) -> int:
        # Func: 计算偏移表
        def calShiftMat(st):
            dic = {}
            for i in range(len(st) - 1, -1, -1):
                if not dic.get(st[i]):
                    dic[st[i]] = len(st) - i
            dic["ot"] = len(st) + 1
            return dic

        # 其他情况判断
        if len(needle) > len(haystack): return -1
        if needle == "": return 0

        # 偏移表预处理
        dic = calShiftMat(needle)
        idx = 0

        while idx + len(needle) <= len(haystack):

            # 待匹配字符串
            str_cut = haystack[idx:idx + len(needle)]

            # 判断是否匹配
            if str_cut == needle:
                return idx
            else:
                # 边界处理
                if idx + len(needle) >= len(haystack):
                    return -1
                # 不匹配情况下，根据下一个字符的偏移，移动idx
                cur_c = haystack[idx + len(needle)]
                if dic.get(cur_c):
                    idx += dic[cur_c]
                else:
                    idx += dic["ot"]

        return -1 if idx + len(needle) >= len(haystack) else idx

    def gen(self):
        return gen_str(), gen_str()


class Solution7(BenchTesting):
    cangjie = "func solve(nums: Array<Int64>, target: Int64): Int64 {\n"
    python = "def solve(self, nums: List[int], target: int) -> int:"
    idx = "35"
    degree = "0"
    des = "给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。\n\n请必须使用时间复杂度为 O(log n) 的算法。\n\n \n\n示例 1:\n\n输入: nums = [1,3,5,6], target = 5\n输出: 2\n示例 2:\n\n输入: nums = [1,3,5,6], target = 2\n输出: 1\n示例 3:\n\n输入: nums = [1,3,5,6], target = 7\n输出: 4\n "

    def solve(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return left

    def gen(self):
        return gen_lists_int(is_sorted=True), gen_int()


class Solution8(BenchTesting):
    cangjie = "func solve(s: String): Int64{\n"
    python = "def solve(self, s: str) -> int:"
    idx = "58"
    degree = "0"
    des = "给你一个字符串 s，由若干单词组成，单词前后用一些空格字符隔开。返回字符串中 最后一个 单词的长度。\n\n单词 是指仅由字母组成、不包含任何空格字符的最大\n子字符串\n。\n\n \n\n示例 1：\n\n输入：s = \"Hello World\"\n输出：5\n解释：最后一个单词是“World”，长度为 5。\n示例 2：\n\n输入：s = \"   fly me   to   the moon  \"\n输出：4\n解释：最后一个单词是“moon”，长度为 4。\n示例 3：\n\n输入：s = \"luffy is still joyboy\"\n输出：6\n解释：最后一个单词是长度为 6 的“joyboy”。"

    def solve(self, s: str) -> int:
        i = len(s) - 1
        while s[i] == ' ':
            i -= 1

        j = i - 1
        while j >= 0 and s[j] != ' ':
            j -= 1

        return i - j

    def gen(self):
        return tuple([gen_str(vocab=string.ascii_lowercase + " ")])


class Solution9(BenchTesting):
    cangjie = "func solve(digits: Array<Int64>): Array<Int64>{\n"
    python = "def solve(self, digits: List[int]) -> List[int]:"
    idx = "66"
    degree = "0"
    des = "定一个由 整数 组成的 非空 数组所表示的非负整数，在该数的基础上加一。\n\n最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。\n\n你可以假设除了整数 0 之外，这个整数不会以零开头。\n\n \n\n示例 1：\n\n输入：digits = [1,2,3]\n输出：[1,2,4]\n解释：输入数组表示数字 123。\n示例 2：\n\n输入：digits = [4,3,2,1]\n输出：[4,3,2,2]\n解释：输入数组表示数字 4321。\n示例 3：\n\n输入：digits = [9]\n输出：[1,0]\n解释：输入数组表示数字 9。\n加 1 得到了 9 + 1 = 10。\n因此，结果应该是 [1,0]。"

    def solve(self, digits: List[int]) -> List[int]:
        for i in range(len(digits) - 1, -1, -1):
            if digits[i] < 9:
                digits[i] += 1
                return digits
            digits[i] = 0

        return [1] + [0] * len(digits)

    def gen(self):
        return tuple([gen_lists_int()])


class Solution10(BenchTesting):
    cangjie = "func solve(x: Int64):Int64{\n"
    python = "def solve(self, x: int) -> int:"
    idx = "69"
    degree = "0"
    des = "给你一个非负整数 x ，计算并返回 x 的 算术平方根 。\n\n由于返回类型是整数，结果只保留 整数部分 ，小数部分将被 舍去 。\n\n注意：不允许使用任何内置指数函数和算符，例如 pow(x, 0.5) 或者 x ** 0.5 。\n\n \n\n示例 1：\n\n输入：x = 4\n输出：2\n示例 2：\n\n输入：x = 8\n输出：2\n解释：8 的算术平方根是 2.82842..., 由于返回类型是整数，小数部分将被舍去。"

    def solve(self, x: int) -> int:
        # 开区间 (left, right)
        left, right = 0, min(x + 1, 46341)
        while left + 1 < right:  # 开区间不为空
            # 循环不变量：left^2 <= x
            # 循环不变量：right^2 > x
            m = (left + right) // 2
            if m * m <= x:
                left = m
            else:
                right = m
        # 循环结束时 left+1 == right
        # 此时 left^2 <= x 且 right^2 > x
        # 所以 left 最大的满足 m^2 <= x 的数
        return left

    def gen(self):
        return tuple([gen_int()])


class Solution11(BenchTesting):
    cangjie = "func solve(x: Int64):Int64{\n"
    python = "def solve(self, n: int) -> int:"
    idx = "70"
    degree = "0"
    des = "假设你正在爬楼梯。需要 n 阶你才能到达楼顶。\n\n每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？\n\n \n\n示例 1：\n\n输入：n = 2\n输出：2\n解释：有两种方法可以爬到楼顶。\n1. 1 阶 + 1 阶\n2. 2 阶\n示例 2：\n\n输入：n = 3\n输出：3\n解释：有三种方法可以爬到楼顶。\n1. 1 阶 + 1 阶 + 1 阶\n2. 1 阶 + 2 阶\n3. 2 阶 + 1 阶"

    def solve(self, n: int) -> int:
        a, b = 1, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b

    def gen(self):
        return tuple([gen_int(1, 20)])


class Solution12(BenchTesting):
    cangjie = "func solve(rowIndex: Int64): Array<Int64>{\n"
    python = "def solve(self, rowIndex: int) -> List[int]:"
    idx = "119"
    degree = "0"
    des = "给定一个非负索引 rowIndex，返回「杨辉三角」的第 rowIndex 行。\n\n在「杨辉三角」中，每个数是它左上方和右上方的数的和。\n\n\n\n \n\n示例 1:\n\n输入: rowIndex = 3\n输出: [1,3,3,1]\n示例 2:\n\n输入: rowIndex = 0\n输出: [1]\n示例 3:\n\n输入: rowIndex = 1\n输出: [1,1]"

    def solve(self, rowIndex: int) -> List[int]:
        f = [1] * (rowIndex + 1)
        for i in range(2, rowIndex + 1):
            for j in range(i - 1, 0, -1):
                f[j] += f[j - 1]
        return f

    def gen(self):
        return tuple([gen_int(1, 33)])


class Solution13(BenchTesting):
    cangjie = "func solve(prices: Array<Int64>): Int64{\n"
    python = "def solve(self, prices: List[int]) -> int:"
    idx = "121"
    degree = "0"
    des = "给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。\n\n你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。\n\n返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。\n\n \n\n示例 1：\n\n输入：[7,1,5,3,6,4]\n输出：5\n解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。\n     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。\n示例 2：\n\n输入：prices = [7,6,4,3,1]\n输出：0\n解释：在这种情况下, 没有交易完成, 所以最大利润为 0。"

    def solve(self, prices: List[int]) -> int:
        cost, profit = float('+inf'), 0
        for price in prices:
            cost = min(cost, price)
            profit = max(profit, price - cost)
        return profit

    def gen(self):
        return tuple([gen_lists_int()])


class Solution14(BenchTesting):
    cangjie = "func solve(prices: String): Bool{\n"
    python = "def solve(self, s: str) -> bool:"
    idx = "125"
    degree = "0"
    des = "如果在将所有大写字符转换为小写字符、并移除所有非字母数字字符之后，短语正着读和反着读都一样。则可以认为该短语是一个 回文串 。\n\n字母和数字都属于字母数字字符。\n\n给你一个字符串 s，如果它是 回文串 ，返回 true ；否则，返回 false 。\n\n \n\n示例 1：\n\n输入: s = \"A man, a plan, a canal: Panama\"\n输出：true\n解释：\"amanaplanacanalpanama\" 是回文串。\n示例 2：\n\n输入：s = \"race a car\"\n输出：false\n解释：\"raceacar\" 不是回文串。\n示例 3：\n\n输入：s = \" \"\n输出：true\n解释：在移除非字母数字字符之后，s 是一个空字符串 \"\" 。\n由于空字符串正着反着读都一样，所以是回文串。"

    def solve(self, s: str) -> bool:
        i, j = 0, len(s) - 1
        while i < j:
            if not s[i].isalnum():
                i += 1
            elif not s[j].isalnum():
                j -= 1
            elif s[i].lower() == s[j].lower():
                i += 1
                j -= 1
            else:
                return False
        return True

    def gen(self):
        return tuple([gen_str()])


class Solution15(BenchTesting):
    cangjie = "func solve(nums: Array<Int64>): Int64{\n"
    python = "def solve(self, nums: List[int]) -> int:"
    idx = "136"
    degree = "0"
    des = "给你一个 非空 整数数组 nums ，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。\n\n你必须设计并实现线性时间复杂度的算法来解决此问题，且该算法只使用常量额外空间。\n\n \n\n示例 1 ：\n\n输入：nums = [2,2,1]\n\n输出：1\n\n示例 2 ：\n\n输入：nums = [4,1,2,1,2]\n\n输出：4\n\n示例 3 ：\n\n输入：nums = [1]\n\n输出：1"

    def solve(self, nums: List[int]) -> int:
        x = 0
        for num in nums:  # 1. 遍历 nums 执行异或运算
            x ^= num
        return x

    def gen(self):
        return tuple([self.gen_params()])

    def gen_params(self):
        for _ in range(NUM):
            uncatches = list(range(1, LEN_LIST_MIN)) * 2 + [random.choice(range(LEN_LIST_MIN, LEN_LIST_MIN * 2))]
            random.shuffle(uncatches)
            yield uncatches


class Solution16(BenchTesting):
    cangjie = "func solve(n: Int64): String{\n"
    python = "def solve(self, n: int) -> str:"
    idx = "168"
    degree = "0"
    des = "给你一个整数 columnNumber ，返回它在 Excel 表中相对应的列名称。\n\n例如：\n\nA -> 1\nB -> 2\nC -> 3\n...\nZ -> 26\nAA -> 27\nAB -> 28 \n...\n \n\n示例 1：\n\n输入：columnNumber = 1\n输出：\"A\"\n示例 2：\n\n输入：columnNumber = 28\n输出：\"AB\"\n示例 3：\n\n输入：columnNumber = 701\n输出：\"ZY\"\n示例 4：\n\n输入：columnNumber = 2147483647\n输出：\"FXSHRXW\""

    def solve(self, n: int) -> str:
        res = ""
        while n:
            n, y = divmod(n, 26)
            if y == 0:
                n -= 1
                y = 26
            res = chr(y + 64) + res
        return res

    def gen(self):
        return tuple([gen_int(int_max=1000)])


class Solution17(BenchTesting):
    cangjie = "func solve(nums: Array<Int64>): Int64{\n"
    python = "def solve(self, nums: List[int]) -> int:"
    idx = "169"
    degree = "0"
    des = "给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。\n\n你可以假设数组是非空的，并且给定的数组总是存在多数元素。\n\n \n\n示例 1：\n\n输入：nums = [3,2,3]\n输出：3\n示例 2：\n\n输入：nums = [2,2,1,1,1,2,2]\n输出：2"

    def solve(self, nums: List[int]) -> int:
        votes = 0
        for num in nums:
            if votes == 0:
                x = num
            votes += 1 if num == x else -1
        return x

    def gen(self):
        return tuple([self.gen_params()])

    def gen_params(self):
        for _ in range(NUM):
            uncatches = list(range(1, LEN_LIST_MIN)) * 2 + [
                random.choice(range(LEN_LIST_MIN, LEN_LIST_MIN * 2))] * LEN_LIST_MIN * 4
            random.shuffle(uncatches)
            yield uncatches


class Solution18(BenchTesting):
    cangjie = "func solve(columnTitle: String): Int64{\n"
    python = "def solve(self, columnTitle: str) -> int:"
    idx = "171"
    degree = "0"
    des = "给你一个字符串 columnTitle ，表示 Excel 表格中的列名称。返回 该列名称对应的列序号 。\n\n例如：\n\nA -> 1\nB -> 2\nC -> 3\n...\nZ -> 26\nAA -> 27\nAB -> 28 \n...\n \n\n示例 1:\n\n输入: columnTitle = \"A\"\n输出: 1\n示例 2:\n\n输入: columnTitle = \"AB\"\n输出: 28\n示例 3:\n\n输入: columnTitle = \"ZY\"\n输出: 701"

    def solve(self, columnTitle: str) -> int:
        c = columnTitle[::-1]
        sum = 0
        for i in range(0, len(columnTitle)):
            a = ord(c[i]) - ord("A")
            sum += 26 ** i * (a + 1)
        return sum

    def gen(self):
        return tuple([gen_str(vocab=string.ascii_uppercase, length=3)])


class Solution19(BenchTesting):
    cangjie = "func solve(n: Int64): Int64{\n"
    python = "def solve(self, n: int) -> int:"
    idx = "191"
    degree = "0"
    des = "给定一个正整数 n，编写一个函数，获取一个正整数的二进制形式并返回其二进制表达式中 \n设置位\n 的个数（也被称为汉明重量）。\n\n \n\n示例 1：\n\n输入：n = 11\n输出：3\n解释：输入的二进制串 1011 中，共有 3 个设置位。\n示例 2：\n\n输入：n = 128\n输出：1\n解释：输入的二进制串 10000000 中，共有 1 个设置位。\n示例 3：\n\n输入：n = 2147483645\n输出：30\n解释：输入的二进制串 1111111111111111111111111111101 中，共有 30 个设置位。"

    def solve(self, n: int) -> int:
        res = 0
        while n:
            res += n & 1
            n >>= 1
        return res

    def gen(self):
        return tuple([gen_int()])


class Solution20(BenchTesting):
    cangjie = "func solve(n: Int64): Bool{\n"
    python = "def solve(self, n: int) -> bool:"
    idx = "202"
    degree = "0"
    des = "编写一个算法来判断一个数 n 是不是快乐数。\n\n「快乐数」 定义为：\n\n对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。\n然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。\n如果这个过程 结果为 1，那么这个数就是快乐数。\n如果 n 是 快乐数 就返回 true ；不是，则返回 false 。\n\n \n\n示例 1：\n\n输入：n = 19\n输出：true\n解释：\n12 + 92 = 82\n82 + 22 = 68\n62 + 82 = 100\n12 + 02 + 02 = 1\n示例 2：\n\n输入：n = 2\n输出：false"

    def solve(self, n: int) -> bool:
        fast = n
        slow = n

        while True:
            # 快慢指针异速前进
            fast = self.getNext(fast)
            fast = self.getNext(fast)
            slow = self.getNext(slow)

            if fast == 1:
                return True
            if fast == slow:
                return False

    # 取各位数字的平方和，即下一个数
    def getNext(self, num: int) -> int:
        sum = 0
        localNum = num

        while localNum > 0:
            unitsDigit = localNum % 10  # 取个位数字
            sum += unitsDigit * unitsDigit
            localNum //= 10

        return sum

    def gen(self):
        return tuple([gen_int()])


class Solution21(BenchTesting):
    cangjie = "func solve(s: String, t: String): Bool{\n"
    python = "def solve(self, s: str, t: str) -> bool:"
    idx = "205"
    degree = "0"
    des = "给定两个字符串 s 和 t ，判断它们是否是同构的。\n\n如果 s 中的字符可以按某种映射关系替换得到 t ，那么这两个字符串是同构的。\n\n每个出现的字符都应当映射到另一个字符，同时不改变字符的顺序。不同字符不能映射到同一个字符上，相同字符只能映射到同一个字符上，字符可以映射到自己本身。\n\n \n\n示例 1:\n\n输入：s = \"egg\", t = \"add\"\n输出：true\n示例 2：\n\n输入：s = \"foo\", t = \"bar\"\n输出：false\n示例 3：\n\n输入：s = \"paper\", t = \"title\"\n输出：true"

    def solve(self, s: str, t: str) -> bool:
        mp1, mp2 = {}, {}
        for a, b in zip(s, t):
            if a in mp1 and mp1[a] != b:
                return False
            if b in mp2 and mp2[b] != a:
                return False
            mp1[a] = b
            mp2[b] = a
        return True

    def gen(self):
        return tuple([gen_str(), gen_str()])


class Solution22(BenchTesting):
    cangjie = "func solve(nums: Array<Int64>): Bool{\n"
    python = "def solve(self, nums: List[int]) -> bool:"
    idx = "217"
    degree = "0"
    des = "给你一个整数数组 nums 。如果任一值在数组中出现 至少两次 ，返回 true ；如果数组中每个元素互不相同，返回 false 。\n \n\n示例 1：\n\n输入：nums = [1,2,3,1]\n\n输出：true\n\n解释：\n\n元素 1 在下标 0 和 3 出现。\n\n示例 2：\n\n输入：nums = [1,2,3,4]\n\n输出：false\n\n解释：\n\n所有元素都不同。\n\n示例 3：\n\n输入：nums = [1,1,1,3,3,4,3,2,4,2]\n\n输出：true"

    def solve(self, nums: List[int]) -> bool:
        return len(set(nums)) < len(nums)

    def gen(self):
        return tuple([gen_lists_int()])


class Solution23(BenchTesting):
    cangjie = "func solve(nums: Array<Int64>): Bool{\n"
    python = "def solve(self, nums: List[int]) -> bool:"
    idx = "219"
    degree = "0"
    des = "给你一个整数数组 nums 和一个整数 k ，判断数组中是否存在两个 不同的索引 i 和 j ，满足 nums[i] == nums[j] 且 abs(i - j) <= k 。如果存在，返回 true ；否则，返回 false 。\n\n \n\n示例 1：\n\n输入：nums = [1,2,3,1], k = 3\n输出：true\n示例 2：\n\n输入：nums = [1,0,1,1], k = 1\n输出：true\n示例 3：\n\n输入：nums = [1,2,3,1,2,3], k = 2\n输出：false"

    def solve(self, nums: List[int], k: int) -> bool:
        last = {}
        for i, x in enumerate(nums):
            if x in last and i - last[x] <= k:
                return True
            last[x] = i
        return False

    def gen(self):
        return tuple([gen_lists_int(), gen_int()])


class Solution24(BenchTesting):
    cangjie = "func solve(nums: Array<Int64>): Array<String>{\n"
    python = "def solve(self, nums: List[int]) -> List[str]:"
    idx = "228"
    degree = "0"
    des = "给定一个  无重复元素 的 有序 整数数组 nums 。\n\n返回 恰好覆盖数组中所有数字 的 最小有序 区间范围列表 。也就是说，nums 的每个元素都恰好被某个区间范围所覆盖，并且不存在属于某个范围但不属于 nums 的数字 x 。\n\n列表中的每个区间范围 [a,b] 应该按如下格式输出：\n\n\"a->b\" ，如果 a != b\n\"a\" ，如果 a == b\n \n\n示例 1：\n\n输入：nums = [0,1,2,4,5,7]\n输出：[\"0->2\",\"4->5\",\"7\"]\n解释：区间范围是：\n[0,2] --> \"0->2\"\n[4,5] --> \"4->5\"\n[7,7] --> \"7\"\n示例 2：\n\n输入：nums = [0,2,3,4,6,8,9]\n输出：[\"0\",\"2->4\",\"6\",\"8->9\"]\n解释：区间范围是：\n[0,0] --> \"0\"\n[2,4] --> \"2->4\"\n[6,6] --> \"6\"\n[8,9] --> \"8->9\""

    def solve(self, nums: List[int]) -> List[str]:
        ans = []
        i = 0
        n = len(nums)
        while i < n:
            start = i
            while i < n - 1 and nums[i] + 1 == nums[i + 1]:
                i += 1
            if start == i:
                ans.append(str(nums[start]))
            else:
                ans.append(f"{nums[start]}->{nums[i]}")
            i += 1
        return ans

    def gen(self):
        return tuple([gen_lists_int(is_sorted=True)])


class Solution25(BenchTesting):
    cangjie = "func solve(s: String, t: String): Bool{\n"
    python = "def solve(self, s: str, t: str) -> bool:"
    idx = "242"
    degree = "0"
    des = "给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的 \n字母异位词\n。字母异位词是通过重新排列不同单词或短语的字母而形成的单词或短语，并使用所有原字母一次。\n\n \n\n示例 1:\n\n输入: s = \"anagram\", t = \"nagaram\"\n输出: true\n示例 2:\n\n输入: s = \"rat\", t = \"car\"\n输出: false"

    def solve(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        dic = defaultdict(int)
        for c in s:
            dic[c] += 1
        for c in t:
            dic[c] -= 1
        for val in dic.values():
            if val != 0:
                return False
        return True

    def gen(self):
        return tuple([gen_str(), gen_str()])


class Solution26(BenchTesting):
    cangjie = "func solve(num: Int64): Int64{\n"
    python = "def solve(self, num: int) -> int:"
    idx = "258"
    degree = "0"
    des = "给定一个非负整数 num，反复将各个位上的数字相加，直到结果为一位数。返回这个结果。\n\n \n\n示例 1:\n\n输入: num = 38\n输出: 2 \n解释: 各位相加的过程为：\n38 --> 3 + 8 --> 11\n11 --> 1 + 1 --> 2\n由于 2 是一位数，所以返回 2。\n示例 2:\n\n输入: num = 0\n输出: 0"

    def solve(self, num: int) -> int:
        return (num - 1) % 9 + 1 if num else 0

    def gen(self):
        return tuple([gen_int(int_max=10e8)])


class Solution27(BenchTesting):
    cangjie = "func solve(num: Int64): Bool{\n"
    python = "def solve(self, n: int) -> bool:"
    idx = "263"
    degree = "0"
    des = "丑数 就是只包含质因数 2、3 和 5 的 正 整数。\n\n给你一个整数 n ，请你判断 n 是否为 丑数 。如果是，返回 true ；否则，返回 false 。\n\n \n\n示例 1：\n\n输入：n = 6\n输出：true\n解释：6 = 2 × 3\n示例 2：\n\n输入：n = 1\n输出：true\n解释：1 没有质因数。\n示例 3：\n\n输入：n = 14\n输出：false\n解释：14 不是丑数，因为它包含了另外一个质因数 7 。"

    def solve(self, n: int) -> bool:
        if n <= 0:
            return False
        while n % 3 == 0:
            n //= 3
        while n % 5 == 0:
            n //= 5
        return n & (n - 1) == 0

    def gen(self):
        return tuple([gen_int(int_max=10e8)])


class Solution28(BenchTesting):
    cangjie = "func solve(nums: Array<Int64>): Int64{\n"
    python = "def solve(self, nums: List[int]) -> int::"
    idx = "268"
    degree = "0"
    des = "给定一个包含 [0, n] 中 n 个数的数组 nums ，找出 [0, n] 这个范围内没有出现在数组中的那个数。\n\n \n\n示例 1：\n\n输入：nums = [3,0,1]\n\n输出：2\n\n解释：n = 3，因为有 3 个数字，所以所有的数字都在范围 [0,3] 内。2 是丢失的数字，因为它没有出现在 nums 中。\n\n示例 2：\n\n输入：nums = [0,1]\n\n输出：2\n\n解释：n = 2，因为有 2 个数字，所以所有的数字都在范围 [0,2] 内。2 是丢失的数字，因为它没有出现在 nums 中。\n\n示例 3：\n\n输入：nums = [9,6,4,2,3,5,7,0,1]\n\n输出：8\n\n解释：n = 9，因为有 9 个数字，所以所有的数字都在范围 [0,9] 内。8 是丢失的数字，因为它没有出现在 nums 中。"

    def solve(self, nums: List[int]) -> int:
        length = len(nums)
        sums = (length * (length + 1)) // 2
        for i in range(length):
            sums = sums - nums[i]
        return sums

    def gen(self):
        return tuple([self.gen_params()])

    def gen_params(self):
        for _ in range(NUM):
            L = random.randint(LEN_LIST_MIN, LEN_LIST_MIN + 500)
            nums = list(range(L + 1))
            random.shuffle(nums)
            yield nums[:-1]


class Solution29(BenchTesting):
    cangjie = "func solve(n: Int64): Bool{\n"
    python = "def solve(self, n: int) -> bool:"
    idx = "292"
    degree = "0"
    des = "你和你的朋友，两个人一起玩 Nim 游戏：\n\n桌子上有一堆石头。\n你们轮流进行自己的回合， 你作为先手 。\n每一回合，轮到的人拿掉 1 - 3 块石头。\n拿掉最后一块石头的人就是获胜者。\n假设你们每一步都是最优解。请编写一个函数，来判断你是否可以在给定石头数量为 n 的情况下赢得游戏。如果可以赢，返回 true；否则，返回 false 。\n\n \n\n示例 1：\n\n输入：n = 4\n输出：false \n解释：以下是可能的结果:\n1. 移除1颗石头。你的朋友移走了3块石头，包括最后一块。你的朋友赢了。\n2. 移除2个石子。你的朋友移走2块石头，包括最后一块。你的朋友赢了。\n3.你移走3颗石子。你的朋友移走了最后一块石头。你的朋友赢了。\n在所有结果中，你的朋友是赢家。\n示例 2：\n\n输入：n = 1\n输出：true\n示例 3：\n\n输入：n = 2\n输出：true"

    def solve(self, n: int) -> bool:
        m = 3
        return n % (m + 1) != 0

    def gen(self):
        return tuple([gen_int()])


class Solution30(BenchTesting):
    cangjie = "func solve(n: Int64): Array<Int64>{\n"
    python = "def solve(self, n: int) -> List[int]:"
    idx = "338"
    degree = "0"
    des = "给你一个整数 n ，对于 0 <= i <= n 中的每个 i ，计算其二进制表示中 1 的个数 ，返回一个长度为 n + 1 的数组 ans 作为答案。\n\n \n\n示例 1：\n\n输入：n = 2\n输出：[0,1,1]\n解释：\n0 --> 0\n1 --> 1\n2 --> 10\n示例 2：\n\n输入：n = 5\n输出：[0,1,1,2,1,2]\n解释：\n0 --> 0\n1 --> 1\n2 --> 10\n3 --> 11\n4 --> 100\n5 --> 101"

    def solve(self, n: int) -> List[int]:
        nums = [i for i in range(n + 1)]
        for i in range(n + 1):
            ant = bin(nums[i])
            nums[i] = ant[2:].count('1')
        return nums

    def gen(self):
        return tuple([gen_int(int_max=100)])


def solutions(nums=30):
    for i in range(1, nums + 1):
        solution = globals()[f"Solution{i}"]()
        yield solution

def solutions_test(nums=1):
    for i in range(1, nums + 1):
        solution = globals()[f"Solution{i}"]()
        yield solution
