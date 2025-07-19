import os
import random
from bisect import bisect_left
from collections import defaultdict, Counter
from functools import cache
from itertools import pairwise
from math import inf, lcm
from typing import List, Tuple

from leetcode_gen_base import BenchTesting, gen_int, gen_lists_str, gen_lists_int, intToRoman, gen_str, NUM, \
    LEN_LIST_MIN, gen_matirx_int


def solutions(nums=20, begin=1):
    for i in range(begin, nums + 1):
        solution = globals()[f"Solution{i}"]()
        yield solution


# 426-421week
# 426 week
class Solution1(BenchTesting):
    """
给你一个整数数组 nums。该数组包含 n 个元素，其中 恰好 有 n - 2 个元素是 特殊数字 。剩下的 两个 元素中，一个是所有 特殊数字 的 和 ，另一个是 异常值 。

异常值 的定义是：既不是原始特殊数字之一，也不是所有特殊数字的和。

注意，特殊数字、和 以及 异常值 的下标必须 不同 ，但可以共享 相同 的值。

返回 nums 中可能的 最大异常值。


示例 1：

输入： nums = [2,3,5,10]

输出： 10

解释：

特殊数字可以是 2 和 3，因此和为 5，异常值为 10。

示例 2：

输入： nums = [-2,-1,-3,-6,4]

输出： 4

解释：

特殊数字可以是 -2、-1 和 -3，因此和为 -6，异常值为 4。

示例 3：

输入： nums = [1,1,1,1,1,5,5]

输出： 5

解释：

特殊数字可以是 1、1、1、1 和 1，因此和为 5，另一个 5 为异常值。
    """

    cangjie = "func solve(nums: Array<Int64>): Int64 {\n"
    python = "def solve(self, nums: List[int]) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3371"

    def solve(self, nums: List[int]) -> int:
        cnt = Counter(nums)
        total = sum(nums)

        ans = -inf
        for x in nums:
            cnt[x] -= 1
            if (total - x) % 2 == 0 and cnt[(total - x) // 2] > 0:
                ans = max(ans, x)
            cnt[x] += 1
        return ans

    def gen(self):
        nums = []
        for _ in range(NUM):
            length = random.randint(4, 50)
            special = [random.randint(-100, 100) for _ in range(length - 2)]
            total = sum(special)

            while True:
                outlier = random.randint(-100, 100)
                if outlier not in special and outlier != total:
                    break

            num = special + [total, outlier]
            random.shuffle(num)

            nums.append(num)

        return [nums]


# 425 week
class Solution2(BenchTesting):
    """
给你一个整数数组 nums 和 两个 整数 l 和 r。你的任务是找到一个长度在 l 和 r 之间（包含）且和大于 0 的 子数组 的 最小 和。

返回满足条件的子数组的 最小 和。如果不存在这样的子数组，则返回 -1。

子数组 是数组中的一个连续 非空 元素序列。



示例 1：

输入： nums = [3, -2, 1, 4], l = 2, r = 3

输出： 1

解释：

长度在 l = 2 和 r = 3 之间且和大于 0 的子数组有：

[3, -2] 和为 1
[1, 4] 和为 5
[3, -2, 1] 和为 2
[-2, 1, 4] 和为 3
其中，子数组 [3, -2] 的和为 1，是所有正和中最小的。因此，答案为 1。

示例 2：

输入： nums = [-2, 2, -3, 1], l = 2, r = 3

输出： -1

解释：

不存在长度在 l 和 r 之间且和大于 0 的子数组。因此，答案为 -1。

示例 3：

输入： nums = [1, 2, 3, 4], l = 2, r = 4

输出： 3

解释：

子数组 [1, 2] 的长度为 2，和为 3，是所有正和中最小的。因此，答案为 3。
    """

    cangjie = "func solve(nums: ArrayList<Int64>, l: Int64, r: Int64): Int64 {\n"
    python = "def solve(self, nums: List[int], l: int, r: int) -> int:\n"
    des = __doc__
    degree = 0
    idx = "3364"

    def solve(self, nums: List[int], l: int, r: int) -> int:
        from itertools import accumulate
        from sortedcontainers import SortedList
        ans = inf
        s = list(accumulate(nums, initial=0))
        sl = SortedList()  # sortedcontainers
        for j in range(l, len(nums) + 1):
            sl.add(s[j - l])
            k = sl.bisect_left(s[j])
            if k:
                ans = min(ans, s[j] - sl[k - 1])
            if j >= r:
                sl.remove(s[j - r])
        return -1 if ans == inf else ans

    def gen(self):
        nums_list = []
        ls = []
        rs = []

        for _ in range(NUM):
            length = random.randint(1, 100)
            nums = [random.randint(-100, 100) for _ in range(length)]
            l = random.randint(1, length)
            r = random.randint(l, length)

            nums_list.append(nums)
            ls.append(l)
            rs.append(r)

        return nums_list, ls, rs


class Solution3(BenchTesting):
    """
给你两个字符串 s 和 t（它们互为字母异位词），以及一个整数 k。

你的任务是判断是否可以将字符串 s 分割成 k 个等长的子字符串，然后重新排列这些子字符串，并以任意顺序连接它们，使得最终得到的新字符串与给定的字符串 t 相匹配。

如果可以做到，返回 true；否则，返回 false。

字母异位词 是指由另一个单词或短语的所有字母重新排列形成的单词或短语，使用所有原始字母恰好一次。

子字符串 是字符串中的一个连续 非空 字符序列。



示例 1：

输入： s = "abcd", t = "cdab", k = 2

输出： true

解释：

将 s 分割成 2 个长度为 2 的子字符串：["ab", "cd"]。
重新排列这些子字符串为 ["cd", "ab"]，然后连接它们得到 "cdab"，与 t 相匹配。
示例 2：

输入： s = "aabbcc", t = "bbaacc", k = 3

输出： true

解释：

将 s 分割成 3 个长度为 2 的子字符串：["aa", "bb", "cc"]。
重新排列这些子字符串为 ["bb", "aa", "cc"]，然后连接它们得到 "bbaacc"，与 t 相匹配。
示例 3：

输入： s = "aabbcc", t = "bbaacc", k = 2

输出： false

解释：

将 s 分割成 2 个长度为 3 的子字符串：["aab", "bcc"]。
这些子字符串无法重新排列形成 t = "bbaacc"，所以输出 false。
    """
    cangjie = "func solve(s: String, t: String, k: Int64): Bool {\n"
    python = "def solve(self, s: str, t: str, k: int) -> bool:\n"
    des = __doc__
    degree = 1
    idx = "3365"

    def solve(self, s: str, t: str, k: int) -> bool:
        n = len(s)
        k = n // k
        cnt_s = Counter(s[i: i + k] for i in range(0, n, k))
        cnt_t = Counter(t[i: i + k] for i in range(0, n, k))
        return cnt_s == cnt_t

    def gen(self):
        ss = []
        ts = []
        ks = []

        for _ in range(NUM):
            import string
            length = random.randint(1, 100)
            possible_k = [i for i in range(1, length + 1) if length % i == 0]

            k = random.choice(possible_k)

            chars = [random.choice(string.ascii_lowercase) for _ in range(length)]

            s_chars = chars.copy()
            t_chars = chars.copy()
            random.shuffle(t_chars)

            s = ''.join(s_chars)
            t = ''.join(t_chars)

            ks.append(k)
            ss.append(s)
            ts.append(t)

        return ss, ts, ks


class Solution4(BenchTesting):
    """
给你一个整数数组 nums 和三个整数 k、op1 和 op2。

你可以对 nums 执行以下操作：

操作 1：选择一个下标 i，将 nums[i] 除以 2，并 向上取整 到最接近的整数。你最多可以执行此操作 op1 次，并且每个下标最多只能执行一次。
操作 2：选择一个下标 i，仅当 nums[i] 大于或等于 k 时，从 nums[i] 中减去 k。你最多可以执行此操作 op2 次，并且每个下标最多只能执行一次。
注意： 两种操作可以应用于同一下标，但每种操作最多只能应用一次。

返回在执行任意次数的操作后，nums 中所有元素的 最小 可能 和 。



示例 1：

输入： nums = [2,8,3,19,3], k = 3, op1 = 1, op2 = 1

输出： 23

解释：

对 nums[1] = 8 应用操作 2，使 nums[1] = 5。
对 nums[3] = 19 应用操作 1，使 nums[3] = 10。
结果数组变为 [2, 5, 3, 10, 3]，在应用操作后具有最小可能和 23。
示例 2：

输入： nums = [2,4,3], k = 3, op1 = 2, op2 = 1

输出： 3

解释：

对 nums[0] = 2 应用操作 1，使 nums[0] = 1。
对 nums[1] = 4 应用操作 1，使 nums[1] = 2。
对 nums[2] = 3 应用操作 2，使 nums[2] = 0。
结果数组变为 [1, 2, 0]，在应用操作后具有最小可能和 3。
    """

    cangjie = "func solve(nums: Array<Int64>, k: Int64, op1: Int64, op2: Int64): Int64 {\n"
    python = "def solve(self, nums: List[int], k: int, op1: int, op2: int) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3366"

    def solve(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        n = len(nums)
        f = [[[0] * (op2 + 1) for _ in range(op1 + 1)] for _ in range(n + 1)]
        for i, x in enumerate(nums):
            for p in range(op1 + 1):
                for q in range(op2 + 1):
                    res = f[i][p][q] + x
                    if p:
                        res = min(res, f[i][p - 1][q] + (x + 1) // 2)
                    if q and x >= k:
                        res = min(res, f[i][p][q - 1] + x - k)
                        if p:
                            y = (x + 1) // 2 - k if (x + 1) // 2 >= k else (x - k + 1) // 2
                            res = min(res, f[i][p - 1][q - 1] + y)
                    f[i + 1][p][q] = res
        return f[n][op1][op2]

    def gen(self):
        nums = gen_lists_int(int_min=0, int_max=100)
        nums_list = []
        ks = []
        op1s = []
        op2s = []
        for num in nums:
            nums_list.append(num)

            k = random.randint(1, 100)
            ks.append(k)

            length = len(num)
            op1 = random.randint(0, length)
            op2 = random.randint(0, length)

            op1s.append(op1)
            op2s.append(op2)

        return nums_list, ks, op1s, op2s


# 424 week
class Solution5(BenchTesting):
    """
给你一个整数数组 nums 。

开始时，选择一个满足 nums[curr] == 0 的起始位置 curr ，并选择一个移动 方向 ：向左或者向右。

此后，你需要重复下面的过程：

如果 curr 超过范围 [0, n - 1] ，过程结束。
如果 nums[curr] == 0 ，沿当前方向继续移动：如果向右移，则 递增 curr ；如果向左移，则 递减 curr 。
如果 nums[curr] > 0:
将 nums[curr] 减 1 。
反转 移动方向（向左变向右，反之亦然）。
沿新方向移动一步。
如果在结束整个过程后，nums 中的所有元素都变为 0 ，则认为选出的初始位置和移动方向 有效 。

返回可能的有效选择方案数目。



示例 1：

输入：nums = [1,0,2,0,3]

输出：2

解释：

可能的有效选择方案如下：

选择 curr = 3 并向左移动。
[1,0,2,0,3] -> [1,0,2,0,3] -> [1,0,1,0,3] -> [1,0,1,0,3] -> [1,0,1,0,2] -> [1,0,1,0,2] -> [1,0,0,0,2] -> [1,0,0,0,2] -> [1,0,0,0,1] -> [1,0,0,0,1] -> [1,0,0,0,1] -> [1,0,0,0,1] -> [0,0,0,0,1] -> [0,0,0,0,1] -> [0,0,0,0,1] -> [0,0,0,0,1] -> [0,0,0,0,0].
选择 curr = 3 并向右移动。
[1,0,2,0,3] -> [1,0,2,0,3] -> [1,0,2,0,2] -> [1,0,2,0,2] -> [1,0,1,0,2] -> [1,0,1,0,2] -> [1,0,1,0,1] -> [1,0,1,0,1] -> [1,0,0,0,1] -> [1,0,0,0,1] -> [1,0,0,0,0] -> [1,0,0,0,0] -> [1,0,0,0,0] -> [1,0,0,0,0] -> [0,0,0,0,0].
示例 2：

输入：nums = [2,3,4,0,4,1,0]

输出：0

解释：

不存在有效的选择方案。

    """

    cangjie = "func solve(nums: Array<Int64>): Int64 {\n"
    python = "def solve(self, nums: List[int]) -> int:\n"
    des = __doc__
    degree = 0
    idx = "3354"

    def solve(self, nums: List[int]) -> int:
        total = sum(nums)
        ans = pre = 0
        for x in nums:
            if x:
                pre += x
            elif pre * 2 == total:
                ans += 2
            elif abs(pre * 2 - total) == 1:
                ans += 1
        return ans

    def gen(self):
        nums_list = []

        for _ in range(NUM):
            length = random.randint(1, 10)
            other_elements = [random.randint(0, 5) for _ in range(length - 1)]
            zero_pos = random.randint(0, length - 1)
            nums = other_elements[:zero_pos] + [0] + other_elements[zero_pos:]
            nums_list.append(nums)

        return [nums_list]


class Solution6(BenchTesting):
    """
给定一个长度为 n 的整数数组 nums 和一个二维数组 queries，其中 queries[i] = [li, ri]。

对于每个查询 queries[i]：

在 nums 的下标范围 [li, ri] 内选择一个下标 子集。
将选中的每个下标对应的元素值减 1。
零数组 是指所有元素都等于 0 的数组。

如果在按顺序处理所有查询后，可以将 nums 转换为 零数组 ，则返回 true，否则返回 false。



示例 1：

输入： nums = [1,0,1], queries = [[0,2]]

输出： true

解释：

对于 i = 0：
选择下标子集 [0, 2] 并将这些下标处的值减 1。
数组将变为 [0, 0, 0]，这是一个零数组。
示例 2：

输入： nums = [4,3,2,1], queries = [[1,3],[0,2]]

输出： false

解释：

对于 i = 0：
选择下标子集 [1, 2, 3] 并将这些下标处的值减 1。
数组将变为 [4, 2, 1, 0]。
对于 i = 1：
选择下标子集 [0, 1, 2] 并将这些下标处的值减 1。
数组将变为 [3, 1, 0, 0]，这不是一个零数组。
    """

    cangjie = "func solve(nums: Array<Int64>, queries: Array<Array<Int64>>): Bool {\n"
    python = "def solve(self, nums: List[int], queries: List[List[int]]) -> bool:\n"
    des = __doc__
    degree = 1
    idx = "3355"

    def solve(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        diff = [0] * (n + 1)
        for l, r in queries:
            # 区间 [l,r] 中的数都加一
            diff[l] += 1
            diff[r + 1] -= 1

        from itertools import accumulate
        for x, sum_d in zip(nums, accumulate(diff)):
            # 此时 sum_d 表示 x=nums[i] 要减掉多少
            if x > sum_d:  # x 无法变成 0
                return False
        return True

    def gen(self):
        nums_list = []
        queries_list = []

        for _ in range(NUM):
            nums_length = random.randint(1, 10)
            nums = [random.randint(0, 5) for _ in range(nums_length)]
            nums_list.append(nums)

            queries_length = random.randint(1, 10)
            queries = []
            for _ in range(queries_length):
                l = random.randint(0, nums_length - 1)
                r = random.randint(l, nums_length - 1)
                queries.append([l, r])
            queries_list.append(queries)

        return nums_list, queries_list


class Solution7(BenchTesting):
    """
给你一个长度为 n 的整数数组 nums 和一个二维数组 queries，其中 queries[i] = [li, ri, vali]。

每个 queries[i] 表示在 nums 上执行以下操作：

将 nums 中 [li, ri] 范围内的每个下标对应元素的值 最多 减少 vali。
每个下标的减少的数值可以独立选择。
零数组 是指所有元素都等于 0 的数组。

返回 k 可以取到的 最小非负 值，使得在 顺序 处理前 k 个查询后，nums 变成 零数组。如果不存在这样的 k，则返回 -1。



示例 1：

输入： nums = [2,0,2], queries = [[0,2,1],[0,2,1],[1,1,3]]

输出： 2

解释：

对于 i = 0（l = 0, r = 2, val = 1）：
在下标 [0, 1, 2] 处分别减少 [1, 0, 1]。
数组将变为 [1, 0, 1]。
对于 i = 1（l = 0, r = 2, val = 1）：
在下标 [0, 1, 2] 处分别减少 [1, 0, 1]。
数组将变为 [0, 0, 0]，这是一个零数组。因此，k 的最小值为 2。
示例 2：

输入： nums = [4,3,2,1], queries = [[1,3,2],[0,2,1]]

输出： -1

解释：

对于 i = 0（l = 1, r = 3, val = 2）：
在下标 [1, 2, 3] 处分别减少 [2, 2, 1]。
数组将变为 [4, 1, 0, 0]。
对于 i = 1（l = 0, r = 2, val = 1）：
在下标 [0, 1, 2] 处分别减少 [1, 1, 0]。
数组将变为 [3, 0, 0, 0]，这不是一个零数组。
    """
    cangjie = "func solve(nums: Array<Int64>, queries: Array<Array<Int64>>): Int64 {\n"
    python = "def solve(self, nums: List[int], queries: List[List[int]]) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3356"

    def solve(self, nums: List[int], queries: List[List[int]]) -> int:
        diff = [0] * (len(nums) + 1)
        sum_d = k = 0
        for i, (x, d) in enumerate(zip(nums, diff)):
            sum_d += d
            while k < len(queries) and sum_d < x:  # 需要添加询问，把 x 减小
                l, r, val = queries[k]
                diff[l] += val
                diff[r + 1] -= val
                if l <= i <= r:  # x 在更新范围中
                    sum_d += val
                k += 1
            if sum_d < x:  # 无法更新
                return -1
        return k

    def gen(self):
        nums_list = []
        queries_list = []

        for _ in range(NUM):
            nums_length = random.randint(1, 10)
            nums = [random.randint(0, 5) for _ in range(nums_length)]
            nums_list.append(nums)

            queries_length = random.randint(1, 10)
            queries = []
            for _ in range(queries_length):
                l = random.randint(0, nums_length - 1)
                r = random.randint(l, nums_length - 1)
                val = random.randint(0, 5)
                queries.append([l, r, val])
            queries_list.append(queries)

        return nums_list, queries_list


class Solution8(BenchTesting):
    """
给你一个整数数组 nums 。nums 中的一些值 缺失 了，缺失的元素标记为 -1 。

你需要选择 一个正 整数数对 (x, y) ，并将 nums 中每一个 缺失 元素用 x 或者 y 替换。

你的任务是替换 nums 中的所有缺失元素，最小化 替换后数组中相邻元素 绝对差值 的 最大值 。

请你返回上述要求下的 最小值 。



示例 1：

输入：nums = [1,2,-1,10,8]

输出：4

解释：

选择数对 (6, 7) ，nums 变为 [1, 2, 6, 10, 8] 。

相邻元素的绝对差值分别为：

|1 - 2| == 1
|2 - 6| == 4
|6 - 10| == 4
|10 - 8| == 2
示例 2：

输入：nums = [-1,-1,-1]

输出：0

解释：

选择数对 (4, 4) ，nums 变为 [4, 4, 4] 。

示例 3：

输入：nums = [-1,10,-1,8]

输出：1

解释：

选择数对 (11, 9) ，nums 变为 [11, 10, 9, 8] 。
    """
    cangjie = "func solve(nums: Array<Int64>): Int64 {\n"
    python = "def solve(self, nums: List[int]) -> int:\n"
    des = __doc__
    degree = 2
    idx = "3357"

    def solve(self, nums: List[int]) -> int:
        n = len(nums)
        # 和空位相邻的最小数字 min_l 和最大数字 max_r
        min_l, max_r = inf, 0
        for i, v in enumerate(nums):
            if v != -1 and (i > 0 and nums[i - 1] == -1 or i < n - 1 and nums[i + 1] == -1):
                min_l = min(min_l, v)
                max_r = max(max_r, v)

        def calc_diff(l: int, r: int, big: bool) -> int:
            d = (min(r - min_l, max_r - l) + 1) // 2
            if big:
                d = min(d, (max_r - min_l + 2) // 3)  # d 不能超过上界
            return d

        ans = 0
        pre_i = -1
        for i, v in enumerate(nums):
            if v == -1:
                continue
            if pre_i >= 0:
                if i - pre_i == 1:
                    ans = max(ans, abs(v - nums[pre_i]))
                else:
                    ans = max(ans, calc_diff(min(nums[pre_i], v), max(nums[pre_i], v), i - pre_i > 2))
            elif i > 0:
                ans = max(ans, calc_diff(v, v, False))
            pre_i = i
        if 0 <= pre_i < n - 1:
            ans = max(ans, calc_diff(nums[pre_i], nums[pre_i], False))
        return ans

    def gen(self):
        nums_list = []

        for _ in range(NUM):
            length = random.randint(2, 100)
            nums = [-1 if random.random() < 0.5 else random.randint(1, 100) for _ in range(length)]
            nums_list.append(nums)

        return [nums_list]


# 423 week
class Solution9(BenchTesting):
    """
给你一个由 n 个整数组成的数组 nums 和一个整数 k，请你确定是否存在 两个 相邻 且长度为 k 的 严格递增 子数组。具体来说，需要检查是否存在从下标 a 和 b (a < b) 开始的 两个 子数组，并满足下述全部条件：

这两个子数组 nums[a..a + k - 1] 和 nums[b..b + k - 1] 都是 严格递增 的。
这两个子数组必须是 相邻的，即 b = a + k。
如果可以找到这样的 两个 子数组，请返回 true；否则返回 false。

子数组 是数组中的一个连续 非空 的元素序列。



示例 1：

输入：nums = [2,5,7,8,9,2,3,4,3,1], k = 3

输出：true

解释：

从下标 2 开始的子数组为 [7, 8, 9]，它是严格递增的。
从下标 5 开始的子数组为 [2, 3, 4]，它也是严格递增的。
两个子数组是相邻的，因此结果为 true。
示例 2：

输入：nums = [1,2,3,4,4,4,4,5,6,7], k = 5

输出：false
    """

    cangjie = "func solve(nums: ArrayList<Int64>, k: Int64): Bool {\n"
    python = "def solve(self, nums: List[int], k: int) -> bool:\n"
    des = __doc__
    degree = 0
    idx = "3349"

    def solve(self, nums: List[int], k: int) -> bool:
        lst = [x < y for x, y in pairwise(nums)]
        return any(lst[i - k + 1: i] == [True] * (k - 1) == lst[i + 1: i + k] for i in range(k - 1, len(nums) - k + 1))

    def gen(self):
        nums = gen_lists_int(int_min=-1000, int_max=1000)
        nums = list(nums)
        ks = []

        for num in nums:
            length = len(num)
            k = random.randint(1, length // 2)
            ks.append(k)

        return nums, ks


class Solution10(BenchTesting):
    """
给你一个由 n 个整数组成的数组 nums ，请你找出 k 的 最大值，使得存在 两个 相邻 且长度为 k 的 严格递增 子数组。具体来说，需要检查是否存在从下标 a 和 b (a < b) 开始的 两个 子数组，并满足下述全部条件：

这两个子数组 nums[a..a + k - 1] 和 nums[b..b + k - 1] 都是 严格递增 的。
这两个子数组必须是 相邻的，即 b = a + k。
返回 k 的 最大可能 值。

子数组 是数组中的一个连续 非空 的元素序列。



示例 1：

输入：nums = [2,5,7,8,9,2,3,4,3,1]

输出：3

解释：

从下标 2 开始的子数组是 [7, 8, 9]，它是严格递增的。
从下标 5 开始的子数组是 [2, 3, 4]，它也是严格递增的。
这两个子数组是相邻的，因此 3 是满足题目条件的 最大 k 值。
示例 2：

输入：nums = [1,2,3,4,4,4,4,5,6,7]

输出：2

解释：

从下标 0 开始的子数组是 [1, 2]，它是严格递增的。
从下标 2 开始的子数组是 [3, 4]，它也是严格递增的。
这两个子数组是相邻的，因此 2 是满足题目条件的 最大 k 值。
    """

    cangjie = "func solve(nums: ArrayList<Int64>): Int64 {\n"
    python = "def solve(self, nums: List[int]) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3350"

    def solve(self, nums: List[int]) -> int:
        ans = pre_cnt = cnt = 0
        for i, x in enumerate(nums):
            cnt += 1
            if i == len(nums) - 1 or x >= nums[i + 1]:  # i 是严格递增段的末尾
                ans = max(ans, cnt // 2, min(pre_cnt, cnt))
                pre_cnt = cnt
                cnt = 0
        return ans

    def gen(self):
        nums = gen_lists_int(int_min=-100000, int_max=100000)

        return [nums]


class Solution11(BenchTesting):
    """
给你一个整数数组 nums。好子序列 的定义是：子序列中任意 两个 连续元素的绝对差 恰好 为 1。

子序列 是指可以通过删除某个数组的部分元素（或不删除）得到的数组，并且不改变剩余元素的顺序。

返回 nums 中所有 可能存在的 好子序列的 元素之和。

因为答案可能非常大，返回结果需要对 109 + 7 取余。

注意，长度为 1 的子序列默认为好子序列。



示例 1：

输入：nums = [1,2,1]

输出：14

解释：

好子序列包括：[1], [2], [1], [1,2], [2,1], [1,2,1]。
这些子序列的元素之和为 14。
示例 2：

输入：nums = [3,4,5]

输出：40

解释：

好子序列包括：[3], [4], [5], [3,4], [4,5], [3,4,5]。
这些子序列的元素之和为 40。
    """

    cangjie = "func solve(nums: Array<Int64>): Int64 {\n"
    python = "def solve(self, nums: List[int]) -> int:\n"
    des = __doc__
    degree = 2
    idx = "3351"

    def solve(self, nums: List[int]) -> int:
        MOD = 1_000_000_007
        f = defaultdict(int)
        cnt = defaultdict(int)
        for x in nums:
            c = cnt[x - 1] + cnt[x + 1] + 1
            f[x] = (f[x] + f[x - 1] + f[x + 1] + x * c) % MOD
            cnt[x] = (cnt[x] + c) % MOD
        return sum(f.values()) % MOD

    def gen(self):
        nums = gen_lists_int(int_min=0, int_max=100)

        return [nums]


class Solution12(BenchTesting):
    """
给你一个 二进制 字符串 s，它表示数字 n 的二进制形式。

同时，另给你一个整数 k。

如果整数 x 可以通过最多 k 次下述操作约简到 1 ，则将整数 x 称为 k-可约简 整数：

将 x 替换为其二进制表示中的置位数（即值为 1 的位）。

例如，数字 6 的二进制表示是 "110"。一次操作后，它变为 2（因为 "110" 中有两个置位）。再对 2（二进制为 "10"）进行操作后，它变为 1（因为 "10" 中有一个置位）。

返回小于 n 的正整数中有多少个是 k-可约简 整数。

由于答案可能很大，返回结果需要对 109 + 7 取余。

二进制中的置位是指二进制表示中值为 1 的位。



示例 1：

输入： s = "111", k = 1

输出： 3

解释：

n = 7。小于 7 的 1-可约简整数有 1，2 和 4。

示例 2：

输入： s = "1000", k = 2

输出： 6

解释：

n = 8。小于 8 的 2-可约简整数有 1，2，3，4，5 和 6。

示例 3：

输入： s = "1", k = 3

输出： 0

解释：

小于 n = 1 的正整数不存在，因此答案为 0。
    """

    cangjie = "func solve(s: String, k: Int64): Int64 {\n"
    python = "def solve(self, s: str, k: int) -> int:\n"
    des = __doc__
    degree = 2
    idx = "3352"

    def solve(self, s: str, k: int) -> int:
        MOD = 1_000_000_007
        n = len(s)

        @cache
        def dfs(i: int, left1: int, is_limit: bool) -> int:
            if i == n:
                return 0 if is_limit or left1 else 1
            up = int(s[i]) if is_limit else 1
            res = 0
            for d in range(min(up, left1) + 1):
                res += dfs(i + 1, left1 - d, is_limit and d == up)
            return res % MOD

        ans = 0
        f = [0] * n
        for i in range(1, n):
            f[i] = f[i.bit_count()] + 1
            if f[i] <= k:
                # 计算有多少个二进制数恰好有 i 个 1
                ans += dfs(0, i, True)
        dfs.cache_clear()  # 防止爆内存
        return ans % MOD

    def gen(self):
        s_list = []
        ks = []

        for _ in range(NUM):
            length = random.randint(1, 100)
            k = random.randint(1, 5)
            s = '1' + ''.join(random.choice(['0', '1']) for _ in range(length - 1))

            s_list.append(s)
            ks.append(k)

        return s_list, ks


# 422 week
class Solution13(BenchTesting):
    """
给你一个仅由数字 0 - 9 组成的字符串 num。如果偶数下标处的数字之和等于奇数下标处的数字之和，则认为该数字字符串是一个 平衡字符串。

如果 num 是一个 平衡字符串，则返回 true；否则，返回 false。



示例 1：

输入：num = "1234"

输出：false

解释：

偶数下标处的数字之和为 1 + 3 = 4，奇数下标处的数字之和为 2 + 4 = 6。
由于 4 不等于 6，num 不是平衡字符串。
示例 2：

输入：num = "24123"

输出：true

解释：

偶数下标处的数字之和为 2 + 1 + 3 = 6，奇数下标处的数字之和为 4 + 2 = 6。
由于两者相等，num 是平衡字符串。
    """

    cangjie = "func solve(num: String): Bool {\n"
    python = "def solve(self, num: str) -> bool:\n"
    des = __doc__
    degree = 0
    idx = "3340"

    def solve(self, num: str) -> bool:
        s = 0
        for i, c in enumerate(map(int, num)):
            s += c if i % 2 else -c
        return s == 0

    def gen(self):
        nums = []

        for _ in range(NUM):
            import string
            length = random.randint(2, 10)
            num = ''.join(random.choice(string.digits) for _ in range(length))
            nums.append(num)

        return [nums]


class Solution14(BenchTesting):
    """
有一个地窖，地窖中有 n x m 个房间，它们呈网格状排布。

给你一个大小为 n x m 的二维数组 moveTime ，其中 moveTime[i][j] 表示在这个时刻 以后 你才可以 开始 往这个房间 移动 。你在时刻 t = 0 时从房间 (0, 0) 出发，每次可以移动到 相邻 的一个房间。在 相邻 房间之间移动需要的时间为 1 秒。

请你返回到达房间 (n - 1, m - 1) 所需要的 最少 时间。

如果两个房间有一条公共边（可以是水平的也可以是竖直的），那么我们称这两个房间是 相邻 的。



示例 1：

输入：moveTime = [[0,4],[4,4]]

输出：6

解释：

需要花费的最少时间为 6 秒。

在时刻 t == 4 ，从房间 (0, 0) 移动到房间 (1, 0) ，花费 1 秒。
在时刻 t == 5 ，从房间 (1, 0) 移动到房间 (1, 1) ，花费 1 秒。
示例 2：

输入：moveTime = [[0,0,0],[0,0,0]]

输出：3

解释：

需要花费的最少时间为 3 秒。

在时刻 t == 0 ，从房间 (0, 0) 移动到房间 (1, 0) ，花费 1 秒。
在时刻 t == 1 ，从房间 (1, 0) 移动到房间 (1, 1) ，花费 1 秒。
在时刻 t == 2 ，从房间 (1, 1) 移动到房间 (1, 2) ，花费 1 秒。
示例 3：

输入：moveTime = [[0,1],[1,2]]

输出：3
    """

    cangjie = "func solve(moveTime: Array<Array<Int64>>): Int64 {\n"
    python = "def solve(self, moveTime: List[List[int]]) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3341"

    def solve(self, moveTime: List[List[int]]) -> int:
        from heapq import heappop, heappush
        # Dijkstra，并允许同一个节点的重复访问
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # 最小堆，存储元组 (t, x, y)，表示到达 (x, y) 的时间为 t
        heap = [(0, 0, 0)]

        n, m = len(moveTime), len(moveTime[0])
        # time[i][j]表示到达(i,j)的最少时间
        time = [[-1] * m for _ in range(n)]
        time[0][0] = 0  # 初始化

        # 当右下角未被访问时
        while time[-1][-1] == -1:
            t, x, y = heappop(heap)
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m and time[nx][ny] == -1:
                    new_time = max(t, moveTime[nx][ny]) + 1
                    time[nx][ny] = new_time
                    heappush(heap, (new_time, nx, ny))

        return time[n - 1][m - 1]  # 终点

    def gen(self):
        moveTime_list = []
        for _ in range(NUM):
            n = random.randint(2, 50)
            m = random.randint(2, 50)
            moveTime = [[random.randint(0, 100) for _ in range(m)] for _ in range(n)]
            moveTime_list.append(moveTime)

        return [moveTime_list]


class Solution15(BenchTesting):
    """
有一个地窖，地窖中有 n x m 个房间，它们呈网格状排布。

给你一个大小为 n x m 的二维数组 moveTime ，其中 moveTime[i][j] 表示在这个时刻 以后 你才可以 开始 往这个房间 移动 。你在时刻 t = 0 时从房间 (0, 0) 出发，每次可以移动到 相邻 的一个房间。在 相邻 房间之间移动需要的时间为：第一次花费 1 秒，第二次花费 2 秒，第三次花费 1 秒，第四次花费 2 秒……如此 往复 。

请你返回到达房间 (n - 1, m - 1) 所需要的 最少 时间。

如果两个房间有一条公共边（可以是水平的也可以是竖直的），那么我们称这两个房间是 相邻 的。



示例 1：

输入：moveTime = [[0,4],[4,4]]

输出：7

解释：

需要花费的最少时间为 7 秒。

在时刻 t == 4 ，从房间 (0, 0) 移动到房间 (1, 0) ，花费 1 秒。
在时刻 t == 5 ，从房间 (1, 0) 移动到房间 (1, 1) ，花费 2 秒。
示例 2：

输入：moveTime = [[0,0,0,0],[0,0,0,0]]

输出：6

解释：

需要花费的最少时间为 6 秒。

在时刻 t == 0 ，从房间 (0, 0) 移动到房间 (1, 0) ，花费 1 秒。
在时刻 t == 1 ，从房间 (1, 0) 移动到房间 (1, 1) ，花费 2 秒。
在时刻 t == 3 ，从房间 (1, 1) 移动到房间 (1, 2) ，花费 1 秒。
在时刻 t == 4 ，从房间 (1, 2) 移动到房间 (1, 3) ，花费 2 秒。
示例 3：

输入：moveTime = [[0,1],[1,2]]

输出：4
    """

    cangjie = "func solve(moveTime: Array<Array<Int64>>): Int64 {\n"
    python = "def solve(self, moveTime: List[List[int]]) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3342"

    def solve(self, moveTime: List[List[int]]) -> int:
        from heapq import heappop, heappush
        n, m = len(moveTime), len(moveTime[0])
        dis = [[inf] * m for _ in range(n)]
        dis[0][0] = 0
        h = [(0, 0, 0)]
        while True:
            d, i, j = heappop(h)
            if i == n - 1 and j == m - 1:
                return d
            if d > dis[i][j]:
                continue
            time = (i + j) % 2 + 1
            for x, y in (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1):  # 枚举周围四个格子
                if 0 <= x < n and 0 <= y < m:
                    new_dis = max(d, moveTime[x][y]) + time
                    if new_dis < dis[x][y]:
                        dis[x][y] = new_dis
                        heappush(h, (new_dis, x, y))

    def gen(self):
        moveTime_list = []
        for _ in range(NUM):
            n = random.randint(2, 100)
            m = random.randint(2, 100)
            moveTime = [[random.randint(0, 100) for _ in range(m)] for _ in range(n)]
            moveTime_list.append(moveTime)

        return [moveTime_list]


class Solution16(BenchTesting):
    """
给你一个字符串 num 。如果一个数字字符串的奇数位下标的数字之和与偶数位下标的数字之和相等，那么我们称这个数字字符串是 平衡的 。

请你返回 num 不同排列 中，平衡 字符串的数目。

由于答案可能很大，请你将答案对 109 + 7 取余 后返回。

一个字符串的 排列 指的是将字符串中的字符打乱顺序后连接得到的字符串。



示例 1：

输入：num = "123"

输出：2

解释：

num 的不同排列包括： "123" ，"132" ，"213" ，"231" ，"312" 和 "321" 。
它们之中，"132" 和 "231" 是平衡的。所以答案为 2 。
示例 2：

输入：num = "112"

输出：1

解释：

num 的不同排列包括："112" ，"121" 和 "211" 。
只有 "121" 是平衡的。所以答案为 1 。
示例 3：

输入：num = "12345"

输出：0

解释：

num 的所有排列都是不平衡的。所以答案为 0 。
    """

    cangjie = "func solve(num: String): Int64 {\n"
    python = "def solve(self, num: str) -> int:\n"
    des = __doc__
    degree = 2
    idx = "3343"

    MOD = 1_000_000_007
    MX = 41

    fac = [0] * MX  # f[i] = i!
    fac[0] = 1
    for i in range(1, MX):
        fac[i] = fac[i - 1] * i % MOD

    inv_f = [0] * MX  # inv_f[i] = i!^-1
    inv_f[-1] = pow(fac[-1], -1, MOD)
    for i in range(MX - 1, 0, -1):
        inv_f[i - 1] = inv_f[i] * i % MOD

    def solve(self, num: str) -> int:
        from itertools import accumulate
        cnt = [0] * 10
        total = 0
        for c in map(int, num):
            cnt[c] += 1
            total += c

        if total % 2:
            return 0

        pre = list(accumulate(cnt))

        @cache
        def dfs(i: int, left1: int, left_s: int) -> int:
            if i < 0:
                return 1 if left_s == 0 else 0
            res = 0
            c = cnt[i]
            left2 = pre[i] - left1
            for k in range(max(c - left2, 0), min(c, left1) + 1):
                if k * i > left_s:
                    break
                r = dfs(i - 1, left1 - k, left_s - k * i)
                res += r * self.inv_f[k] * self.inv_f[c - k]
            return res % self.MOD

        n = len(num)
        n1 = n // 2
        return self.fac[n1] * self.fac[n - n1] * dfs(9, n1, total // 2) % self.MOD

    def gen(self):
        import string
        nums = []
        for _ in range(NUM):
            length = random.randint(2, 80)
            num = ''.join(random.choice(string.digits) for _ in range(length))
            nums.append(num)

        return [nums]


# 421 week
class Solution17(BenchTesting):
    """
给你一个整数数组 nums。

因子得分 定义为数组所有元素的最小公倍数（LCM）与最大公约数（GCD）的 乘积。

在 最多 移除一个元素的情况下，返回 nums 的 最大因子得分。

注意，单个数字的 LCM 和 GCD 都是其本身，而 空数组 的因子得分为 0。



示例 1：

输入： nums = [2,4,8,16]

输出： 64

解释：

移除数字 2 后，剩余元素的 GCD 为 4，LCM 为 16，因此最大因子得分为 4 * 16 = 64。

示例 2：

输入： nums = [1,2,3,4,5]

输出： 60

解释：

无需移除任何元素即可获得最大因子得分 60。

示例 3：

输入： nums = [3]

输出： 9
    """

    cangjie = "func solve(nums: Array<Int64>): Int64 {\n"
    python = "def solve(self, nums: List[int]) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3334"

    def solve(self, nums: List[int]) -> int:
        from math import gcd
        n = len(nums)
        suf_gcd = [0] * (n + 1)
        suf_lcm = [0] * n + [1]
        for i in range(n - 1, -1, -1):
            suf_gcd[i] = gcd(suf_gcd[i + 1], nums[i])
            suf_lcm[i] = lcm(suf_lcm[i + 1], nums[i])

        ans = suf_gcd[0] * suf_lcm[0]  # 不移除元素
        pre_gcd, pre_lcm = 0, 1
        for i, x in enumerate(nums):  # 枚举移除 nums[i]
            ans = max(ans, gcd(pre_gcd, suf_gcd[i + 1]) * lcm(pre_lcm, suf_lcm[i + 1]))
            pre_gcd = gcd(pre_gcd, x)
            pre_lcm = lcm(pre_lcm, x)
        return ans

    def gen(self):
        nums = gen_lists_int(int_min=1, int_max=30)

        return [nums]


class Solution18(BenchTesting):
    """
给你一个字符串 s 和一个整数 t，表示要执行的 转换 次数。每次 转换 需要根据以下规则替换字符串 s 中的每个字符：

如果字符是 'z'，则将其替换为字符串 "ab"。
否则，将其替换为字母表中的下一个字符。例如，'a' 替换为 'b'，'b' 替换为 'c'，依此类推。
返回 恰好 执行 t 次转换后得到的字符串的 长度。

由于答案可能非常大，返回其对 109 + 7 取余的结果。



示例 1：

输入： s = "abcyy", t = 2

输出： 7

解释：

第一次转换 (t = 1)
'a' 变为 'b'
'b' 变为 'c'
'c' 变为 'd'
'y' 变为 'z'
'y' 变为 'z'
第一次转换后的字符串为："bcdzz"
第二次转换 (t = 2)
'b' 变为 'c'
'c' 变为 'd'
'd' 变为 'e'
'z' 变为 "ab"
'z' 变为 "ab"
第二次转换后的字符串为："cdeabab"
最终字符串长度：字符串为 "cdeabab"，长度为 7 个字符。
示例 2：

输入： s = "azbk", t = 1

输出： 5

解释：

第一次转换 (t = 1)
'a' 变为 'b'
'z' 变为 "ab"
'b' 变为 'c'
'k' 变为 'l'
第一次转换后的字符串为："babcl"
最终字符串长度：字符串为 "babcl"，长度为 5 个字符。
    """

    cangjie = "func solve(s: String, t: Int64): Int64 {\n"
    python = "def solve(self, s: str, t: int) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3335"

    def solve(self, s: str, t: int) -> int:
        lst = [0] * 26
        for k, v in Counter(s).items(): lst[ord(k) - 97] = v
        for _ in range(t): lst = [lst[-1], (lst[-1] + lst[0]) % 1000000007] + lst[1: -1]
        return sum(lst) % 1000000007

    def gen(self):
        import string
        ss = []
        ts = []

        for _ in range(NUM):
            length = random.randint(1, 100)
            chars = [random.choice(string.ascii_lowercase) for _ in range(length)]
            ss.append("".join(chars))

            t = random.randint(1, 100)
            ts.append(t)

        return ss, ts


class Solution19(BenchTesting):
    """
给你一个整数数组 nums。

请你统计所有满足以下条件的 非空 子序列 对 (seq1, seq2) 的数量：

子序列 seq1 和 seq2 不相交，意味着 nums 中 不存在 同时出现在两个序列中的下标。
seq1 元素的 GCD 等于 seq2 元素的 GCD。

返回满足条件的子序列对的总数。

由于答案可能非常大，请返回其对 109 + 7 取余 的结果。



示例 1：

输入： nums = [1,2,3,4]

输出： 10

解释：

元素 GCD 等于 1 的子序列对有：

([1, 2, 3, 4], [1, 2, 3, 4])
([1, 2, 3, 4], [1, 2, 3, 4])
([1, 2, 3, 4], [1, 2, 3, 4])
([1, 2, 3, 4], [1, 2, 3, 4])
([1, 2, 3, 4], [1, 2, 3, 4])
([1, 2, 3, 4], [1, 2, 3, 4])
([1, 2, 3, 4], [1, 2, 3, 4])
([1, 2, 3, 4], [1, 2, 3, 4])
([1, 2, 3, 4], [1, 2, 3, 4])
([1, 2, 3, 4], [1, 2, 3, 4])
示例 2：

输入： nums = [10,20,30]

输出： 2

解释：

元素 GCD 等于 10 的子序列对有：

([10, 20, 30], [10, 20, 30])
([10, 20, 30], [10, 20, 30])
示例 3：

输入： nums = [1,1,1,1]

输出： 50
    """

    cangjie = "func solve(nums: Array<Int64>): Int64 {\n"
    python = "def solve(self, nums: List[int]) -> int:\n"
    des = __doc__
    degree = 2
    idx = "3336"

    def solve(self, nums: List[int]) -> int:
        from math import gcd
        MOD = 1_000_000_007

        @cache  # 缓存装饰器，避免重复计算 dfs 的结果（记忆化）
        def dfs(i: int, j: int, k: int) -> int:
            if i < 0:
                return 1 if j == k else 0
            return (dfs(i - 1, j, k) + dfs(i - 1, gcd(j, nums[i]), k) + dfs(i - 1, j, gcd(k, nums[i]))) % MOD

        return (dfs(len(nums) - 1, 0, 0) - 1) % MOD

    def gen(self):
        nums = gen_lists_int(int_min=1, int_max=200)

        return [nums]


class Solution20(BenchTesting):
    """
给你一个由小写英文字母组成的字符串 s，一个整数 t 表示要执行的 转换 次数，以及一个长度为 26 的数组 nums。每次 转换 需要根据以下规则替换字符串 s 中的每个字符：

将 s[i] 替换为字母表中后续的 nums[s[i] - 'a'] 个连续字符。例如，如果 s[i] = 'a' 且 nums[0] = 3，则字符 'a' 转换为它后面的 3 个连续字符，结果为 "bcd"。
如果转换超过了 'z'，则 回绕 到字母表的开头。例如，如果 s[i] = 'y' 且 nums[24] = 3，则字符 'y' 转换为它后面的 3 个连续字符，结果为 "zab"。

返回 恰好 执行 t 次转换后得到的字符串的 长度。

由于答案可能非常大，返回其对 109 + 7 取余的结果。



示例 1：

输入： s = "abcyy", t = 2, nums = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2]

输出： 7

解释：

第一次转换 (t = 1)

'a' 变为 'b' 因为 nums[0] == 1
'b' 变为 'c' 因为 nums[1] == 1
'c' 变为 'd' 因为 nums[2] == 1
'y' 变为 'z' 因为 nums[24] == 1
'y' 变为 'z' 因为 nums[24] == 1
第一次转换后的字符串为: "bcdzz"
第二次转换 (t = 2)

'b' 变为 'c' 因为 nums[1] == 1
'c' 变为 'd' 因为 nums[2] == 1
'd' 变为 'e' 因为 nums[3] == 1
'z' 变为 'ab' 因为 nums[25] == 2
'z' 变为 'ab' 因为 nums[25] == 2
第二次转换后的字符串为: "cdeabab"
字符串最终长度： 字符串为 "cdeabab"，长度为 7 个字符。

示例 2：

输入： s = "azbk", t = 1, nums = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]

输出： 8

解释：

第一次转换 (t = 1)

'a' 变为 'bc' 因为 nums[0] == 2
'z' 变为 'ab' 因为 nums[25] == 2
'b' 变为 'cd' 因为 nums[1] == 2
'k' 变为 'lm' 因为 nums[10] == 2
第一次转换后的字符串为: "bcabcdlm"
字符串最终长度： 字符串为 "bcabcdlm"，长度为 8 个字符。
    """

    cangjie = "func solve(s: String, t: Int64, nums: ArrayList<Int64>): Int64 {\n"
    python = "def solve(self, s: str, t: int, nums: List[int]) -> int:\n"
    des = __doc__
    degree = 2
    idx = "3337"

    def solve(self, s: str, t: int, nums: List[int]) -> int:
        '''
        # 差分前缀和
        快速幂获取转换关系
        '''
        mod = int(1e9 + 7)

        # l,r 都是 26 * 26 的矩阵
        def merge(l, r):
            res = [[0] * 26 for _ in range(26)]
            for i in range(26):
                for j in range(26):
                    for h in range(26):
                        res[i][h] += l[i][j] * r[j][h]
                        res[i][h] %= mod
            return res

        def myPow(t):
            if t == 1:
                res = [[0] * 26 for _ in range(26)]
                for i in range(26):
                    j = i
                    # 延后 nums[i] 个
                    for _ in range(nums[i]):
                        j += 1
                        if j >= 26:
                            j = 0
                        res[i][j] += 1
                return res
            # 快速幂模板
            x = myPow(t // 2)
            res = merge(x, x)
            if t & 1:
                res = merge(res, myPow(1))
            return res

        # 计数
        cnt = [0] * 26
        for w in s:
            w = ord(w) - ord('a')
            cnt[w] += 1

        # 迭代 t 次后的转移矩阵
        dp = myPow(t)
        res = 0
        for i in range(26):
            for j in range(26):
                res += cnt[i] * dp[i][j]
                res %= mod

        return res

    def gen(self):
        import string
        ss = []
        ts = []
        nums = []

        for _ in range(NUM):
            length = random.randint(1, 100)
            chars = [random.choice(string.ascii_lowercase) for _ in range(length)]
            ss.append("".join(chars))

            t = random.randint(1, 100)
            ts.append(t)

            num = [random.randint(1, 25) for _ in range(26)]
            nums.append(num)

        return ss, ts, nums