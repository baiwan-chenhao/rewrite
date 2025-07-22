"""
testing rule.

model, idx, language --> score
"""
import json
import os
from collections import Counter, deque, defaultdict
from enum import Enum
from functools import cache
from math import factorial, inf, ceil, log

from openai import OpenAI
from sortedcontainers import SortedList

from leetcode_gen_base import gen_int, gen_lists_int, gen_lists_str, LEN_LIST_MIN, gen_str, NUM, gen_matirx_int
from typing import List, Optional
from string import ascii_lowercase
from forging_utils.coder.sender import notify

from leetcode_gen_base import LANG_TYPE, BenchTesting


class Solution1(BenchTesting):
    cangjie = "func solve(a: Array<Int64>, b: Array<Int64>) : Float32{\n"
    python = "def solve(self, a: List[int], b: List[int]) -> float:"
    idx = "4"
    des = "给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。\n\n算法的时间复杂度应该为 O(log (m+n)) 。\n\n \n\n示例 1：\n\n输入：nums1 = [1,3], nums2 = [2]\n输出：2.00000\n解释：合并数组 = [1,2,3] ，中位数 2\n示例 2：\n\n输入：nums1 = [1,2], nums2 = [3,4]\n输出：2.50000\n解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5"
    degree = 2

    def solve(self, a: List[int], b: List[int]) -> float:
        inf = float('inf')
        if len(a) > len(b):
            a, b = b, a

        m, n = len(a), len(b)
        left, right = -1, m
        while left + 1 < right:
            i = (left + right) // 2
            j = (m + n - 3) // 2 - i
            if a[i] <= b[j + 1]:
                left = i
            else:
                right = i
        i = left
        j = (m + n - 3) // 2 - i
        ai = a[i] if i >= 0 else -inf
        bj = b[j] if j >= 0 else -inf
        ai1 = a[i + 1] if i + 1 < m else inf
        bj1 = b[j + 1] if j + 1 < n else inf
        max1 = max(ai, bj)
        min2 = min(ai1, bj1)
        return max1 if (m + n) % 2 else (max1 + min2) / 2

    def gen(self):
        return gen_lists_int(), gen_lists_int()


class Solution2(BenchTesting):
    python = "def solve(self, s: str, p: str) -> bool:"
    cangjie = "func solve(s: String, p: String) : Bool{\n"
    idx = "10"
    des = "给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。\n\n'.' 匹配任意单个字符\n'*' 匹配零个或多个前面的那一个元素\n所谓匹配，是要涵盖 整个 字符串 s 的，而不是部分字符串。\n\n \n示例 1：\n\n输入：s = \"aa\", p = \"a\"\n输出：false\n解释：\"a\" 无法匹配 \"aa\" 整个字符串。\n示例 2:\n\n输入：s = \"aa\", p = \"a*\"\n输出：true\n解释：因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 \"aa\" 可被视为 'a' 重复了一次。\n示例 3：\n\n输入：s = \"ab\", p = \".*\"\n输出：true\n解释：\".*\" 表示可匹配零个或多个（'*'）任意字符（'.'）。"
    degree = 2

    def solve(self, s: str, p: str) -> bool:
        m, n = len(s) + 1, len(p) + 1
        dp = [[False] * n for _ in range(m)]
        dp[0][0] = True
        for j in range(2, n, 2):
            dp[0][j] = dp[0][j - 2] and p[j - 1] == '*'
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i][j - 2] or dp[i - 1][j] and (s[i - 1] == p[j - 2] or p[j - 2] == '.') \
                    if p[j - 1] == '*' else \
                    dp[i - 1][j - 1] and (p[j - 1] == '.' or s[i - 1] == p[j - 1])
        return dp[-1][-1]

    def gen(self):
        return gen_str(), gen_str(vocab=ascii_lowercase + ".*", length=10)


class Solution3(BenchTesting):
    cangjie = "func solve(s: String, p: Array<String>) : Array<Int64>{\n"
    python = "def solve(self, s: str, words: List[str]) -> List[int]:"
    idx = "30"
    des = "给定一个字符串 s 和一个字符串数组 words。 words 中所有字符串 长度相同。\n\n s 中的 串联子串 是指一个包含  words 中所有字符串以任意顺序排列连接起来的子串。\n\n例如，如果 words = [\"ab\",\"cd\",\"ef\"]， 那么 \"abcdef\"， \"abefcd\"，\"cdabef\"， \"cdefab\"，\"efabcd\"， 和 \"efcdab\" 都是串联子串。 \"acdbef\" 不是串联子串，因为他不是任何 words 排列的连接。\n返回所有串联子串在 s 中的开始索引。你可以以 任意顺序 返回答案。\n\n \n\n示例 1：\n\n输入：s = \"barfoothefoobarman\", words = [\"foo\",\"bar\"]\n输出：[0,9]\n解释：因为 words.length == 2 同时 words[i].length == 3，连接的子字符串的长度必须为 6。\n子串 \"barfoo\" 开始位置是 0。它是 words 中以 [\"bar\",\"foo\"] 顺序排列的连接。\n子串 \"foobar\" 开始位置是 9。它是 words 中以 [\"foo\",\"bar\"] 顺序排列的连接。\n输出顺序无关紧要。返回 [9,0] 也是可以的。\n示例 2：\n\n输入：s = \"wordgoodgoodgoodbestword\", words = [\"word\",\"good\",\"best\",\"word\"]\n输出：[]\n解释：因为 words.length == 4 并且 words[i].length == 4，所以串联子串的长度必须为 16。\ns 中没有子串长度为 16 并且等于 words 的任何顺序排列的连接。\n所以我们返回一个空数组。\n示例 3：\n\n输入：s = \"barfoofoobarthefoobarman\", words = [\"bar\",\"foo\",\"the\"]\n输出：[6,9,12]\n解释：因为 words.length == 3 并且 words[i].length == 3，所以串联子串的长度必须为 9。\n子串 \"foobarthe\" 开始位置是 6。它是 words 中以 [\"foo\",\"bar\",\"the\"] 顺序排列的连接。\n子串 \"barthefoo\" 开始位置是 9。它是 words 中以 [\"bar\",\"the\",\"foo\"] 顺序排列的连接。\n子串 \"thefoobar\" 开始位置是 12。它是 words 中以 [\"the\",\"foo\",\"bar\"] 顺序排列的连接。"
    degree = 2

    def solve(self, s: str, words: List[str]) -> List[int]:
        ans = []
        m = len(words)
        n = len(words[0])
        word_cnt = Counter(words)
        for start in range(n):
            cnt = Counter()
            cnt_right = 0
            left = start  # 滑窗左端点
            for right in range(start, len(s), n):
                rs = s[right: right + n]
                cnt[rs] += 1
                if cnt[rs] == word_cnt[rs]:
                    cnt_right += 1
                if right - left >= m * n:
                    ls = s[left: left + n]
                    if word_cnt[ls] == cnt[ls]:
                        cnt_right -= 1
                    cnt[ls] -= 1
                    left += n
                if cnt_right == len(word_cnt):
                    ans.append(left)
        return ans

    def gen(self):
        return gen_str(), gen_lists_str()


class Solution4(BenchTesting):
    cangjie = "func solve(s: String): Int64{\n"
    python = "def solve(self, s: str) -> int:"
    idx = "32"
    des = "给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。\n\n \n\n示例 1：\n\n输入：s = \"(()\"\n输出：2\n解释：最长有效括号子串是 \"()\"\n示例 2：\n\n输入：s = \")()())\"\n输出：4\n解释：最长有效括号子串是 \"()()\"\n示例 3：\n\n输入：s = \"\"\n输出：0"
    degree = 2

    def solve(self, s: str) -> int:
        stack = []
        maxL = 0
        n = len(s)
        tmp = [0] * n
        cur = 0
        for i in range(n):
            if s[i] == '(':
                stack.append(i)
            else:
                if stack:
                    j = stack.pop()
                    tmp[i], tmp[j] = 1, 1
        for num in tmp:
            if num:
                cur += 1
            else:
                maxL = max(cur, maxL)
                cur = 0
        maxL = max(cur, maxL)
        return maxL

    def gen(self):
        return gen_str(vocab="()"),


class Solution5(BenchTesting):
    cangjie = "func solve(nums: Array<Int64>): Int64{\n"
    python = "def firstMissingPositive(self, nums: List[int]) -> int:"
    idx = "41"
    degree = 2
    des = "给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。\n\n请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。\n \n\n示例 1：\n\n输入：nums = [1,2,0]\n输出：3\n解释：范围 [1,2] 中的数字都在数组中。\n示例 2：\n\n输入：nums = [3,4,-1,1]\n输出：2\n解释：1 在数组中，但 2 没有。\n示例 3：\n\n输入：nums = [7,8,9,11,12]\n输出：1\n解释：最小的正数 1 没有出现。"

    def solve(self, nums: List[int]) -> int:
        n = len(nums)
        hash_size = n + 1
        for i in range(n):
            if nums[i] <= 0 or nums[i] >= hash_size:
                nums[i] = 0
        for i in range(n):
            if nums[i] % hash_size != 0:
                pos = (nums[i] % hash_size) - 1
                nums[pos] = (nums[pos] % hash_size) + hash_size
        for i in range(n):
            if nums[i] < hash_size:
                return i + 1
        return hash_size

    def gen(self):
        return gen_lists_int(int_min=-10000, int_max=10000),


class Solution6(BenchTesting):
    cangjie = "func solve(height: Array<Int64>): Int64{\n"
    python = "def solve(self, height: List[int]) -> int:"
    idx = "42"
    degree = 2
    des = "给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。\n\n \n\n示例 1：\n\n\n\n输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]\n输出：6\n。 \n示例 2：\n\n输入：height = [4,2,0,3,2,5]\n输出：9"

    def solve(self, height: List[int]) -> int:
        n = len(height)
        pre_max = [0] * n  # pre_max[i] 表示从 height[0] 到 height[i] 的最大值
        pre_max[0] = height[0]
        for i in range(1, n):
            pre_max[i] = max(pre_max[i - 1], height[i])

        suf_max = [0] * n  # suf_max[i] 表示从 height[i] 到 height[n-1] 的最大值
        suf_max[-1] = height[-1]
        for i in range(n - 2, -1, -1):
            suf_max[i] = max(suf_max[i + 1], height[i])

        ans = 0
        for h, pre, suf in zip(height, pre_max, suf_max):
            ans += min(pre, suf) - h  # 累加每个水桶能接多少水
        return ans

    def gen(self):
        return gen_lists_int(int_min=1, int_max=10000),


class Solution7(BenchTesting):
    cangjie = "func solve(n: Int64, l: Int64): String{\n"
    python = "def solve(self, n: int, k: int) -> str:"
    idx = "60"
    degree = 2
    des = "给出集合 [1,2,3,...,n]，其所有元素共有 n! 种排列。\n\n按大小顺序列出所有排列情况，并一一标记，当 n = 3 时, 所有排列如下：\n\n\"123\"\n\"132\"\n\"213\"\n\"231\"\n\"312\"\n\"321\"\n给定 n 和 l，返回第 l 个排列。\n\n \n\n示例 1：\n\n输入：n = 3, k = l\n输出：\"213\"\n示例 2：\n\n输入：n = 4, l = 9\n输出：\"2314\"\n示例 3：\n\n输入：n = 3, l = 1\n输出：\"123\""

    def solve(self, n: int, k: int) -> str:
        nums = list(range(1, n + 1))
        k -= 1
        result = []
        for i in range(n, 0, -1):
            fact = factorial(i - 1)
            index = k // fact
            result.append(str(nums.pop(index)))
            k %= fact
        return ''.join(result)

    def gen(self):
        return gen_int(int_min=5, int_max=8), gen_int(int_min=1, int_max=24)


class Solution8(BenchTesting):
    cangjie = "func solve(heights: Array<Int64>): Int64{\n"
    python = "def solve(self, heights: List[int]) -> int:"
    idx = "84"
    degree = 2
    des = "给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。\n\n求在该柱状图中，能够勾勒出来的矩形的最大面积。\n\n \n\n示例 1:\n\n\n\n输入：heights = [2,1,5,6,2,3]\n输出：10\n解释：最大的矩形为图中红色区域，面积为 10\n示例 2：\n\n\n\n输入： heights = [2,4]\n输出： 4"

    def solve(self, heights: List[int]) -> int:
        stack = []
        heights = [0] + heights + [0]
        res = 0
        for i in range(len(heights)):
            # print(stack)
            while stack and heights[stack[-1]] > heights[i]:
                tmp = stack.pop()
                res = max(res, (i - stack[-1] - 1) * heights[tmp])
            stack.append(i)
        return res

    def gen(self):
        return gen_lists_int(),


class Solution9(BenchTesting):
    cangjie = "func solve(s: String, t: String): Int64{\n"
    python = "def solve(self, s: str, t: str) -> int:"
    idx = "115"
    degree = 2
    des = "给你两个字符串 s 和 t ，统计并返回在 s 的 子序列 中 t 出现的个数，结果需要对 109 + 7 取模。\n\n \n\n示例 1：\n\n输入：s = \"rabbbit\", t = \"rabbit\"\n输出：3\n解释：\n如下所示, 有 3 种可以从 s 中得到 \"rabbit\" 的方案。\nrabbbit\nrabbbit\nrabbbit\n示例 2：\n\n输入：s = \"babgbag\", t = \"bag\"\n输出：5\n解释：\n如下所示, 有 5 种可以从 s 中得到 \"bag\" 的方案。 \nbabgbag\nbabgbag\nbabgbag\nbabgbag\nbabgbag\n \n\n提示：\n\n1 <= s.length, t.length <= 1000\ns 和 t 由英文字母组成"

    def solve(self, s: str, t: str) -> int:
        n1 = len(s)
        n2 = len(t)
        dp = [[0] * (n1 + 1) for _ in range(n2 + 1)]
        for j in range(n1 + 1):
            dp[0][j] = 1
        for i in range(1, n2 + 1):
            for j in range(1, n1 + 1):
                if t[i - 1] == s[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + dp[i][j - 1]
                else:
                    dp[i][j] = dp[i][j - 1]
        return dp[-1][-1]

    def gen(self):
        return gen_str(), gen_str(length=5)


class Solution10(BenchTesting):
    cangjie = "func solve(prices: Array<Int64>): Int64{\n"
    python = "def solve(self, prices: List[int]) -> int:"
    idx = "123"
    degree = 2
    des = "给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。\n\n设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。\n\n注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。\n\n \n\n示例 1:\n\n输入：prices = [3,3,5,0,0,3,1,4]\n输出：6\n解释：在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。\n     随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3 。\n示例 2：\n\n输入：prices = [1,2,3,4,5]\n输出：4\n解释：在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。   \n     注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。   \n     因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。\n示例 3：\n\n输入：prices = [7,6,4,3,1] \n输出：0 \n解释：在这个情况下, 没有交易完成, 所以最大利润为 0。\n示例 4：\n\n输入：prices = [1]\n输出：0"

    def solve(self, prices: List[int]) -> int:
        k = 2
        f = [[-1000000] * 2 for _ in range(k + 2)]
        for j in range(1, k + 2):
            f[j][0] = 0
        for p in prices:
            for j in range(k + 1, 0, -1):
                f[j][0] = max(f[j][0], f[j][1] + p)
                f[j][1] = max(f[j][1], f[j - 1][0] - p)
        return f[-1][0]

    def gen(self):
        return gen_lists_int(),


class Solution11(BenchTesting):
    """
    n 个孩子站成一排。给你一个整数数组 ratings 表示每个孩子的评分。

你需要按照以下要求，给这些孩子分发糖果：

每个孩子至少分配到 1 个糖果。
相邻两个孩子评分更高的孩子会获得更多的糖果。
请你给每个孩子分发糖果，计算并返回需要准备的 最少糖果数目 。



示例 1：

输入：ratings = [1,0,2]
输出：5
解释：你可以分别给第一个、第二个、第三个孩子分发 2、1、2 颗糖果。
示例 2：

输入：ratings = [1,2,2]
输出：4
解释：你可以分别给第一个、第二个、第三个孩子分发 1、2、1 颗糖果。
     第三个孩子只得到 1 颗糖果，这满足题面中的两个条件。
    """
    cangjie = "func solve(ratings: Array<Int64>): Int64{\n"
    python = "def solve(self, ratings: List[int]) -> int:"
    idx = "125"
    degree = 2
    des = __doc__

    def solve(self, ratings: List[int]) -> int:
        left = [1 for _ in range(len(ratings))]
        right = left[:]
        for i in range(1, len(ratings)):
            if ratings[i] > ratings[i - 1]: left[i] = left[i - 1] + 1
        count = left[-1]
        for i in range(len(ratings) - 2, -1, -1):
            if ratings[i] > ratings[i + 1]: right[i] = right[i + 1] + 1
            count += max(left[i], right[i])
        return count

    def gen(self):
        return gen_lists_int(),


class Solution12(BenchTesting):
    """
    给你一个数组 points ，其中 points[i] = [xi, yi] 表示 X-Y 平面上的一个点。求最多有多少个点在同一条直线上。



示例 1：


输入：points = [[1,1],[2,2],[3,3]]
输出：3
示例 2：


输入：points = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
输出：4
    """
    cangjie = "func solve(points: Array<Array<Int64>>): Int64{\n"
    python = "def solve(self, points: List[List[int]]) -> int:"
    idx = "149"
    degree = 2
    des = __doc__

    def solve(self, points: List[List[int]]) -> int:
        n, ans = len(points), 1
        for i, x in enumerate(points):
            for j in range(i + 1, n):
                y = points[j]
                cnt = 2
                for k in range(j + 1, n):
                    p = points[k]
                    s1 = (y[1] - x[1]) * (p[0] - y[0])
                    s2 = (p[1] - y[1]) * (y[0] - x[0])
                    if s1 == s2: cnt += 1
                ans = max(ans, cnt)
        return ans

    def gen(self):
        return gen_matirx_int(m_min=2, m_max=2),


class Solution13(BenchTesting):
    """
    已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,4,4,5,6,7] 在变化后可能得到：
若旋转 4 次，则可以得到 [4,5,6,7,0,1,4]
若旋转 7 次，则可以得到 [0,1,4,4,5,6,7]
注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。

给你一个可能存在 重复 元素值的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。

你必须尽可能减少整个过程的操作步骤。



示例 1：

输入：nums = [1,3,5]
输出：1
示例 2：

输入：nums = [2,2,2,0,1]
输出：0
    """
    cangjie = "func solve(nums: Array<Int64>): Int64{\n"
    python = "def solve(self, nums: List[int]) -> int:"
    idx = "154"
    degree = 2
    des = __doc__

    def solve(self, nums: List[int]) -> int:
        i, j = 0, len(nums) - 1
        while i < j:
            m = (i + j) // 2
            if nums[m] > nums[j]:
                i = m + 1
            elif nums[m] < nums[j]:
                j = m
            else:
                j -= 1
        return nums[i]

    def gen(self):
        ls = list()
        for l in gen_lists_int(is_sorted=True):
            ls.append(l[int(LEN_LIST_MIN / 2):] + l[:int(LEN_LIST_MIN / 2)])
        return ls,


class Solution14(BenchTesting):
    """
    恶魔们抓住了公主并将她关在了地下城 dungeon 的 右下角 。地下城是由 m x n 个房间组成的二维网格。我们英勇的骑士最初被安置在 左上角 的房间里，他必须穿过地下城并通过对抗恶魔来拯救公主。

骑士的初始健康点数为一个正整数。如果他的健康点数在某一时刻降至 0 或以下，他会立即死亡。

有些房间由恶魔守卫，因此骑士在进入这些房间时会失去健康点数（若房间里的值为负整数，则表示骑士将损失健康点数）；其他房间要么是空的（房间里的值为 0），要么包含增加骑士健康点数的魔法球（若房间里的值为正整数，则表示骑士将增加健康点数）。

为了尽快解救公主，骑士决定每次只 向右 或 向下 移动一步。

返回确保骑士能够拯救到公主所需的最低初始健康点数。

注意：任何房间都可能对骑士的健康点数造成威胁，也可能增加骑士的健康点数，包括骑士进入的左上角房间以及公主被监禁的右下角房间。



示例 1：


输入：dungeon = [[-2,-3,3],[-5,-10,1],[10,30,-5]]
输出：7
解释：如果骑士遵循最佳路径：右 -> 右 -> 下 -> 下 ，则骑士的初始健康点数至少为 7 。
示例 2：

输入：dungeon = [[0]]
输出：1
    """
    cangjie = "func solve(dungeon: Array<Array<Int64>>): Int64{\n"
    python = "def solve(self, dungeon: List[List[int]]) -> int:"
    idx = "174"
    degree = 2
    des = __doc__

    def solve(self, dungeon: List[List[int]]) -> int:
        m = len(dungeon)
        n = len(dungeon[0])
        if m == 1 and n == 1:  # 1x1的地牢
            return max(1, 1 - dungeon[0][0])
        elif m == 1 and n != 1:  # 1xn的地牢
            newlist = []
            for i in range(n):
                newlist.append(-1)
            newlist[-1] = max(1, 1 - dungeon[0][-1])
            for i in range(n - 1):
                tempindex = n - i - 2
                newlist[tempindex] = max(1, newlist[tempindex + 1] - dungeon[0][tempindex])
            return newlist[0]
        elif m != 1 and n == 1:  # nx1的地牢
            newlist = []
            for i in range(m):
                newlist.append(-1)
            newlist[-1] = max(1, 1 - dungeon[-1][0])
            for i in range(m - 1):
                tempindex = m - i - 2
                newlist[tempindex] = max(1, newlist[tempindex + 1] - dungeon[tempindex][0])
            return newlist[0]
        newlist = []
        for i in range(m):
            templist = []
            for j in range(n):
                templist.append(-1)
            newlist.append(templist)
        newlist[m - 1][n - 1] = max(1, 1 - dungeon[-1][-1])
        tempinit = newlist[-1][-1]
        for i in range(n - 1):
            tempindex = n - i - 2
            newlist[-1][tempindex] = max(1, newlist[-1][tempindex + 1] - dungeon[-1][tempindex])
        for j in range(m - 1):
            tempindex = m - j - 2
            newlist[tempindex][-1] = max(1, newlist[tempindex + 1][-1] - dungeon[tempindex][-1])
        for i in range(m - 1):
            tempi = m - i - 2
            for j in range(n - 1):
                tempj = n - j - 2
                newlist[tempi][tempj] = max(1,
                                            min(newlist[tempi + 1][tempj], newlist[tempi][tempj + 1]) - dungeon[tempi][
                                                tempj])
        return newlist[0][0]

    def gen(self):
        return gen_matirx_int(int_min=-100, int_max=100),


class Solution15(BenchTesting):
    """
    给你一个整数数组 prices 和一个整数 k ，其中 prices[i] 是某支给定的股票在第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。也就是说，你最多可以买 k 次，卖 k 次。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。



示例 1：

输入：k = 2, prices = [2,4,1]
输出：2
解释：在第 1 天 (股票价格 = 2) 的时候买入，在第 2 天 (股票价格 = 4) 的时候卖出，这笔交易所能获得利润 = 4-2 = 2 。
示例 2：

输入：k = 2, prices = [3,2,6,5,0,3]
输出：7
解释：在第 2 天 (股票价格 = 2) 的时候买入，在第 3 天 (股票价格 = 6) 的时候卖出, 这笔交易所能获得利润 = 6-2 = 4 。
     随后，在第 5 天 (股票价格 = 0) 的时候买入，在第 6 天 (股票价格 = 3) 的时候卖出, 这笔交易所能获得利润 = 3-0 = 3 。
    """
    cangjie = "func solve(k: Int64, prices: Array<Int64>): Int64{\n"
    python = "def solve(self, k: int, prices: List[int]) -> int:"
    idx = "188"
    degree = 2
    des = __doc__

    def solve(self, k: int, prices: List[int]) -> int:
        n = len(prices)

        @cache  # 缓存装饰器，避免重复计算 dfs 的结果（记忆化）
        def dfs(i: int, j: int, hold: bool) -> int:
            if j < 0:
                return -inf
            if i < 0:
                return -inf if hold else 0
            if hold:
                return max(dfs(i - 1, j, True), dfs(i - 1, j - 1, False) - prices[i])
            return max(dfs(i - 1, j, False), dfs(i - 1, j, True) + prices[i])

        return dfs(n - 1, k, False)

    def gen(self):
        return gen_int(int_max=LEN_LIST_MIN - 1), gen_lists_int()


class Solution16(BenchTesting):
    """
    给你一个整数数组 nums 和两个整数 indexDiff 和 valueDiff 。

找出满足下述条件的下标对 (i, j)：

i != j,
abs(i - j) <= indexDiff
abs(nums[i] - nums[j]) <= valueDiff
如果存在，返回 true ；否则，返回 false 。



示例 1：

输入：nums = [1,2,3,1], indexDiff = 3, valueDiff = 0
输出：true
解释：可以找出 (i, j) = (0, 3) 。
满足下述 3 个条件：
i != j --> 0 != 3
abs(i - j) <= indexDiff --> abs(0 - 3) <= 3
abs(nums[i] - nums[j]) <= valueDiff --> abs(1 - 1) <= 0
示例 2：

输入：nums = [1,5,9,1,5,9], indexDiff = 2, valueDiff = 3
输出：false
解释：尝试所有可能的下标对 (i, j) ，均无法满足这 3 个条件，因此返回 false 。
    """
    cangjie = "func solve(nums: Array<Int64>, k: Int64, t: Int64): Bool{\n"
    python = "def solve(self, nums: List[int], k: int, t: int) -> bool:"
    idx = "220"
    degree = 2
    des = __doc__

    def solve(self, nums: List[int], k: int, t: int) -> bool:
        if t < 0 or k < 0:
            return False
        all_buckets = {}
        bucket_size = t + 1  # 桶的大小设成t+1更加方便
        for i in range(len(nums)):
            bucket_num = nums[i] // bucket_size  # 放入哪个桶

            if bucket_num in all_buckets:  # 桶中已经有元素了
                return True

            all_buckets[bucket_num] = nums[i]  # 把nums[i]放入桶中

            if (bucket_num - 1) in all_buckets and abs(all_buckets[bucket_num - 1] - nums[i]) <= t:  # 检查前一个桶
                return True

            if (bucket_num + 1) in all_buckets and abs(all_buckets[bucket_num + 1] - nums[i]) <= t:  # 检查后一个桶
                return True

            # 如果不构成返回条件，那么当i >= k 的时候就要删除旧桶了，以维持桶中的元素索引跟下一个i+1索引只差不超过k
            if i >= k:
                all_buckets.pop(nums[i - k] // bucket_size)

        return False

    def gen(self):
        return gen_lists_int(), gen_int(int_max=LEN_LIST_MIN - 3, int_min=1), gen_int()


class Solution17(BenchTesting):
    """
    给定一个整数 n，计算所有小于等于 n 的非负整数中数字 1 出现的个数。



示例 1：

输入：n = 13
输出：6
示例 2：

输入：n = 0
输出：0
    """
    cangjie = "func solve(n: Int64): Int64{\n"
    python = "def solve(self, n: int) -> int:"
    idx = "233"
    degree = 2
    des = __doc__

    def solve(self, n: int) -> int:
        digit, res = 1, 0
        high, cur, low = n // 10, n % 10, 0
        while high != 0 or cur != 0:
            if cur == 0:
                res += high * digit
            elif cur == 1:
                res += high * digit + low + 1
            else:
                res += (high + 1) * digit
            low += cur * digit
            cur = high % 10
            high //= 10
            digit *= 10
        return res

    def gen(self):
        return gen_int(),


class Solution18(BenchTesting):
    """
    给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

返回 滑动窗口中的最大值 。



示例 1：

输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
示例 2：

输入：nums = [1], k = 1
输出：[1]
    """
    cangjie = "func solve(nums: Array<Int64>, k: Int64): Array<Int64>{\n"
    python = "def solve(self, nums: List[int], k: int) -> List[int]:"
    idx = "239"
    degree = 2
    des = __doc__

    def solve(self, nums: List[int], k: int) -> List[int]:
        ans = []
        q = deque()  # 双端队列
        for i, x in enumerate(nums):
            # 1. 入
            while q and nums[q[-1]] <= x:
                q.pop()  # 维护 q 的单调性
            q.append(i)  # 入队
            # 2. 出
            if i - q[0] >= k:  # 队首已经离开窗口了
                q.popleft()
            # 3. 记录答案
            if i >= k - 1:
                # 由于队首到队尾单调递减，所以窗口最大值就是队首
                ans.append(nums[q[0]])
        return ans

    def gen(self):
        return gen_lists_int(), gen_int(int_max=LEN_LIST_MIN - 1, int_min=1)


class Solution19(BenchTesting):
    """
    有 n 个气球，编号为0 到 n - 1，每个气球上都标有一个数字，这些数字存在数组 nums 中。

现在要求你戳破所有的气球。戳破第 i 个气球，你可以获得 nums[i - 1] * nums[i] * nums[i + 1] 枚硬币。 这里的 i - 1 和 i + 1 代表和 i 相邻的两个气球的序号。如果 i - 1或 i + 1 超出了数组的边界，那么就当它是一个数字为 1 的气球。

求所能获得硬币的最大数量。



示例 1：
输入：nums = [3,1,5,8]
输出：167
解释：
nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
coins =  3*1*5    +   3*5*8   +  1*3*8  + 1*8*1 = 167
示例 2：

输入：nums = [1,5]
输出：10
    """
    cangjie = "func solve(nums: Array<Int64>): Int64{\n"
    python = "def solve(self, nums: List[int]) -> int:"
    idx = "312"
    degree = 2
    des = __doc__

    def solve(self, nums: List[int]) -> int:
        nums.insert(0, 1)
        nums.insert(len(nums), 1)

        store = [[0] * (len(nums)) for i in range(len(nums))]

        def range_best(i, j):
            m = 0
            for k in range(i + 1, j):
                left = store[i][k]
                right = store[k][j]
                a = left + nums[i] * nums[k] * nums[j] + right
                if a > m:
                    m = a
            store[i][j] = m

        for n in range(2, len(nums)):
            for i in range(0, len(nums) - n):
                range_best(i, i + n)

        return store[0][len(nums) - 1]

    def gen(self):
        return gen_lists_int(),


class Solution20(BenchTesting):
    """
    给你一个整数数组 nums ，按要求返回一个新数组 counts 。数组 counts 有该性质： counts[i] 的值是  nums[i] 右侧小于 nums[i] 的元素的数量。



示例 1：

输入：nums = [5,2,6,1]
输出：[2,1,1,0]
解释：
5 的右侧有 2 个更小的元素 (2 和 1)
2 的右侧仅有 1 个更小的元素 (1)
6 的右侧有 1 个更小的元素 (1)
1 的右侧有 0 个更小的元素
示例 2：

输入：nums = [-1]
输出：[0]
示例 3：

输入：nums = [-1,-1]
输出：[0,0]
    """
    cangjie = "func solve(nums: Array<Int64>): Array<Int64>{\n"
    python = "def solve(self, nums: List[int]) -> List[int]:\n"
    idx = "315"
    degree = 2
    des = __doc__

    def solve(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [0] * n
        sl = SortedList()

        for i in range(n - 1, -1, -1):  # 反向遍历
            cnt = sl.bisect_left(nums[i])  # 找到右边比当前值小的元素个数
            res[i] = cnt  # 记入答案
            sl.add(nums[i])  # 将当前值加入有序数组中

        return res

    def gen(self):
        return gen_lists_int(int_min=-100),


class Solution21(BenchTesting):
    """
    给你两个整数数组 nums1 和 nums2，它们的长度分别为 m 和 n。数组 nums1 和 nums2 分别代表两个数各位上的数字。同时你也会得到一个整数 k。

请你利用这两个数组中的数字创建一个长度为 k <= m + n 的最大数。同一数组中数字的相对顺序必须保持不变。

返回代表答案的长度为 k 的数组。



示例 1：

输入：nums1 = [3,4,6,5], nums2 = [9,1,2,5,8,3], k = 5
输出：[9,8,6,5,3]
示例 2：

输入：nums1 = [6,7], nums2 = [6,0,4], k = 5
输出：[6,7,6,0,4]
示例 3：

输入：nums1 = [3,9], nums2 = [8,9], k = 3
输出：[9,8,9]
    """
    cangjie = "func solve(nums1: Array<Int64>, nums2: Array<Int64>, k: Int64): Array<Int64>{\n"
    python = "def solve(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:\n"
    idx = "321"
    degree = 2
    des = __doc__

    def solve(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        # 数组选k位最大自然数
        def selectMax(nums, k):
            stack = []
            remain = len(nums) - k  # 可以舍弃的元素个数
            for num in nums:
                while remain and stack and num > stack[-1]:  # 注意是while，只要还能丢，就一直丢掉小的
                    stack.pop()
                    remain -= 1
                stack.append(num)
            return stack[:k]  # 单调栈可能超出k个，比如nums一直递减的情况

        # 合并两个数组为最大自然数
        def merge(A, B):
            ans = []
            while A or B:
                bigger = max(A, B)  # python数组直接能按元素顺序比大小,利用了python特性，不然要新定义一个方法
                ans.append(bigger[0])
                bigger.pop(0)  # A/B会跟着bigger变
            return ans

        # 对所有可能的两数组长度组合遍历，寻找最大数组
        maxlist = [0] * k
        for i in range(k + 1):  # i可以是0或者k，换句话说：可以只用到一个数组
            if i <= len(nums1) and k - i <= len(nums2):
                maxlist = max(maxlist, merge(selectMax(nums1, i), selectMax(nums2, k - i)))  # 还是利用了python特性
        return maxlist

    def gen(self):
        return gen_lists_int(int_min=1), gen_lists_int(int_min=1), gen_int(int_max=LEN_LIST_MIN)


class Solution22(BenchTesting):
    """
    给你一个整数数组 nums 以及两个整数 lower 和 upper 。求数组中，值位于范围 [lower, upper] （包含 lower 和 upper）之内的 区间和的个数 。

区间和 S(i, j) 表示在 nums 中，位置从 i 到 j 的元素之和，包含 i 和 j (i ≤ j)。



示例 1：
输入：nums = [-2,5,-1], lower = -2, upper = 2
输出：3
解释：存在三个区间：[0,0]、[2,2] 和 [0,2] ，对应的区间和分别是：-2 、-1 、2 。
示例 2：

输入：nums = [0], lower = 0, upper = 0
输出：1
    """
    cangjie = "func solve(nums: Array<Int64>, lower: Int64, upper: Int64): Int64{\n"
    python = "def solve(self, nums: List[int], lower: int, upper: int) -> int:\n"
    idx = "327"
    degree = 2
    des = __doc__

    def solve(self, nums: List[int], lower: int, upper: int) -> int:
        sl = SortedList([0])
        s = 0
        res = 0
        for x in nums:
            s += x
            l = sl.bisect_left(s - upper)
            r = sl.bisect_right(s - lower)
            res += r - l
            sl.add(s)
        return res

    def gen(self):
        return gen_lists_int(), gen_int(int_max=100), gen_int(int_min=101)


class Solution23(BenchTesting):
    """
    给定一个 m x n 整数矩阵 matrix ，找出其中 最长递增路径 的长度。

对于每个单元格，你可以往上，下，左，右四个方向移动。 你 不能 在 对角线 方向上移动或移动到 边界外（即不允许环绕）。



示例 1：


输入：matrix = [[9,9,4],[6,6,8],[2,1,1]]
输出：4
解释：最长递增路径为 [1, 2, 6, 9]。
示例 2：


输入：matrix = [[3,4,5],[3,2,6],[2,2,1]]
输出：4
解释：最长递增路径是 [3, 4, 5, 6]。注意不允许在对角线方向上移动。
示例 3：

输入：matrix = [[1]]
输出：1
    """
    cangjie = "func solve(matrix: Array<Array<Int64>>): Int64{\n"
    python = "def solve(self, matrix: List[List[int]]) -> int:\n"
    idx = "329"
    degree = 2
    des = __doc__

    def solve(self, matrix: List[List[int]]) -> int:
        if not matrix:
            return 0
        h, w = len(matrix), len(matrix[0])
        store = [[None] * (w) for i in range(h)]
        m = 0  # 储存max路径值

        def search_nearby(i, j):
            nonlocal m

            compare = []  # 储存可以比较的候选人
            # 上
            if i != 0 and matrix[i - 1][j] < matrix[i][j]:  # 有上边且上边小于当前数的话
                compare.append(store[i - 1][j]) if store[i - 1][j] else compare.append(search_nearby(i - 1, j))

            # 左
            if j != 0 and matrix[i][j - 1] < matrix[i][j]:  # 有左边且左边小于当前数的话
                compare.append(store[i][j - 1]) if store[i][j - 1] else compare.append(search_nearby(i, j - 1))

            # 下
            if i != h - 1 and matrix[i + 1][j] < matrix[i][j]:  # 有下边且下边小于当前数的话
                compare.append(store[i + 1][j]) if store[i + 1][j] else compare.append(search_nearby(i + 1, j))

            # 右
            if j != w - 1 and matrix[i][j + 1] < matrix[i][j]:  # 有右边且右边小于当前数的话
                compare.append(store[i][j + 1]) if store[i][j + 1] else compare.append(search_nearby(i, j + 1))

            store[i][j] = max(compare) + 1 if compare else 1
            m = max(m, store[i][j])
            return (store[i][j])

        for i in range(h):
            for j in range(w):
                if not store[i][j]:
                    search_nearby(i, j)

        return m

    def gen(self):
        return gen_matirx_int(m_max=5, n_max=5),


class Solution24(BenchTesting):
    """
    给定一个已排序的正整数数组 nums ，和一个正整数 n 。从 [1, n] 区间内选取任意个数字补充到 nums 中，使得 [1, n] 区间内的任何数字都可以用 nums 中某几个数字的和来表示。

请返回 满足上述要求的最少需要补充的数字个数 。



示例 1:

输入: nums = [1,3], n = 6
输出: 1
解释:
根据 nums 里现有的组合 [1], [3], [1,3]，可以得出 1, 3, 4。
现在如果我们将 2 添加到 nums 中， 组合变为: [1], [2], [3], [1,3], [2,3], [1,2,3]。
其和可以表示数字 1, 2, 3, 4, 5, 6，能够覆盖 [1, 6] 区间里所有的数。
所以我们最少需要添加一个数字。
示例 2:

输入: nums = [1,5,10], n = 20
输出: 2
解释: 我们需要添加 [2,4]。
示例 3:

输入: nums = [1,2,2], n = 5
输出: 0
    """
    cangjie = "func solve(nums: Array<Int64>, n: Int64): Int64{\n"
    python = "def solve(self, nums: List[int], n: int) -> int:\n"
    idx = "330"
    degree = 2
    des = __doc__

    def solve(self, nums: List[int], n: int) -> int:
        ans, s, i = 0, 1, 0
        while s <= n:
            if i < len(nums) and nums[i] <= s:
                s += nums[i]
                i += 1
            else:
                s *= 2  # 必须添加 s
                ans += 1
        return ans

    def gen(self):
        return gen_lists_int(is_sorted=True), gen_int()


class Solution25(BenchTesting):
    """
    给你一个整数数组 distance 。

从 X-Y 平面上的点 (0,0) 开始，先向北移动 distance[0] 米，然后向西移动 distance[1] 米，向南移动 distance[2] 米，向东移动 distance[3] 米，持续移动。也就是说，每次移动后你的方位会发生逆时针变化。

判断你所经过的路径是否相交。如果相交，返回 true ；否则，返回 false 。



示例 1：


输入：distance = [2,1,1,2]
输出：true
示例 2：


输入：distance = [1,2,3,4]
输出：false
示例 3：


输入：distance = [1,1,1,1]
输出：true
    """
    cangjie = "func solve(x: Array<Int64>): Bool{\n"
    python = "def solve(self, x: List[int]) -> bool:\n"
    idx = "335"
    degree = 2
    des = __doc__

    def solve(self, x: List[int]) -> bool:
        n = len(x)
        if n < 4:
            return False
        for i in range(3, n):
            if x[i] >= x[i - 2] and x[i - 1] <= x[i - 3]:
                return True
            if i > 3 and x[i - 1] == x[i - 3] and x[i] + x[i - 4] == x[i - 2]:
                return True
            if i > 4 and x[i] + x[i - 4] >= x[i - 2] and x[i - 3] - x[i - 5] <= x[i - 1] <= x[i - 3] and x[i - 2] >= x[
                i - 4] and x[i - 3] >= x[i - 5]:
                return True
        return False

    def gen(self):
        return gen_lists_int(),


class Solution26(BenchTesting):
    """
    给你一个二维整数数组 envelopes ，其中 envelopes[i] = [wi, hi] ，表示第 i 个信封的宽度和高度。

当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。

请计算 最多能有多少个 信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。

注意：不允许旋转信封。


示例 1：

输入：envelopes = [[5,4],[6,4],[6,7],[2,3]]
输出：3
解释：最多信封的个数为 3, 组合为: [2,3] => [5,4] => [6,7]。
示例 2：

输入：envelopes = [[1,1],[1,1],[1,1]]
输出：1
    """
    cangjie = "func solve(envelopes: Array<Array<Int64>>): Int64{\n"
    python = "def solve(self, envelopes: List[List[int]]) -> int:\n"
    idx = "354"
    degree = 2
    des = __doc__

    def solve(self, envelopes: List[List[int]]) -> int:
        if not envelopes:
            return 0
        N = len(envelopes)
        envelopes.sort()
        res = 0
        dp = [1] * N
        for i in range(N):
            for j in range(i):
                if envelopes[j][0] < envelopes[i][0] and envelopes[j][1] < envelopes[i][1]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    def gen(self):
        return gen_matirx_int(m_min=2, m_max=2),


class Solution27(BenchTesting):
    """
    一只青蛙想要过河。 假定河流被等分为若干个单元格，并且在每一个单元格内都有可能放有一块石子（也有可能没有）。 青蛙可以跳上石子，但是不可以跳入水中。

给你石子的位置列表 stones（用单元格序号 升序 表示）， 请判定青蛙能否成功过河（即能否在最后一步跳至最后一块石子上）。开始时， 青蛙默认已站在第一块石子上，并可以假定它第一步只能跳跃 1 个单位（即只能从单元格 1 跳至单元格 2 ）。

如果青蛙上一步跳跃了 k 个单位，那么它接下来的跳跃距离只能选择为 k - 1、k 或 k + 1 个单位。 另请注意，青蛙只能向前方（终点的方向）跳跃。



示例 1：

输入：stones = [0,1,3,5,6,8,12,17]
输出：true
解释：青蛙可以成功过河，按照如下方案跳跃：跳 1 个单位到第 2 块石子, 然后跳 2 个单位到第 3 块石子, 接着 跳 2 个单位到第 4 块石子, 然后跳 3 个单位到第 6 块石子, 跳 4 个单位到第 7 块石子, 最后，跳 5 个单位到第 8 个石子（即最后一块石子）。
示例 2：

输入：stones = [0,1,2,3,4,8,9,11]
输出：false
解释：这是因为第 5 和第 6 个石子之间的间距太大，没有可选的方案供青蛙跳跃过去。
    """
    cangjie = "func solve(stones: Array<Int64>): Bool{\n"
    python = "def solve(self, stones: List[int]) -> bool:\n"
    idx = "403"
    degree = 2
    des = __doc__

    def solve(self, stones: List[int]) -> bool:
        set_stones = set(stones)
        dp = defaultdict(set)
        dp[0] = {0}
        for s in stones:
            for step in dp[s]:
                for d in [-1, 0, 1]:
                    if step + d > 0 and s + step + d in set_stones:
                        dp[s + step + d].add(step + d)
        return len(dp[stones[-1]]) > 0

    def gen(self):
        ls = list()
        for l in gen_lists_int(is_sorted=True):
            l[0] = 0
            ls.append(l)
        return ls,


class Solution28(BenchTesting):
    """
    给定一个非负整数数组 nums 和一个整数 k ，你需要将这个数组分成 k 个非空的连续子数组，使得这 k 个子数组各自和的最大值 最小。

返回分割后最小的和的最大值。

子数组 是数组中连续的部份。



示例 1：

输入：nums = [7,2,5,10,8], k = 2
输出：18
解释：
一共有四种方法将 nums 分割为 2 个子数组。
其中最好的方式是将其分为 [7,2,5] 和 [10,8] 。
因为此时这两个子数组各自的和的最大值为18，在所有情况中最小。
示例 2：

输入：nums = [1,2,3,4,5], k = 2
输出：9
示例 3：

输入：nums = [1,4,4], k = 3
输出：4
    """
    cangjie = "func solve(nums: Array<Int64>, k: Int64): Int64{\n"
    python = "def solve(self, nums: List[int], k: int) -> int:\n"
    idx = "410"
    degree = 2
    des = __doc__

    def solve(self, nums: List[int], k: int) -> int:
        def check(mx: int) -> bool:
            cnt = 1
            s = 0
            for x in nums:
                if s + x <= mx:
                    s += x
                else:  # 新划分一段
                    if cnt == k:
                        return False
                    cnt += 1
                    s = x
            return True

        right = sum(nums)
        left = max(max(nums) - 1, (right - 1) // k)
        while left + 1 < right:
            mid = (left + right) // 2
            if check(mid):
                right = mid
            else:
                left = mid
        return right

    def gen(self):
        return gen_lists_int(), gen_int(int_max=LEN_LIST_MIN - 1, int_min=1)


class Solution29(BenchTesting):
    """
    给定整数 n 和 l，返回  [1, n] 中字典序第 l 小的数字。



示例 1:

输入: n = 13, l = 2
输出: 10
解释: 字典序的排列是 [1, 10, 11, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9]，所以第二小的数字是 10。
示例 2:

输入: n = 1, l = 1
输出: 1
    """
    cangjie = "func solve(n: Int64, l: Int64): Int64{\n"
    python = "def solve(self, n: int, l: int) -> int:\n"
    idx = "440"
    degree = 2
    des = __doc__

    def solve(self, n: int, k: int) -> int:
        def get_cnt(x, limit):
            a, b = str(x), str(limit)
            k = len(b) - len(a)
            ans = sum(10 ** i for i in range(k)) if k else 0
            ans += 10 ** k if (u := int(b[:len(a)])) > x else limit - x * 10 ** k + 1 if u == x else 0
            return ans

        ans = 1
        while k > 1:
            ans = ans + 1 if (cnt := get_cnt(ans, n)) < k else ans * 10
            k -= cnt if cnt < k else 1
        return ans

    def gen(self):
        return gen_int(int_min=5000), gen_int(int_max=4999)


class Solution30(BenchTesting):
    """
    有 buckets 桶液体，其中 正好有一桶 含有毒药，其余装的都是水。它们从外观看起来都一样。为了弄清楚哪只水桶含有毒药，你可以喂一些猪喝，通过观察猪是否会死进行判断。不幸的是，你只有 minutesToTest 分钟时间来确定哪桶液体是有毒的。

喂猪的规则如下：

选择若干活猪进行喂养
可以允许小猪同时饮用任意数量的桶中的水，并且该过程不需要时间。
小猪喝完水后，必须有 minutesToDie 分钟的冷却时间。在这段时间里，你只能观察，而不允许继续喂猪。
过了 minutesToDie 分钟后，所有喝到毒药的猪都会死去，其他所有猪都会活下来。
重复这一过程，直到时间用完。
给你桶的数目 buckets ，minutesToDie 和 minutesToTest ，返回 在规定时间内判断哪个桶有毒所需的 最小 猪数 。



示例 1：

输入：buckets = 1000, minutesToDie = 15, minutesToTest = 60
输出：5
示例 2：

输入：buckets = 4, minutesToDie = 15, minutesToTest = 15
输出：2
示例 3：

输入：buckets = 4, minutesToDie = 15, minutesToTest = 30
输出：2
    """
    cangjie = "func solve(buckets: Int64, minutesToDie: Int64, minutesToTest: Int64): Int64{\n"
    python = "def solve(self, buckets: int, minutesToDie: int, minutesToTest: int) -> int:\n"
    idx = "458"
    degree = 2
    des = __doc__

    def solve(self, buckets: int, minutesToDie: int, minutesToTest: int) -> int:
        return ceil(log(buckets, minutesToTest // minutesToDie + 1))

    def gen(self):
        return gen_int(int_min=1, int_max=1000), gen_int(int_min=1, int_max=30), gen_int(int_min=30, int_max=50)


def solutions(nums=30, begin=1):
    ss = list()
    for i in range(begin, nums + 1):
        solution = globals()[f"Solution{i}"]()
        ss.append(solution)
    return ss


