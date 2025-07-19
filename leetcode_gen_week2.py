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

# 441-432week
# 441 week
class Solution1(BenchTesting):
    """
给你一个长度为 n 的整数数组 nums 和一个二维数组 queries ，其中 queries[i] = [li, ri, vali]。

每个 queries[i] 表示以下操作在 nums 上执行：

从数组 nums 中选择范围 [li, ri] 内的一个下标子集。
将每个选中下标处的值减去 正好 vali。
零数组 是指所有元素都等于 0 的数组。

返回使得经过前 k 个查询（按顺序执行）后，nums 转变为 零数组 的最小可能 非负 值 k。如果不存在这样的 k，返回 -1。

数组的 子集 是指从数组中选择的一些元素（可能为空）。



示例 1：

输入： nums = [2,0,2], queries = [[0,2,1],[0,2,1],[1,1,3]]

输出： 2

解释：

对于查询 0 （l = 0, r = 2, val = 1）：
将下标 [0, 2] 的值减 1。
数组变为 [1, 0, 1]。
对于查询 1 （l = 0, r = 2, val = 1）：
将下标 [0, 2] 的值减 1。
数组变为 [0, 0, 0]，这就是一个零数组。因此，最小的 k 值为 2。
示例 2：

输入： nums = [4,3,2,1], queries = [[1,3,2],[0,2,1]]

输出： -1

解释：

即使执行完所有查询，也无法使 nums 变为零数组。

示例 3：

输入： nums = [1,2,3,2,1], queries = [[0,1,1],[1,2,1],[2,3,2],[3,4,1],[4,4,1]]

输出： 4

解释：

对于查询 0 （l = 0, r = 1, val = 1）：
将下标 [0, 1] 的值减 1。
数组变为 [0, 1, 3, 2, 1]。
对于查询 1 （l = 1, r = 2, val = 1）：
将下标 [1, 2] 的值减 1。
数组变为 [0, 0, 2, 2, 1]。
对于查询 2 （l = 2, r = 3, val = 2）：
将下标 [2, 3] 的值减 2。
数组变为 [0, 0, 0, 0, 1]。
对于查询 3 （l = 3, r = 4, val = 1）：
将下标 4 的值减 1。
数组变为 [0, 0, 0, 0, 0]。因此，最小的 k 值为 4。
示例 4：

输入： nums = [1,2,3,2,6], queries = [[0,1,1],[0,2,1],[1,4,2],[4,4,4],[3,4,1],[4,4,5]]

输出： 4
    """

    cangjie = "func solve(nums: Array<Int64>, queries: Array<Array<Int64>>): Int64 {\n"
    python = "def solve(self, nums: List[int], queries: List[List[int]]) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3489"

    def solve(self, nums: List[int], queries: List[List[int]]) -> int:
        ans = 0
        for i, x in enumerate(nums):  # 每个 nums[i] 单独计算 0-1 背包
            if x == 0:
                continue
            f = [True] + [False] * x
            for k, (l, r, val) in enumerate(queries):
                if not l <= i <= r:
                    continue
                for j in range(x, val - 1, -1):
                    f[j] = f[j] or f[j - val]
                if f[x]:  # 满足要求
                    ans = max(ans, k + 1)
                    break
            else:  # 没有中途 break，说明无法满足要求
                return -1
        return ans

    def gen(self):
        nums = []
        queries = []
        for _ in range(NUM):
            num_length = random.randint(1, 10)
            num = [random.randint(0, 1000) for _ in range(num_length)]
            nums.append(num)

            query_length = random.randint(1, 1000)
            query = []
            for _ in range(query_length):
                li = random.randint(0, num_length - 1)
                ri = random.randint(li, num_length - 1)
                vali = random.randint(1, 10)
                query.append([li, ri, vali])

            queries.append(query)
        return nums, queries


class Solution2(BenchTesting):
    """
给你两个正整数 l 和 r 。如果正整数每一位上的数字的乘积可以被这些数字之和整除，则认为该整数是一个 美丽整数 。

统计并返回 l 和 r 之间（包括 l 和 r ）的 美丽整数 的数目。



示例 1：

输入：l = 10, r = 20

输出：2

解释：

范围内的美丽整数为 10 和 20 。

示例 2：

输入：l = 1, r = 15

输出：10

解释：

范围内的美丽整数为 1、2、3、4、5、6、7、8、9 和 10 。
    """

    cangjie = "func solve(l: Int64, r: Int64): Int64 {\n"
    python = "def solve(self, l: int, r: int) -> int:\n"
    des = __doc__
    degree = 2
    idx = "3490"

    def solve(self, l: int, r: int) -> int:
        low = list(map(int, str(l)))
        high = list(map(int, str(r)))
        n = len(high)
        diff_lh = n - len(low)  # 这样写无需给 low 补前导零，也无需 is_num 参数

        @cache
        def dfs(i: int, m: int, s: int, limit_low: bool, limit_high: bool) -> int:
            if i == n:
                return 1 if s and m % s == 0 else 0

            lo = low[i - diff_lh] if limit_low and i >= diff_lh else 0
            hi = high[i] if limit_high else 9

            res = 0
            if limit_low and i < diff_lh:
                res += dfs(i + 1, 1, 0, True, False)  # 什么也不填
                d = 1  # 下面循环从 1 开始
            else:
                d = lo
            # 枚举填数字 d
            for d in range(d, hi + 1):
                res += dfs(i + 1, m * d, s + d, limit_low and d == lo, limit_high and d == hi)
            return res

        return dfs(0, 1, 0, True, True)

    def gen(self):
        ls = []
        rs = []
        for _ in range(NUM):
            l = random.randint(1, 99)
            r = random.randint(l, 100)

            ls.append(l)
            rs.append(r)

        return ls, rs


# 440 week
class Solution3(BenchTesting):
    """
给你两个长度为 n 的整数数组，fruits 和 baskets，其中 fruits[i] 表示第 i 种水果的 数量，baskets[j] 表示第 j 个篮子的 容量。

你需要对 fruits 数组从左到右按照以下规则放置水果：

每种水果必须放入第一个 容量大于等于 该水果数量的 最左侧可用篮子 中。
每个篮子只能装 一种 水果。
如果一种水果 无法放入 任何篮子，它将保持 未放置。
返回所有可能分配完成后，剩余未放置的水果种类的数量。



示例 1

输入： fruits = [4,2,5], baskets = [3,5,4]

输出： 1

解释：

fruits[0] = 4 放入 baskets[1] = 5。
fruits[1] = 2 放入 baskets[0] = 3。
fruits[2] = 5 无法放入 baskets[2] = 4。
由于有一种水果未放置，我们返回 1。

示例 2

输入： fruits = [3,6,1], baskets = [6,4,7]

输出： 0

解释：

fruits[0] = 3 放入 baskets[0] = 6。
fruits[1] = 6 无法放入 baskets[1] = 4（容量不足），但可以放入下一个可用的篮子 baskets[2] = 7。
fruits[2] = 1 放入 baskets[1] = 4。
由于所有水果都已成功放置，我们返回 0。
    """
    cangjie = "func solve(fruits: Array<Int64>, baskets: Array<Int64>): Int64 {\n"
    python = "def solve(self, fruits: List[int], baskets: List[int]) -> int:\n"
    des = __doc__
    degree = 0
    idx = "3477"

    def solve(self, fruits: List[int], baskets: List[int]) -> int:
        t = self.SegmentTree(baskets)
        n = len(baskets)
        ans = 0
        for x in fruits:
            if t.find_first_and_update(1, 0, n - 1, x) < 0:
                ans += 1
        return ans

    class SegmentTree:
        def __init__(self, a: List[int]):
            n = len(a)
            self.max = [0] * (2 << (n - 1).bit_length())
            self.build(a, 1, 0, n - 1)

        def maintain(self, o: int):
            self.max[o] = max(self.max[o * 2], self.max[o * 2 + 1])

        # 初始化线段树
        def build(self, a: List[int], o: int, l: int, r: int):
            if l == r:
                self.max[o] = a[l]
                return
            m = (l + r) // 2
            self.build(a, o * 2, l, m)
            self.build(a, o * 2 + 1, m + 1, r)
            self.maintain(o)

        # 找区间内的第一个 >= x 的数，并更新为 -1，返回这个数的下标（没有则返回 -1）
        def find_first_and_update(self, o: int, l: int, r: int, x: int) -> int:
            if self.max[o] < x:  # 区间没有 >= x 的数
                return -1
            if l == r:
                self.max[o] = -1  # 更新为 -1，表示不能放水果
                return l
            m = (l + r) // 2
            i = self.find_first_and_update(o * 2, l, m, x)  # 先递归左子树
            if i < 0:  # 左子树没找到
                i = self.find_first_and_update(o * 2 + 1, m + 1, r, x)  # 再递归右子树
            self.maintain(o)
            return i

    def gen(self):
        fruits = []
        baskets = []
        for _ in range(NUM):
            length = random.randint(1, 100)
            fruit = [random.randint(1, 1000) for _ in range(length)]
            basket = [random.randint(1, 1000) for _ in range(length)]

            fruits.append(fruit)
            baskets.append(basket)

        return fruits, baskets


class Solution4(BenchTesting):
    """
给你两个整数数组，nums1 和 nums2，长度均为 n，以及一个正整数 k 。

对从 0 到 n - 1 每个下标 i ，执行下述操作：

找出所有满足 nums1[j] 小于 nums1[i] 的下标 j 。
从这些下标对应的 nums2[j] 中选出 至多 k 个，并 最大化 这些值的总和作为结果。
返回一个长度为 n 的数组 answer ，其中 answer[i] 表示对应下标 i 的结果。



示例 1：

输入：nums1 = [4,2,1,5,3], nums2 = [10,20,30,40,50], k = 2

输出：[80,30,0,80,50]

解释：

对于 i = 0 ：满足 nums1[j] < nums1[0] 的下标为 [1, 2, 4] ，选出其中值最大的两个，结果为 50 + 30 = 80 。
对于 i = 1 ：满足 nums1[j] < nums1[1] 的下标为 [2] ，只能选择这个值，结果为 30 。
对于 i = 2 ：不存在满足 nums1[j] < nums1[2] 的下标，结果为 0 。
对于 i = 3 ：满足 nums1[j] < nums1[3] 的下标为 [0, 1, 2, 4] ，选出其中值最大的两个，结果为 50 + 30 = 80 。
对于 i = 4 ：满足 nums1[j] < nums1[4] 的下标为 [1, 2] ，选出其中值最大的两个，结果为 30 + 20 = 50 。
示例 2：

输入：nums1 = [2,2,2,2], nums2 = [3,1,2,3], k = 1

输出：[0,0,0,0]

解释：由于 nums1 中的所有元素相等，不存在满足条件 nums1[j] < nums1[i]，所有位置的结果都是 0 。
    """

    cangjie = "func solve(nums1: Array<Int64>, nums2: Array<Int64>, k: Int64): Array<Int64> {\n"
    python = "def solve(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:\n"
    des = __doc__
    degree = 1
    idx = "3478"

    def solve(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        from heapq import heappush, heappop
        a = sorted((x, y, i) for i, (x, y) in enumerate(zip(nums1, nums2)))
        n = len(a)
        ans = [0] * n
        h = []
        s = 0
        for i, (x, y, idx) in enumerate(a):
            ans[idx] = ans[a[i - 1][2]] if i and x == a[i - 1][0] else s
            s += y
            heappush(h, y)
            if len(h) > k:
                s -= heappop(h)
        return ans

    def gen(self):
        nums1 = []
        nums2 = []
        ks = []

        for _ in range(NUM):
            length = random.randint(1, 10)
            k = random.randint(1, length)
            ks.append(k)

            num1 = [random.randint(1, 100) for _ in range(length)]
            num2 = [random.randint(1, 100) for _ in range(length)]

            nums1.append(num1)
            nums2.append(num2)

        return nums1, nums2, ks


class Solution5(BenchTesting):
    """
给你两个长度为 n 的整数数组，fruits 和 baskets，其中 fruits[i] 表示第 i 种水果的 数量，baskets[j] 表示第 j 个篮子的 容量。

你需要对 fruits 数组从左到右按照以下规则放置水果：

每种水果必须放入第一个 容量大于等于 该水果数量的 最左侧可用篮子 中。
每个篮子只能装 一种 水果。
如果一种水果 无法放入 任何篮子，它将保持 未放置。
返回所有可能分配完成后，剩余未放置的水果种类的数量。



示例 1

输入： fruits = [4,2,5], baskets = [3,5,4]

输出： 1

解释：

fruits[0] = 4 放入 baskets[1] = 5。
fruits[1] = 2 放入 baskets[0] = 3。
fruits[2] = 5 无法放入 baskets[2] = 4。
由于有一种水果未放置，我们返回 1。

示例 2

输入： fruits = [3,6,1], baskets = [6,4,7]

输出： 0

解释：

fruits[0] = 3 放入 baskets[0] = 6。
fruits[1] = 6 无法放入 baskets[1] = 4（容量不足），但可以放入下一个可用的篮子 baskets[2] = 7。
fruits[2] = 1 放入 baskets[1] = 4。
由于所有水果都已成功放置，我们返回 0。
    """

    cangjie = "func solve(fruits: Array<Int64>, baskets: Array<Int64>): Int64 {\n"
    python = "def solve(self, fruits: List[int], baskets: List[int]) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3479"

    def solve(self, fruits: List[int], baskets: List[int]) -> int:
        t = self.SegmentTree(baskets)
        n = len(baskets)
        ans = 0
        for x in fruits:
            if t.find_first_and_update(1, 0, n - 1, x) < 0:
                ans += 1
        return ans

    class SegmentTree:
        def __init__(self, a: List[int]):
            n = len(a)
            self.max = [0] * (2 << (n - 1).bit_length())
            self.build(a, 1, 0, n - 1)

        def maintain(self, o: int):
            self.max[o] = max(self.max[o * 2], self.max[o * 2 + 1])

        # 初始化线段树
        def build(self, a: List[int], o: int, l: int, r: int):
            if l == r:
                self.max[o] = a[l]
                return
            m = (l + r) // 2
            self.build(a, o * 2, l, m)
            self.build(a, o * 2 + 1, m + 1, r)
            self.maintain(o)

        # 找区间内的第一个 >= x 的数，并更新为 -1，返回这个数的下标（没有则返回 -1）
        def find_first_and_update(self, o: int, l: int, r: int, x: int) -> int:
            if self.max[o] < x:  # 区间没有 >= x 的数
                return -1
            if l == r:
                self.max[o] = -1  # 更新为 -1，表示不能放水果
                return l
            m = (l + r) // 2
            i = self.find_first_and_update(o * 2, l, m, x)  # 先递归左子树
            if i < 0:  # 左子树没找到
                i = self.find_first_and_update(o * 2 + 1, m + 1, r, x)  # 再递归右子树
            self.maintain(o)
            return i

    def gen(self):
        fruits = []
        baskets = []
        for _ in range(NUM):
            length = random.randint(1, 1000)
            fruit = [random.randint(1, 10000) for _ in range(length)]
            basket = [random.randint(1, 10000) for _ in range(length)]

            fruits.append(fruit)
            baskets.append(basket)

        return fruits, baskets


class Solution6(BenchTesting):
    """
给你一个整数 n，表示一个包含从 1 到 n 按顺序排列的整数数组 nums。此外，给你一个二维数组 conflictingPairs，其中 conflictingPairs[i] = [a, b] 表示 a 和 b 形成一个冲突对。

从 conflictingPairs 中删除 恰好 一个元素。然后，计算数组 nums 中的非空子数组数量，这些子数组都不能同时包含任何剩余冲突对 [a, b] 中的 a 和 b。

返回删除 恰好 一个冲突对后可能得到的 最大 子数组数量。

子数组 是数组中一个连续的 非空 元素序列。



示例 1

输入： n = 4, conflictingPairs = [[2,3],[1,4]]

输出： 9

解释：

从 conflictingPairs 中删除 [2, 3]。现在，conflictingPairs = [[1, 4]]。
在 nums 中，存在 9 个子数组，其中 [1, 4] 不会一起出现。它们分别是 [1]，[2]，[3]，[4]，[1, 2]，[2, 3]，[3, 4]，[1, 2, 3] 和 [2, 3, 4]。
删除 conflictingPairs 中一个元素后，能够得到的最大子数组数量是 9。
示例 2

输入： n = 5, conflictingPairs = [[1,2],[2,5],[3,5]]

输出： 12

解释：

从 conflictingPairs 中删除 [1, 2]。现在，conflictingPairs = [[2, 5], [3, 5]]。
在 nums 中，存在 12 个子数组，其中 [2, 5] 和 [3, 5] 不会同时出现。
删除 conflictingPairs 中一个元素后，能够得到的最大子数组数量是 12。
    """

    cangjie = "func solve(n: Int64, conflictingPairs: Array<Array<Int64>>): Int64 {\n"
    python = "def solve(self, n: int, conflictingPairs: List[List[int]]) -> int:\n"
    des = __doc__
    degree = 2
    idx = "3480"

    def solve(self, n: int, conflictingPairs: List[List[int]]) -> int:
        groups = [[] for _ in range(n + 1)]
        for a, b in conflictingPairs:
            if a > b:
                a, b = b, a
            groups[a].append(b)

        ans = extra = max_extra = 0
        b0 = b1 = n + 1
        for a in range(n, 0, -1):
            pre_b0 = b0
            for b in groups[a]:
                if b < b0:
                    b0, b1 = b, b0
                elif b < b1:
                    b1 = b
            ans += b0 - a
            if b0 != pre_b0:
                extra = 0
            extra += b1 - b0
            max_extra = max(max_extra, extra)  # 这里改成手动 if 会快不少

        return ans + max_extra

    def gen(self):
        ns = []
        conflictingPairs = []

        for _ in range(NUM):
            n = random.randint(2, 50)
            ns.append(n)

            length = random.randint(1, n * 2)
            conflictingPair = []
            for _ in range(length):
                pair = sorted(random.sample(range(1, n + 1), 2))
                if pair not in conflictingPair:
                    conflictingPair.append(pair)

            conflictingPairs.append(conflictingPair)

        return ns, conflictingPairs


# 439 week
class Solution7(BenchTesting):
    """
给你一个整数数组 nums 和一个整数 k 。

如果整数 x 恰好仅出现在 nums 中的一个大小为 k 的子数组中，则认为 x 是 nums 中的几近缺失（almost missing）整数。

返回 nums 中 最大的几近缺失 整数，如果不存在这样的整数，返回 -1 。

子数组 是数组中的一个连续元素序列。


示例 1：

输入：nums = [3,9,2,1,7], k = 3

输出：7

解释：

1 出现在两个大小为 3 的子数组中：[9, 2, 1]、[2, 1, 7]
2 出现在三个大小为 3 的子数组中：[3, 9, 2]、[9, 2, 1]、[2, 1, 7]
3 出现在一个大小为 3 的子数组中：[3, 9, 2]
7 出现在一个大小为 3 的子数组中：[2, 1, 7]
9 出现在两个大小为 3 的子数组中：[3, 9, 2]、[9, 2, 1]
返回 7 ，因为它满足题意的所有整数中最大的那个。

示例 2：

输入：nums = [3,9,7,2,1,7], k = 4

输出：3

解释：

1 出现在两个大小为 3 的子数组中：[9, 7, 2, 1]、[7, 2, 1, 7]
2 出现在三个大小为 3 的子数组中：[3, 9, 7, 2]、[9, 7, 2, 1]、[7, 2, 1, 7]
3 出现在一个大小为 3 的子数组中：[3, 9, 7, 2]
7 出现在三个大小为 3 的子数组中：[3, 9, 7, 2]、[9, 7, 2, 1]、[7, 2, 1, 7]
9 出现在两个大小为 3 的子数组中：[3, 9, 7, 2]、[9, 7, 2, 1]
返回 3 ，因为它满足题意的所有整数中最大的那个。

示例 3：

输入：nums = [0,0], k = 1

输出：-1

解释：

不存在满足题意的整数。
    """
    cangjie = "func solve(nums: Array<Int64>, k: Int64): Int64 {\n"
    python = "def solve(self, nums: List[int], k: int) -> int:\n"
    des = __doc__
    degree = 0
    idx = "3471"
    uncheck = True

    def f(self, nums: List[int], x: int) -> int:
        return -1 if x in nums else x

    def solve(self, nums: List[int], k: int) -> int:
        if k == len(nums):
            return max(nums)
        if k == 1:
            ans = -1
            for x, c in Counter(nums).items():
                if c == 1:
                    ans = max(ans, x)
            return ans
        # nums[0] 不能出现在其他地方，nums[-1] 同理
        return max(self.f(nums[1:], nums[0]), self.f(nums[:-1], nums[-1]))

    def gen(self):
        nums = gen_lists_int(int_min=0, int_max=50, len_list_min=1, len_list_max=50)
        new_nums = []
        ks = []
        for num in nums:
            new_nums.append(num)
            length = len(num)
            k = random.randint(1, length)
            ks.append(k)
        return new_nums, ks


class Solution8(BenchTesting):
    """
给你一个字符串 s 和一个整数 k。

在一次操作中，你可以将任意位置的字符替换为字母表中相邻的字符（字母表是循环的，因此 'z' 的下一个字母是 'a'）。例如，将 'a' 替换为下一个字母结果是 'b'，将 'a' 替换为上一个字母结果是 'z'；同样，将 'z' 替换为下一个字母结果是 'a'，替换为上一个字母结果是 'y'。

返回在进行 最多 k 次操作后，s 的 最长回文子序列 的长度。

子序列 是一个 非空 字符串，可以通过删除原字符串中的某些字符（或不删除任何字符）并保持剩余字符的相对顺序得到。

回文 是正着读和反着读都相同的字符串。



示例 1：

输入: s = "abced", k = 2

输出: 3

解释:

将 s[1] 替换为下一个字母，得到 "acced"。
将 s[4] 替换为上一个字母，得到 "accec"。
子序列 "ccc" 形成一个长度为 3 的回文，这是最长的回文子序列。

示例 2：

输入: s = "aaazzz", k = 4

输出: 6

解释:

将 s[0] 替换为上一个字母，得到 "zaazzz"。
将 s[4] 替换为下一个字母，得到 "zaazaz"。
将 s[3] 替换为下一个字母，得到 "zaaaaz"。
整个字符串形成一个长度为 6 的回文。
    """
    cangjie = "func solve(s: String, k: Int64): Int64 {\n"
    python = "def solve(self, s: str, k: int) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3472"

    def solve(self, s: str, K: int) -> int:
        s = list(map(ord, s))  # 避免频繁计算 ord
        n = len(s)
        cnt = 0
        for i in range(n // 2):
            d = abs(s[i] - s[-1 - i])
            cnt += min(d, 26 - d)
        if cnt <= K:
            return n

        f = [[[0] * n for _ in range(n)] for _ in range(K + 1)]
        for k in range(K + 1):
            for i in range(n - 1, -1, -1):
                f[k][i][i] = 1
                for j in range(i + 1, n):
                    res = max(f[k][i + 1][j], f[k][i][j - 1])
                    d = abs(s[i] - s[j])
                    op = min(d, 26 - d)
                    if op <= k:
                        res = max(res, f[k - op][i + 1][j - 1] + 2)
                    f[k][i][j] = res
        return f[K][0][-1]

    def gen(self):
        import string
        chars = string.ascii_lowercase
        ks = []
        ss = []
        for _ in range(NUM):
            length = random.randint(1, 100)
            s = [random.choice(chars) for _ in range(length)]
            k = random.randint(1, 100)

            ss.append(s)
            ks.append(k)

        return ss, ks


class Solution9(BenchTesting):
    """
给你一个整数数组 nums 和两个整数 k 和 m。

返回数组 nums 中 k 个不重叠子数组的 最大 和，其中每个子数组的长度 至少 为 m。

子数组 是数组中的一个连续序列。



示例 1：

输入: nums = [1,2,-1,3,3,4], k = 2, m = 2

输出: 13

解释:

最优的选择是:

子数组 nums[3..5] 的和为 3 + 3 + 4 = 10（长度为 3 >= m）。
子数组 nums[0..1] 的和为 1 + 2 = 3（长度为 2 >= m）。
总和为 10 + 3 = 13。

示例 2：

输入: nums = [-10,3,-1,-2], k = 4, m = 1

输出: -10

解释:

最优的选择是将每个元素作为一个子数组。输出为 (-10) + 3 + (-1) + (-2) = -10。
    """

    cangjie = "func solve(nums: Array<Int64>, k: Int64, m: Int64): Int64 {\n"
    python = "def solve(self, nums: List[int], k: int, m: int) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3473"

    def solve(self, nums: List[int], k: int, m: int) -> int:
        from itertools import accumulate
        n = len(nums)
        s = list(accumulate(nums, initial=0))
        f = [0] * (n + 1)
        g = [0] * (n + 1)
        for i in range(1, k + 1):
            g[i * m - 1] = mx = -inf
            for j in range(i * m, n - (k - i) * m + 1):
                mx = max(mx, f[j - m] - s[j - m])
                g[j] = max(g[j - 1], mx + s[j])
            f, g = g, f
        return f[n]

    def gen(self):
        import math
        nums = gen_lists_int(int_min=-100, int_max=100)
        new_nums = []
        ms = []
        ks = []
        for num in nums:
            new_nums.append(num)

            m = random.randint(1, 3)
            ms.append(m)

            k_max = math.floor(len(num) / m)
            k = random.randint(1, k_max)
            ks.append(k)

        return new_nums, ks, ms


class Solution10(BenchTesting):
    """
给你两个字符串，str1 和 str2，其长度分别为 n 和 m 。

如果一个长度为 n + m - 1 的字符串 word 的每个下标 0 <= i <= n - 1 都满足以下条件，则称其由 str1 和 str2 生成：

如果 str1[i] == 'T'，则长度为 m 的 子字符串（从下标 i 开始）与 str2 相等，即 word[i..(i + m - 1)] == str2。
如果 str1[i] == 'F'，则长度为 m 的 子字符串（从下标 i 开始）与 str2 不相等，即 word[i..(i + m - 1)] != str2。
返回可以由 str1 和 str2 生成 的 字典序最小 的字符串。如果不存在满足条件的字符串，返回空字符串 ""。

如果字符串 a 在第一个不同字符的位置上比字符串 b 的对应字符在字母表中更靠前，则称字符串 a 的 字典序 小于 字符串 b。
如果前 min(a.length, b.length) 个字符都相同，则较短的字符串字典序更小。

子字符串 是字符串中的一个连续、非空 的字符序列。



示例 1：

输入: str1 = "TFTF", str2 = "ab"

输出: "ababa"

解释:

下表展示了字符串 "ababa" 的生成过程：
下标    T/F    长度为 m 的子字符串
0	    'T'	    "ab"
1	    'F'	    "ba"
2	    'T'	    "ab"
3	    'F'	    "ba"
字符串 "ababa" 和 "ababb" 都可以由 str1 和 str2 生成。

返回 "ababa"，因为它的字典序更小。

示例 2：

输入: str1 = "TFTF", str2 = "abc"

输出: ""

解释:

无法生成满足条件的字符串。

示例 3：

输入: str1 = "F", str2 = "d"

输出: "a"
    """

    cangjie = "func solve(str1: String, str2: String): String {\n"
    python = "def solve(self, str1: str, str2: str) -> str:\n"
    des = __doc__
    degree = 2
    idx = "3474"

    def calc_z(self, s: str) -> List[int]:
        n = len(s)
        z = [0] * n
        box_l, box_r = 0, 0  # z-box 左右边界（闭区间）
        for i in range(1, n):
            if i <= box_r:
                z[i] = min(z[i - box_l], box_r - i + 1)
            while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                box_l, box_r = i, i + z[i]
                z[i] += 1
        z[0] = n
        return z

    def solve(self, s: str, t: str) -> str:
        n, m = len(s), len(t)
        ans = ['?'] * (n + m - 1)

        # 处理 T
        z = self.calc_z(t)
        pre = -m
        for i, b in enumerate(s):
            if b != 'T':
                continue
            size = max(pre + m - i, 0)
            # t 的长为 size 的前后缀必须相同
            if size > 0 and z[m - size] < size:
                return ""
            # size 后的内容都是 '?'，填入 t
            ans[i + size: i + m] = t[size:]
            pre = i

        # 计算 <= i 的最近待定位置
        pre_q = [-1] * len(ans)
        pre = -1
        for i, c in enumerate(ans):
            if c == '?':
                ans[i] = 'a'  # 待定位置的初始值为 a
                pre = i
            pre_q[i] = pre

        # 找 ans 中的等于 t 的位置，可以用 KMP 或者 Z 函数
        z = self.calc_z(t + ''.join(ans))

        # 处理 F
        i = 0
        while i < n:
            if s[i] != 'F':
                i += 1
                continue
            # 子串必须不等于 t
            if z[m + i] < m:
                i += 1
                continue
            # 找最后一个待定位置
            j = pre_q[i + m - 1]
            if j < i:  # 没有
                return ""
            ans[j] = 'b'
            i = j + 1  # 直接跳过 j

        return ''.join(ans)

    def gen(self):
        import string
        chars = string.ascii_lowercase
        str1_list = []
        str2_list = []
        for _ in range(NUM):
            str1_length = random.randint(1, 20)
            binary_list = [random.randint(0, 1) for _ in range(str1_length)]
            str1 = ['F' if num == 0 else 'T' for num in binary_list]
            str1_list.append(''.join(str1))

            str2_length = random.randint(1, 10)
            str2 = [random.choice(chars) for _ in range(str2_length)]
            str2_list.append(''.join(str2))

        return str1_list, str2_list


# 434 week
class Solution11(BenchTesting):
    """
给你一个长度为 n 的整数数组 nums 。

分区 是指将数组按照下标 i （0 <= i < n - 1）划分成两个 非空 子数组，其中：

左子数组包含区间 [0, i] 内的所有下标。
右子数组包含区间 [i + 1, n - 1] 内的所有下标。
对左子数组和右子数组先求元素 和 再做 差 ，统计并返回差值为 偶数 的 分区 方案数。



示例 1：

输入：nums = [10,10,3,7,6]

输出：4

解释：

共有 4 个满足题意的分区方案：

[10]、[10, 3, 7, 6] 元素和的差值为 10 - 26 = -16 ，是偶数。
[10, 10]、[3, 7, 6] 元素和的差值为 20 - 16 = 4，是偶数。
[10, 10, 3]、[7, 6] 元素和的差值为 23 - 13 = 10，是偶数。
[10, 10, 3, 7]、[6] 元素和的差值为 30 - 6 = 24，是偶数。
示例 2：

输入：nums = [1,2,2]

输出：0

解释：

不存在元素和的差值为偶数的分区方案。

示例 3：

输入：nums = [2,4,6,8]

输出：3

解释：

所有分区方案都满足元素和的差值为偶数。
    """

    cangjie = "func solve(nums: Array<Int64>): Int64 {\n"
    python = "def solve(self, nums: List[int]) -> int:\n"
    des = __doc__
    degree = 0
    idx = "3432"

    def solve(self, nums: List[int]) -> int:
        return 0 if sum(nums) % 2 else len(nums) - 1

    def gen(self):
        nums = gen_lists_int(int_min=1, int_max=100)
        return [nums]


class Solution12(BenchTesting):
    """
给你一个长度为 n 的数组 nums ，同时给你一个整数 k 。

你可以对 nums 执行以下操作 一次 ：

选择一个子数组 nums[i..j] ，其中 0 <= i <= j <= n - 1 。
选择一个整数 x 并将 nums[i..j] 中 所有 元素都增加 x 。
请你返回执行以上操作以后数组中 k 出现的 最大 频率。

子数组 是一个数组中一段连续 非空 的元素序列。



示例 1：

输入：nums = [1,2,3,4,5,6], k = 1

输出：2

解释：

将 nums[2..5] 增加 -5 后，1 在数组 [1, 2, -2, -1, 0, 1] 中的频率为最大值 2 。

示例 2：

输入：nums = [10,2,3,4,5,5,4,3,2,2], k = 10

输出：4

解释：

将 nums[1..9] 增加 8 以后，10 在数组 [10, 10, 11, 12, 13, 13, 12, 11, 10, 10] 中的频率为最大值 4 。
    """

    cangjie = "func solve(nums: Array<Int64>, k: Int64): Int64 {\n"
    python = "def solve(self, nums: List[int], k: int) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3434"

    max = lambda a, b: a if a > b else b  # 手写 max 更快

    def solve(self, nums: List[int], k: int) -> int:
        f0 = max_f1 = f2 = 0
        f1 = [0] * 51  # 或者用 defaultdict(int)
        for x in nums:
            f2 = max(f2, max_f1) + (x == k)
            f1[x] = max(f1[x], f0) + 1
            f0 += (x == k)
            max_f1 = max(max_f1, f1[x])
        return max(max_f1, f2)

    def gen(self):
        nums = gen_lists_int(int_min=1, int_max=50)
        ks = gen_int(int_min=1, int_max=50)

        return nums, ks


class Solution13(BenchTesting):
    """
给你一个字符串数组 words 。请你找到 words 所有 最短公共超序列 ，且确保它们互相之间无法通过排列得到。

最短公共超序列 指的是一个字符串，words 中所有字符串都是它的子序列，且它的长度 最短 。

请你返回一个二维整数数组 freqs ，表示所有的最短公共超序列，其中 freqs[i] 是一个长度为 26 的数组，它依次表示一个最短公共超序列的所有小写英文字母的出现频率。你可以以任意顺序返回这个频率数组。

排列 指的是一个字符串中所有字母重新安排顺序以后得到的字符串。

一个 子序列 是从一个字符串中删除一些（也可以不删除）字符后，剩余字符不改变顺序连接得到的 非空 字符串。



示例 1：

输入：words = ["ab","ba"]

输出：[[1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

解释：

两个最短公共超序列分别是 "aba" 和 "bab" 。输出分别是两者的字母出现频率。

示例 2：

输入：words = ["aa","ac"]

输出：[[2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

解释：

两个最短公共超序列分别是 "aac" 和 "aca" 。由于它们互为排列，所以只保留 "aac" 。

示例 3：

输入：words = ["aa","bb","cc"]

输出：[[2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

解释：

"aabbcc" 和它所有的排列都是最短公共超序列。
    """

    cangjie = "func solve(words: Array<String>): Array<Array<Int64>> {\n"
    python = "def solve(self, words: List[str]) -> List[List[int]]:\n"
    des = __doc__
    degree = 2
    idx = "3435"

    def solve(self, words: List[str]) -> List[List[int]]:
        # 收集有哪些字母，同时建图
        all_mask = mask2 = 0
        g = defaultdict(list)
        for x, y in words:
            x, y = ord(x) - ord('a'), ord(y) - ord('a')
            all_mask |= 1 << x | 1 << y
            if x == y:
                mask2 |= 1 << x
            g[x].append(y)

        # 判断是否有环
        def has_cycle(sub: int) -> bool:
            color = [0] * 26

            def dfs(x: int) -> bool:
                color[x] = 1
                for y in g[x]:
                    # 只遍历在 sub 中的字母
                    if (sub >> y & 1) == 0:
                        continue
                    if color[y] == 1 or color[y] == 0 and dfs(y):
                        return True
                color[x] = 2
                return False

            for i, c in enumerate(color):
                # 只遍历在 sub 中的字母
                if c == 0 and sub >> i & 1 and dfs(i):
                    return True
            return False

        st = set()
        max_size = 0
        # 枚举 mask1 的所有子集 sub
        sub = mask1 = all_mask ^ mask2
        while True:
            size = sub.bit_count()
            # 剪枝：如果 size < max_size 就不需要判断了
            if size >= max_size and not has_cycle(sub):
                if size > max_size:
                    max_size = size
                    st.clear()
                st.add(sub)
            sub = (sub - 1) & mask1
            if sub == mask1:
                break

        return [[(all_mask >> i & 1) + ((all_mask ^ sub) >> i & 1) for i in range(26)]
                for sub in st]

    def gen(self):
        import string
        words_list = []

        for _ in range(NUM):
            words_length = random.randint(1, 10)
            letters = random.sample(string.ascii_lowercase, random.randint(1, 16))

            words = set()
            for _ in range(words_length):
                word = ''.join(random.choices(letters, k=2))
                words.add(word)

            words = list(words)
            words_list.append(words)

        return [words_list]


# 433 week
class Solution14(BenchTesting):
    """
给你一个长度为 n 的整数数组 nums 。对于 每个 下标 i（0 <= i < n），定义对应的子数组 nums[start ... i]（start = max(0, i - nums[i])）。

返回为数组中每个下标定义的子数组中所有元素的总和。

子数组 是数组中的一个连续、非空 的元素序列。


示例 1：

输入：nums = [2,3,1]

输出：11

解释：

下标 i	子数组	和
0	nums[0] = [2]	2
1	nums[0 ... 1] = [2, 3]	5
2	nums[1 ... 2] = [3, 1]	4
总和	 	11
总和为 11 。因此，输出 11 。

示例 2：

输入：nums = [3,1,1,2]

输出：13

解释：

下标 i	子数组	和
0	nums[0] = [3]	3
1	nums[0 ... 1] = [3, 1]	4
2	nums[1 ... 2] = [1, 1]	2
3	nums[1 ... 3] = [1, 1, 2]	4
总和	 	13
总和为 13 。因此，输出为 13 。
    """

    cangjie = "func solve(nums: Array<Int64>): Int64 {\n"
    python = "def solve(self, nums: List[int]) -> int:\n"
    des = __doc__
    degree = 0
    idx = "3427"

    def solve(self, nums: List[int]) -> int:
        diff = [0] * (len(nums) + 1)
        for i, num in enumerate(nums):
            diff[max(i - num, 0)] += 1
            diff[i + 1] -= 1

        ans = sd = 0
        for x, d in zip(nums, diff):
            sd += d
            ans += x * sd
        return ans

    def gen(self):
        nums = gen_lists_int(int_min=1, int_max=100)
        return [nums]


class Solution15(BenchTesting):
    """
给你一个整数数组 nums 和一个正整数 k，返回所有长度最多为 k 的 子序列 中 最大值 与 最小值 之和的总和。

非空子序列 是指从另一个数组中删除一些或不删除任何元素（且不改变剩余元素的顺序）得到的数组。

由于答案可能非常大，请返回对 109 + 7 取余数的结果。



示例 1：

输入： nums = [1,2,3], k = 2

输出： 24

解释：

数组 nums 中所有长度最多为 2 的子序列如下：

子序列	最小值	最大值	和
[1]	1	1	2
[2]	2	2	4
[3]	3	3	6
[1, 2]	1	2	3
[1, 3]	1	3	4
[2, 3]	2	3	5
总和	 	 	24
因此，输出为 24。

示例 2：

输入： nums = [5,0,6], k = 1

输出： 22

解释：

对于长度恰好为 1 的子序列，最小值和最大值均为元素本身。因此，总和为 5 + 5 + 0 + 0 + 6 + 6 = 22。

示例 3：

输入： nums = [1,1,1], k = 2

输出： 12

解释：

子序列 [1, 1] 和 [1] 各出现 3 次。对于所有这些子序列，最小值和最大值均为 1。因此，总和为 12。
    """

    cangjie = "func solve(nums: Array<Int64>, k: Int64): Int64 {\n"
    python = "def solve(self, nums: List[int], k: int) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3428"

    def solve(self, nums: List[int], k: int) -> int:
        import math
        MOD = 1_000_000_007
        nums.sort()
        ans = 0
        s = 1
        for i, x in enumerate(nums):
            ans += (x + nums[-1 - i]) * s
            s = (s * 2 - math.comb(i, k - 1)) % MOD
        return ans % MOD

    def gen(self):
        nums = gen_lists_int(int_min=0, int_max=10)
        new_nums = []
        ks = []
        for num in nums:
            new_nums.append(num)
            length = len(num)
            k = random.randint(1, min(100, length))
            ks.append(k)

        return new_nums, ks


class Solution16(BenchTesting):
    """
给你一个 偶数 整数 n，表示沿直线排列的房屋数量，以及一个大小为 n x 3 的二维数组 cost，其中 cost[i][j] 表示将第 i 个房屋涂成颜色 j + 1 的成本。

如果房屋满足以下条件，则认为它们看起来 漂亮：

不存在 两个 涂成相同颜色的相邻房屋。
距离行两端 等距 的房屋不能涂成相同的颜色。例如，如果 n = 6，则位置 (0, 5)、(1, 4) 和 (2, 3) 的房屋被认为是等距的。
返回使房屋看起来 漂亮 的 最低 涂色成本。



示例 1：

输入： n = 4, cost = [[3,5,7],[6,2,9],[4,8,1],[7,3,5]]

输出： 9

解释：

最佳涂色顺序为 [1, 2, 3, 2]，对应的成本为 [3, 2, 1, 3]。满足以下条件：

不存在涂成相同颜色的相邻房屋。
位置 0 和 3 的房屋（等距于两端）涂成不同的颜色 (1 != 2)。
位置 1 和 2 的房屋（等距于两端）涂成不同的颜色 (2 != 3)。
使房屋看起来漂亮的最低涂色成本为 3 + 2 + 1 + 3 = 9。



示例 2：

输入： n = 6, cost = [[2,4,6],[5,3,8],[7,1,9],[4,6,2],[3,5,7],[8,2,4]]

输出： 18

解释：

最佳涂色顺序为 [1, 3, 2, 3, 1, 2]，对应的成本为 [2, 8, 1, 2, 3, 2]。满足以下条件：

不存在涂成相同颜色的相邻房屋。
位置 0 和 5 的房屋（等距于两端）涂成不同的颜色 (1 != 2)。
位置 1 和 4 的房屋（等距于两端）涂成不同的颜色 (3 != 1)。
位置 2 和 3 的房屋（等距于两端）涂成不同的颜色 (2 != 3)。
使房屋看起来漂亮的最低涂色成本为 2 + 8 + 1 + 2 + 3 + 2 = 18。
    """

    cangjie = "func solve(n: Int64, cost: Array<Array<Int64>>): Int64 {\n"
    python = "def solve(self, n: int, cost: List[List[int]]) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3429"

    def solve(self, n: int, cost: List[List[int]]) -> int:
        f = [[[0] * 3 for _ in range(3)] for _ in range(n // 2 + 1)]
        for i, row in enumerate(cost[:n // 2]):
            row2 = cost[-1 - i]
            for pre_j in range(3):
                for pre_k in range(3):
                    res = inf
                    for j, c1 in enumerate(row):
                        if j == pre_j:
                            continue
                        for k, c2 in enumerate(row2):
                            if k != pre_k and k != j:
                                res = min(res, f[i][j][k] + c1 + c2)
                    f[i + 1][pre_j][pre_k] = res
        # 枚举所有初始颜色，取最小值
        return min(map(min, f[-1]))

    def gen(self):
        ns = []
        costs = []
        for _ in range(NUM):
            n = random.randrange(2, 50, 2)
            ns.append(n)

            cost = []
            for _ in range(n):
                row = [random.randint(0, 50) for _ in range(3)]
                cost.append(row)
            costs.append(cost)

        return ns, costs


class Solution17(BenchTesting):
    """
给你一个整数数组 nums 和一个 正 整数 k 。 返回 最多 有 k 个元素的所有子数组的 最大 和 最小 元素之和。

子数组 是数组中的一个连续、非空 的元素序列。

示例 1：

输入：nums = [1,2,3], k = 2

输出：20

解释：

最多 2 个元素的 nums 的子数组：

子数组	最小	最大	和
[1]	    1	1	2
[2]	    2	2	4
[3]	    3	3	6
[1, 2]	1	2	3
[2, 3]	2	3	5
总和	 	 	20
输出为 20 。

示例 2：

输入：nums = [1,-3,1], k = 2

输出：-6

解释：

最多 2 个元素的 nums 的子数组：

子数组	最小	最大	和
[1]	    1	1	2
[-3]	-3	-3	-6
[1]	    1	1	2
[1, -3]	-3	1	-2
[-3, 1]	-3	1	-2
总和	 	 	-6
输出为 -6 。
    """

    cangjie = "func solve(nums: Array<Int64>, k: Int64): Int64 {\n"
    python = "def solve(self, nums: List[int], k: int) -> int:\n"
    des = __doc__
    degree = 2
    idx = "3430"

    def sumSubarrayMins(self, nums: List[int], k: int) -> int:
        count = lambda m: (m * 2 - k + 1) * k // 2 if m > k else (m + 1) * m // 2
        ans = 0
        st = [-1]
        for r, x in enumerate(nums + [-inf]):
            while len(st) > 1 and nums[st[-1]] >= x:
                i = st.pop()
                l = st[-1]
                cnt = count(r - l - 1) - count(i - l - 1) - count(r - i - 1)
                ans += nums[i] * cnt
            st.append(r)
        return ans

    def solve(self, nums: List[int], k: int) -> int:
        ans = self.sumSubarrayMins(nums, k)
        ans -= self.sumSubarrayMins([-x for x in nums], k)
        return ans

    def gen(self):
        nums = gen_lists_int(int_min=-10, int_max=10)
        new_nums = []
        ks = []
        for num in nums:
            new_nums.append(num)
            length = len(num)
            k = random.randint(1, length)
            ks.append(k)

        return new_nums, ks


# 432 week
class Solution18(BenchTesting):
    """
给你一个 m x n 的二维数组 grid，数组由 正整数 组成。

你的任务是以 之字形 遍历 grid，同时跳过每个 交替 的单元格。

之字形遍历的定义如下：

从左上角的单元格 (0, 0) 开始。
在当前行中向 右 移动，直到到达该行的末尾。
下移到下一行，然后在该行中向 左 移动，直到到达该行的开头。
继续在行间交替向右和向左移动，直到所有行都被遍历完。
注意：在遍历过程中，必须跳过每个 交替 的单元格。

返回一个整数数组 result，其中包含按 顺序 记录的、且跳过交替单元格后的之字形遍历中访问到的单元格值。



示例 1：

输入： grid = [[1,2],[3,4]]

输出： [1,4]


示例 2：

输入： grid = [[2,1],[2,1],[2,1]]

输出： [2,1,2]


示例 3：

输入： grid = [[1,2,3],[4,5,6],[7,8,9]]

输出： [1,3,5,7,9]

    """

    cangjie = "func solve(grid: Array<Array<Int64>>): Array<Int64> {\n"
    python = "def solve(self, grid: List[List[int]]) -> List[int]:\n"
    des = __doc__
    degree = 0
    idx = "3417"

    def solve(self, grid: List[List[int]]) -> List[int]:
        end = -1 - len(grid[0]) % 2
        ans = []
        for i, row in enumerate(grid):
            ans.extend(row[end::-2] if i % 2 else row[::2])
        return ans

    def gen(self):
        grids = []
        for _ in range(NUM):
            n = random.randint(2, 10)
            m = random.randint(2, 10)
            grid = [[random.randint(1, 20) for _ in range(m)] for _ in range(n)]
            grids.append(grid)

        return [grids]


class Solution19(BenchTesting):
    """
给你一个 m x n 的网格。一个机器人从网格的左上角 (0, 0) 出发，目标是到达网格的右下角 (m - 1, n - 1)。在任意时刻，机器人只能向右或向下移动。

网格中的每个单元格包含一个值 coins[i][j]：

如果 coins[i][j] >= 0，机器人可以获得该单元格的金币。
如果 coins[i][j] < 0，机器人会遇到一个强盗，强盗会抢走该单元格数值的 绝对值 的金币。
机器人有一项特殊能力，可以在行程中 最多感化 2个单元格的强盗，从而防止这些单元格的金币被抢走。

注意：机器人的总金币数可以是负数。

返回机器人在路径上可以获得的 最大金币数 。



示例 1：

输入： coins = [[0,1,-1],[1,-2,3],[2,-3,4]]

输出： 8

解释：

一个获得最多金币的最优路径如下：

从 (0, 0) 出发，初始金币为 0（总金币 = 0）。
移动到 (0, 1)，获得 1 枚金币（总金币 = 0 + 1 = 1）。
移动到 (1, 1)，遇到强盗抢走 2 枚金币。机器人在此处使用一次感化能力，避免被抢（总金币 = 1）。
移动到 (1, 2)，获得 3 枚金币（总金币 = 1 + 3 = 4）。
移动到 (2, 2)，获得 4 枚金币（总金币 = 4 + 4 = 8）。
示例 2：

输入： coins = [[10,10,10],[10,10,10]]

输出： 40

解释：

一个获得最多金币的最优路径如下：

从 (0, 0) 出发，初始金币为 10（总金币 = 10）。
移动到 (0, 1)，获得 10 枚金币（总金币 = 10 + 10 = 20）。
移动到 (0, 2)，再获得 10 枚金币（总金币 = 20 + 10 = 30）。
移动到 (1, 2)，获得 10 枚金币（总金币 = 30 + 10 = 40）。
    """

    cangjie = "func solve(coins: Array<Array<Int64>>): Int64 {\n"
    python = "def solve(self, coins: List[List[int]]) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3418"

    def solve(self, coins: List[List[int]]) -> int:
        n = len(coins[0])
        f = [[-inf] * 3 for _ in range(n + 1)]
        f[1] = [0] * 3
        for row in coins:
            for j, x in enumerate(row):
                f[j + 1][2] = max(f[j][2] + x, f[j + 1][2] + x, f[j][1], f[j + 1][1])
                f[j + 1][1] = max(f[j][1] + x, f[j + 1][1] + x, f[j][0], f[j + 1][0])
                f[j + 1][0] = max(f[j][0], f[j + 1][0]) + x
        return f[n][2]

    def gen(self):
        coins_list = []
        for _ in range(NUM):
            m = random.randint(1, 50)
            n = random.randint(1, 50)
            coins = []
            for _ in range(m):
                row = [random.randint(-100, 100) for _ in range(n)]
                coins.append(row)
            coins_list.append(coins)

        return [coins_list]


class Solution20(BenchTesting):
    """
给你一个长度为 n 的数组 nums 和一个整数 k 。

对于 nums 中的每一个子数组，你可以对它进行 至多 k 次操作。每次操作中，你可以将子数组中的任意一个元素增加 1 。

注意 ，每个子数组都是独立的，也就是说你对一个子数组的修改不会保留到另一个子数组中。

请你返回最多 k 次操作以内，有多少个子数组可以变成 非递减 的。

如果一个数组中的每一个元素都大于等于前一个元素（如果前一个元素存在），那么我们称这个数组是 非递减 的。



示例 1：

输入：nums = [6,3,1,2,4,4], k = 7

输出：17

解释：

nums 的所有 21 个子数组中，只有子数组 [6, 3, 1] ，[6, 3, 1, 2] ，[6, 3, 1, 2, 4] 和 [6, 3, 1, 2, 4, 4] 无法在 k = 7 次操作以内变为非递减的。所以非递减子数组的数目为 21 - 4 = 17 。

示例 2：

输入：nums = [6,3,1,3,6], k = 4

输出：12

解释：

子数组 [3, 1, 3, 6] 和 nums 中所有小于等于三个元素的子数组中，除了 [6, 3, 1] 以外，都可以在 k 次操作以内变为非递减子数组。总共有 5 个包含单个元素的子数组，4 个包含两个元素的子数组，除 [6, 3, 1] 以外有 2 个包含三个元素的子数组，所以总共有 1 + 5 + 4 + 2 = 12 个子数组可以变为非递减的。
    """

    cangjie = "func solve(nums: Array<Int64>, k: Int64): Int64 {\n"
    python = "def solve(self, nums: List[int], k: int) -> int:\n"
    des = __doc__
    degree = 2
    idx = "3420"

    def solve(self, nums: List[int], k: int) -> int:
        from collections import deque
        n = len(nums)
        ans = cnt = 0
        q = deque()  # (根节点的值, 树的大小)
        r = n - 1
        for l in range(n - 1, -1, -1):
            # x 进入窗口
            x = nums[l]
            size = 1  # 统计以 x 为根的树的大小
            while q and x >= q[-1][0]:
                # 以 v 为根的树，现在合并到 x 的下面（x 和 v 连一条边）
                v, sz = q.pop()
                size += sz
                cnt += (x - v) * sz  # 树 v 中的数都变成 x
            q.append([x, size])

            # 操作次数太多，缩小窗口
            while cnt > k:
                # 操作次数的减少量，等于 nums[r] 所在树的根节点值减去 nums[r]
                tree = q[0]  # 最右边的树
                cnt -= tree[0] - nums[r]
                r -= 1
                # nums[r] 离开窗口后，树的大小减一
                tree[1] -= 1
                if tree[1] == 0:  # 这棵树是空的
                    q.popleft()

            ans += r - l + 1

        return ans

    def gen(self):
        nums = gen_lists_int(int_min=1)
        ks = gen_int(int_min=1)
        new_nums = []
        new_ks = []
        for num, k in zip(nums, ks):
            new_nums.append(num)
            new_ks.append(k)

        return new_nums, new_ks