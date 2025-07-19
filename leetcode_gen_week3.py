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

# 431-426week
# 431 week
class Solution1(BenchTesting):
    """
给你一个由 正整数 组成的数组 nums。

如果一个数组 arr 满足 prod(arr) == lcm(arr) * gcd(arr)，则称其为 乘积等价数组 ，其中：

prod(arr) 表示 arr 中所有元素的乘积。
gcd(arr) 表示 arr 中所有元素的最大公因数 (GCD)。
lcm(arr) 表示 arr 中所有元素的最小公倍数 (LCM)。
返回数组 nums 的 最长 乘积等价 子数组 的长度。



示例 1：

输入： nums = [1,2,1,2,1,1,1]

输出： 5

解释：

最长的乘积等价子数组是 [1, 2, 1, 1, 1]，其中 prod([1, 2, 1, 1, 1]) = 2， gcd([1, 2, 1, 1, 1]) = 1，以及 lcm([1, 2, 1, 1, 1]) = 2。

示例 2：

输入： nums = [2,3,4,5,6]

输出： 3

解释：

最长的乘积等价子数组是 [3, 4, 5]。

示例 3：

输入： nums = [1,2,3,1,4,5,1]

输出： 5
    """

    cangjie = "func solve(nums: Array<Int64>): Int64 {\n"
    python = "def solve(self, nums: List[int]) -> int:\n"
    des = __doc__
    degree = 0
    idx = "3411"

    def solve(self, nums: List[int]) -> int:
        import math
        ans = 2
        mul = 1
        left = 0
        for right, x in enumerate(nums):
            while math.gcd(mul, x) > 1:
                mul //= nums[left]
                left += 1
            mul *= x
            ans = max(ans, right - left + 1)
        return ans

    def gen(self):
        nums = gen_lists_int(int_min=1, int_max=10)
        return [nums]


class Solution2(BenchTesting):
    """
给你一个字符串 s。

英文字母中每个字母的 镜像 定义为反转字母表之后对应位置上的字母。例如，'a' 的镜像是 'z'，'y' 的镜像是 'b'。

最初，字符串 s 中的所有字符都 未标记 。

字符串 s 的初始分数为 0 ，你需要对其执行以下过程：

从左到右遍历字符串。
对于每个下标 i ，找到距离最近的 未标记 下标 j，下标 j 需要满足 j < i 且 s[j] 是 s[i] 的镜像。然后 标记 下标 i 和 j，总分加上 i - j 的值。
如果对于下标 i，不存在满足条件的下标 j，则跳过该下标，继续处理下一个下标，不需要进行标记。
返回最终的总分。



示例 1：

输入： s = "aczzx"

输出： 5

解释：

i = 0。没有符合条件的下标 j，跳过。
i = 1。没有符合条件的下标 j，跳过。
i = 2。距离最近的符合条件的下标是 j = 0，因此标记下标 0 和 2，然后将总分加上 2 - 0 = 2 。
i = 3。没有符合条件的下标 j，跳过。
i = 4。距离最近的符合条件的下标是 j = 1，因此标记下标 1 和 4，然后将总分加上 4 - 1 = 3 。
示例 2：

输入： s = "abcdef"

输出： 0

解释：

对于每个下标 i，都不存在满足条件的下标 j。
    """

    cangjie = "func solve(s: String): Int64 {\n"
    python = "def solve(self, s: str) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3412"

    def solve(self, s: str) -> int:
        stk = [[] for _ in range(26)]
        ans = 0
        for i, c in enumerate(map(ord, s)):
            c -= ord('a')
            if stk[25 - c]:
                ans += i - stk[25 - c].pop()
            else:
                stk[c].append(i)
        return ans

    def gen(self):
        ss = []

        for _ in range(NUM):
            import string
            chars = string.ascii_lowercase
            length = random.randint(1, 100)
            s = [random.choice(chars) for _ in range(length)]

            ss.append(''.join(s))

        return [ss]


class Solution3(BenchTesting):
    """
在一条数轴上有无限多个袋子，每个坐标对应一个袋子。其中一些袋子里装有硬币。

给你一个二维数组 coins，其中 coins[i] = [li, ri, ci] 表示从坐标 li 到 ri 的每个袋子中都有 ci 枚硬币。

数组 coins 中的区间互不重叠。

另给你一个整数 k。

返回通过收集连续 k 个袋子可以获得的 最多 硬币数量。



示例 1：

输入： coins = [[8,10,1],[1,3,2],[5,6,4]], k = 4

输出： 10

解释：

选择坐标为 [3, 4, 5, 6] 的袋子可以获得最多硬币：2 + 0 + 4 + 4 = 10。

示例 2：

输入： coins = [[1,10,3]], k = 2

输出： 6

解释：

选择坐标为 [1, 2] 的袋子可以获得最多硬币：3 + 3 = 6。
    """
    cangjie = "func solve(coins: Array<Array<Int64>>, k: Int64): Int64 {\n"
    python = "def solve(self, coins: List[List[int]], k: int) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3413"

    def maximumWhiteTiles(self, tiles: List[List[int]], carpetLen: int) -> int:
        ans = cover = left = 0
        for tl, tr, c in tiles:
            cover += (tr - tl + 1) * c
            while tiles[left][1] < tr - carpetLen + 1:
                cover -= (tiles[left][1] - tiles[left][0] + 1) * tiles[left][2]
                left += 1
            uncover = max((tr - carpetLen + 1 - tiles[left][0]) * tiles[left][2], 0)
            ans = max(ans, cover - uncover)
        return ans

    def solve(self, coins: List[List[int]], k: int) -> int:
        coins.sort(key=lambda c: c[0])
        ans = self.maximumWhiteTiles(coins, k)

        coins.reverse()
        for t in coins:
            t[0], t[1] = -t[1], -t[0]
        return max(ans, self.maximumWhiteTiles(coins, k))

    def gen(self):
        coins_list = []
        ks = []

        for _ in range(NUM):
            length = random.randint(1, 100)
            k = random.randint(1, 100)
            ks.append(k)

            coins = []
            for _ in range(length):
                l = random.randint(1, 99)
                r = random.randint(l, 100)
                c = random.randint(1, 100)
                coins.append([l, r, c])
            coins_list.append(coins)

        return coins_list, ks


class Solution4(BenchTesting):
    """
给你一个二维整数数组 intervals，其中 intervals[i] = [li, ri, weighti]。区间 i 的起点为 li，终点为 ri，权重为 weighti。你最多可以选择 4 个互不重叠 的区间。所选择区间的 得分 定义为这些区间权重的总和。

返回一个至多包含 4 个下标且 字典序最小 的数组，表示从 intervals 中选中的互不重叠且得分最大的区间。

如果两个区间没有任何重叠点，则称二者 互不重叠 。特别地，如果两个区间共享左边界或右边界，也认为二者重叠。



示例 1：

输入： intervals = [[1,3,2],[4,5,2],[1,5,5],[6,9,3],[6,7,1],[8,9,1]]

输出： [2,3]

解释：

可以选择下标为 2 和 3 的区间，其权重分别为 5 和 3。

示例 2：

输入： intervals = [[5,8,1],[6,7,7],[4,7,3],[9,10,6],[7,8,2],[11,14,3],[3,5,5]]

输出： [1,3,5,6]

解释：

可以选择下标为 1、3、5 和 6 的区间，其权重分别为 7、6、3 和 5。
    """

    cangjie = "func solve(intervals: ArrayList<ArrayList<Int64>>): Array<Int64> {\n"
    python = "def solve(self, intervals: List[List[int]]) -> List[int]:\n"
    des = __doc__
    degree = 2
    idx = "3414"

    def solve(self, intervals: List[List[int]]) -> List[int]:
        a = [(r, l, weight, i) for i, (l, r, weight) in enumerate(intervals)]
        a.sort(key=lambda t: t[0])  # 按照右端点排序
        f = [[(0, []) for _ in range(5)] for _ in range(len(intervals) + 1)]
        for i, (r, l, weight, idx) in enumerate(a):
            k = bisect_left(a, (l,), hi=i)  # hi=i 表示二分上界为 i（默认为 n）
            for j in range(1, 5):
                # 为什么是 f[k] 不是 f[k+1]：上面算的是 >= l，-1 后得到 < l，但由于还要 +1，抵消了
                s2, id2 = f[k][j - 1]
                # 注意这里是减去 weight，这样取 min 后相当于计算的是最大和
                f[i + 1][j] = min(f[i][j], (s2 - weight, sorted(id2 + [idx])))
        return f[-1][4][1]

    def gen(self):
        intervals_list = []
        for _ in range(NUM):
            length = random.randint(1, 100)

            intervals = []
            for _ in range(length):
                l = random.randint(1, 99)
                r = random.randint(l, 100)
                weight = random.randint(1, 100)
                intervals.append([l, r, weight])
            intervals_list.append(intervals)
        return [intervals_list]


# 430 week
class Solution5(BenchTesting):
    """
给你一个由 非负 整数组成的 m x n 矩阵 grid。

在一次操作中，你可以将任意元素 grid[i][j] 的值增加 1。

返回使 grid 的所有列 严格递增 所需的 最少 操作次数。



示例 1：

输入: grid = [[3,2],[1,3],[3,4],[0,1]]

输出: 15

解释:

为了让第 0 列严格递增，可以对 grid[1][0] 执行 3 次操作，对 grid[2][0] 执行 2 次操作，对 grid[3][0] 执行 6 次操作。
为了让第 1 列严格递增，可以对 grid[3][1] 执行 4 次操作。

示例 2：

输入: grid = [[3,2,1],[2,1,0],[1,2,3]]

输出: 12

解释:

为了让第 0 列严格递增，可以对 grid[1][0] 执行 2 次操作，对 grid[2][0] 执行 4 次操作。
为了让第 1 列严格递增，可以对 grid[1][1] 执行 2 次操作，对 grid[2][1] 执行 2 次操作。
为了让第 2 列严格递增，可以对 grid[1][2] 执行 2 次操作。

    """

    cangjie = "func solve(grid: Array<Array<Int64>>): Int64 {\n"
    python = "def solve(self, grid: List[List[int]]) -> int:\n"
    des = __doc__
    degree = 0
    idx = "3402"

    def solve(self, grid: List[List[int]]) -> int:
        ans = 0
        for col in zip(*grid):
            pre = -inf
            for x in col:
                ans += max(pre + 1 - x, 0)
                pre = max(pre + 1, x)
        return ans

    def gen(self):
        grid_list = []
        for _ in range(NUM):
            m = random.randint(1, 50)
            n = random.randint(1, 50)

            grid = []
            for _ in range(m):
                grid.append([random.randint(0, 100) for _ in range(n)])

            grid_list.append(grid)

        return [grid_list]


class Solution6(BenchTesting):
    """
给你一个字符串 word 和一个整数 numFriends。

Alice 正在为她的 numFriends 位朋友组织一个游戏。游戏分为多个回合，在每一回合中：

word 被分割成 numFriends 个 非空 字符串，且该分割方式与之前的任意回合所采用的都 不完全相同 。
所有分割出的字符串都会被放入一个盒子中。
在所有回合结束后，找出盒子中 字典序最大的 字符串。



示例 1：

输入: word = "dbca", numFriends = 2

输出: "dbc"

解释:

所有可能的分割方式为：

"d" 和 "bca"。
"db" 和 "ca"。
"dbc" 和 "a"。
示例 2：

输入: word = "gggg", numFriends = 4

输出: "g"

解释:

唯一可能的分割方式为："g", "g", "g", 和 "g"。
    """

    cangjie = "func solve(word: String, numFriends: Int64): String {\n"
    python = "def solve(self, word: str, numFriends: int) -> str:\n"
    des = __doc__
    degree = 1
    idx = "3403"

    def solve(self, s: str, numFriends: int) -> str:
        if numFriends == 1:
            return s
        n = len(s)
        i, j = 0, 1
        while j < n:
            k = 0
            while j + k < n and s[i + k] == s[j + k]:
                k += 1
            if j + k < n and s[i + k] < s[j + k]:
                i, j = j, max(j + 1, i + k + 1)
            else:
                j += k + 1
        return s[i: i + n - numFriends + 1]

    def gen(self):
        word_list = []
        numFriends_list = []

        for _ in range(NUM):
            import string
            chars = string.ascii_lowercase
            length = random.randint(1, 100)

            word = [random.choice(chars) for _ in range(length)]
            word_list.append(''.join(word))

            numFriends = random.randint(1, length)
            numFriends_list.append(numFriends)

        return word_list, numFriends_list


class Solution7(BenchTesting):
    """
给你一个只包含正整数的数组 nums 。

特殊子序列 是一个长度为 4 的子序列，用下标 (p, q, r, s) 表示，它们满足 p < q < r < s ，且这个子序列 必须 满足以下条件：

nums[p] * nums[r] == nums[q] * nums[s]
相邻坐标之间至少间隔 一个 数字。换句话说，q - p > 1 ，r - q > 1 且 s - r > 1 。
子序列指的是从原数组中删除零个或者更多元素后，剩下元素不改变顺序组成的数字序列。

请你返回 nums 中不同 特殊子序列 的数目。



示例 1：

输入：nums = [1,2,3,4,3,6,1]

输出：1

解释：

nums 中只有一个特殊子序列。

(p, q, r, s) = (0, 2, 4, 6) ：
对应的元素为 (1, 3, 3, 1) 。
nums[p] * nums[r] = nums[0] * nums[4] = 1 * 3 = 3
nums[q] * nums[s] = nums[2] * nums[6] = 3 * 1 = 3
示例 2：

输入：nums = [3,4,3,4,3,4,3,4]

输出：3

解释：

nums 中共有三个特殊子序列。

(p, q, r, s) = (0, 2, 4, 6) ：
对应元素为 (3, 3, 3, 3) 。
nums[p] * nums[r] = nums[0] * nums[4] = 3 * 3 = 9
nums[q] * nums[s] = nums[2] * nums[6] = 3 * 3 = 9
(p, q, r, s) = (1, 3, 5, 7) ：
对应元素为 (4, 4, 4, 4) 。
nums[p] * nums[r] = nums[1] * nums[5] = 4 * 4 = 16
nums[q] * nums[s] = nums[3] * nums[7] = 4 * 4 = 16
(p, q, r, s) = (0, 2, 5, 7) ：
对应元素为 (3, 3, 4, 4) 。
nums[p] * nums[r] = nums[0] * nums[5] = 3 * 4 = 12
nums[q] * nums[s] = nums[2] * nums[7] = 3 * 4 = 12
    """
    cangjie = "func solve(nums: Array<Int64>): Int64 {\n"
    python = "def solve(self, nums: List[int]) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3404"

    def solve(self, nums: List[int]) -> int:
        ans = 0
        cnt = defaultdict(int)
        # 枚举 b 和 c
        for i in range(4, len(nums) - 2):
            # 增量式更新，本轮循环只需枚举 b=nums[i-2] 这一个数
            # 至于更前面的 b，已经在前面的循环中添加到 cnt 中了，不能重复添加
            b = nums[i - 2]
            # 枚举 a
            for a in nums[:i - 3]:
                cnt[a / b] += 1

            c = nums[i]
            # 枚举 d
            for d in nums[i + 2:]:
                ans += cnt[d / c]
        return ans

    def gen(self):
        nums = []
        for _ in range(NUM):
            length = random.randint(7, 200)
            num = [random.randint(1, 200) for _ in range(length)]
            nums.append(num)

        return [nums]


class Solution8(BenchTesting):
    """
给你三个整数 n ，m ，k 。长度为 n 的 好数组 arr 定义如下：

arr 中每个元素都在 闭 区间 [1, m] 中。
恰好 有 k 个下标 i （其中 1 <= i < n）满足 arr[i - 1] == arr[i] 。
请你返回可以构造出的 好数组 数目。

由于答案可能会很大，请你将它对 109 + 7 取余 后返回。



示例 1：

输入：n = 3, m = 2, k = 1

输出：4

解释：

总共有 4 个好数组，分别是 [1, 1, 2] ，[1, 2, 2] ，[2, 1, 1] 和 [2, 2, 1] 。
所以答案为 4 。
示例 2：

输入：n = 4, m = 2, k = 2

输出：6

解释：

好数组包括 [1, 1, 1, 2] ，[1, 1, 2, 2] ，[1, 2, 2, 2] ，[2, 1, 1, 1] ，[2, 2, 1, 1] 和 [2, 2, 2, 1] 。
所以答案为 6 。
示例 3：

输入：n = 5, m = 2, k = 0

输出：2

解释：

好数组包括 [1, 2, 1, 2, 1] 和 [2, 1, 2, 1, 2] 。
所以答案为 2 。
    """
    cangjie = "func solve(n: Int64, m: Int64, k: Int64): Int64 {\n"
    python = "def solve(self, n: int, m: int, k: int) -> int:\n"
    des = __doc__
    degree = 2
    idx = "3405"

    def solve(self, n: int, m: int, k: int) -> int:
        import math
        MOD = 1_000_000_007
        return math.comb(n - 1, k) % MOD * m * pow(m - 1, n - k - 1, MOD) % MOD

    def gen(self):
        ns = []
        ms = []
        ks = []

        for _ in range(NUM):
            n = random.randint(1, 10)
            m = random.randint(1, 10)
            k = random.randint(0, n - 1)

            ns.append(n)
            ms.append(m)
            ks.append(k)

        return ns, ms, ks


# 429 week
class Solution9(BenchTesting):
    """
给你一个整数数组 nums，你需要确保数组中的元素 互不相同 。为此，你可以执行以下操作任意次：

从数组的开头移除 3 个元素。如果数组中元素少于 3 个，则移除所有剩余元素。
注意：空数组也视作为数组元素互不相同。返回使数组元素互不相同所需的 最少操作次数 。

示例 1：

输入： nums = [1,2,3,4,2,3,3,5,7]

输出： 2

解释：

第一次操作：移除前 3 个元素，数组变为 [4, 2, 3, 3, 5, 7]。
第二次操作：再次移除前 3 个元素，数组变为 [3, 5, 7]，此时数组中的元素互不相同。
因此，答案是 2。

示例 2：

输入： nums = [4,5,6,4,4]

输出： 2

解释：

第一次操作：移除前 3 个元素，数组变为 [4, 4]。
第二次操作：移除所有剩余元素，数组变为空。
因此，答案是 2。

示例 3：

输入： nums = [6,7,8,9]

输出： 0

解释：

数组中的元素已经互不相同，因此不需要进行任何操作，答案是 0。
    """

    cangjie = "func solve(nums: Array<Int64>): Int64 {\n"
    python = "def solve(self, nums: List[int]) -> int:\n"
    des = __doc__
    degree = 0
    idx = "3396"

    def solve(self, nums: List[int]) -> int:
        seen = set()
        for i in range(len(nums) - 1, -1, -1):
            x = nums[i]
            if x in seen:
                return i // 3 + 1
            seen.add(x)
        return 0

    def gen(self):
        nums = gen_lists_int(int_min=1, int_max=100)

        return [nums]


class Solution10(BenchTesting):
    """
给你一个整数数组 nums 和一个整数 k。

你可以对数组中的每个元素 最多 执行 一次 以下操作：

将一个在范围 [-k, k] 内的整数加到该元素上。
返回执行这些操作后，nums 中可能拥有的不同元素的 最大 数量。



示例 1：

输入： nums = [1,2,2,3,3,4], k = 2

输出： 6

解释：

对前四个元素执行操作，nums 变为 [-1, 0, 1, 2, 3, 4]，可以获得 6 个不同的元素。

示例 2：

输入： nums = [4,4,4,4], k = 1

输出： 3

解释：

对 nums[0] 加 -1，以及对 nums[1] 加 1，nums 变为 [3, 5, 4, 4]，可以获得 3 个不同的元素。
    """

    cangjie = "func solve(nums: Array<Int64>, k: Int64): Int64 {\n"
    python = "def solve(self, nums: List[int], k: int) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3397"

    def solve(self, nums: List[int], k: int) -> int:
        if k * 2 + 1 >= len(nums):
            return len(nums)

        nums.sort()
        ans = 0
        pre = -inf  # 记录每个人左边的人的位置
        for x in nums:
            x = min(max(x - k, pre + 1), x + k)
            if x > pre:
                ans += 1
                pre = x
        return ans

    def gen(self):
        nums = gen_lists_int(int_min=1, int_max=10)
        ks = gen_int(int_min=0, int_max=10)

        return nums, ks


class Solution11(BenchTesting):
    """
给你一个长度为 n 的二进制字符串 s 和一个整数 numOps。

你可以对 s 执行以下操作，最多 numOps 次：

选择任意下标 i（其中 0 <= i < n），并 翻转 s[i]，即如果 s[i] == '1'，则将 s[i] 改为 '0'，反之亦然。
你需要 最小化 s 的最长 相同 子字符串 的长度，相同子字符串 是指子字符串中的所有字符都 相同。

返回执行所有操作后可获得的 最小 长度。



示例 1：

输入: s = "000001", numOps = 1

输出: 2

解释:

将 s[2] 改为 '1'，s 变为 "001001"。最长的所有字符相同的子串为 s[0..1] 和 s[3..4]。

示例 2：

输入: s = "0000", numOps = 2

输出: 1

解释:

将 s[0] 和 s[2] 改为 '1'，s 变为 "1010"。

示例 3：

输入: s = "0101", numOps = 0

输出: 1
    """

    cangjie = "func solve(s: String, numOps: Int64): Int64 {\n"
    python = "def solve(self, s: str, numOps: int) -> int:\n"
    des = __doc__
    degree = 2
    idx = "3398"

    def solve(self, s: str, numOps: int) -> int:
        from heapq import heapify, heapreplace
        from itertools import groupby
        cnt = sum((ord(b) ^ i) & 1 for i, b in enumerate(s))
        if min(cnt, len(s) - cnt) <= numOps:
            return 1

        g = (list(t) for _, t in groupby(s))
        # 子串操作后的最长子段长度，原始子串长度，段数
        h = [(-k, k, 1) for k in map(len, g)]
        heapify(h)
        for _ in range(numOps):
            max_seg, k, seg = h[0]
            if max_seg == -2:
                return 2
            heapreplace(h, (-(k // (seg + 1)), k, seg + 1))  # 重新分割
        return -h[0][0]

    def gen(self):
        s_list = []
        numOps_list = []

        for _ in range(NUM):
            length = random.randint(1, 1000)
            s = [random.randint(0, 1) for _ in range(length)]
            s = ''.join(str(i) for i in s)
            s_list.append(s)

            numOps = random.randint(0, length)
            numOps_list.append(numOps)

        return s_list, numOps_list


class Solution12(BenchTesting):
    """
给你一个长度为 n 的二进制字符串 s 和一个整数 numOps。

你可以对 s 执行以下操作，最多 numOps 次：

选择任意下标 i（其中 0 <= i < n），并 翻转 s[i]，即如果 s[i] == '1'，则将 s[i] 改为 '0'，反之亦然。

你需要 最小化 s 的最长 相同 子字符串 的长度，相同子字符串是指子字符串中的所有字符都相同。

返回执行所有操作后可获得的 最小 长度。



示例 1：

输入: s = "000001", numOps = 1

输出: 2

解释:

将 s[2] 改为 '1'，s 变为 "001001"。最长的所有字符相同的子串为 s[0..1] 和 s[3..4]。

示例 2：

输入: s = "0000", numOps = 2

输出: 1

解释:

将 s[0] 和 s[2] 改为 '1'，s 变为 "1010"。

示例 3：

输入: s = "0101", numOps = 0

输出: 1
    """

    cangjie = "func solve(s: String, numOps: Int64): Int64 {\n"
    python = "def solve(self, s: str, numOps: int) -> int:\n"
    des = __doc__
    degree = 2
    idx = "3399"

    max = lambda a, b: a if a > b else b  # 手写 max 更快

    def solve(self, s: str, numOps: int) -> int:
        from heapq import heapify, heapreplace
        from itertools import groupby
        cnt = sum((ord(b) ^ i) & 1 for i, b in enumerate(s))
        if min(cnt, len(s) - cnt) <= numOps:
            return 1

        g = (list(t) for _, t in groupby(s))
        # 子串操作后的最长子段长度，原始子串长度，段数
        h = [(-k, k, 1) for k in map(len, g)]
        heapify(h)
        for _ in range(numOps):
            max_seg, k, seg = h[0]
            if max_seg == -2:
                return 2
            heapreplace(h, (-(k // (seg + 1)), k, seg + 1))  # 重新分割
        return -h[0][0]

    def gen(self):
        s_list = []
        numOps_list = []

        for _ in range(NUM):
            length = random.randint(1, 10000)
            s = [random.randint(0, 1) for _ in range(length)]
            s = ''.join(str(i) for i in s)
            s_list.append(s)

            numOps = random.randint(0, length)
            numOps_list.append(numOps)

        return s_list, numOps_list


# 428 week
class Solution13(BenchTesting):
    """
给你一个二维数组 events，表示孩子在键盘上按下一系列按钮触发的按钮事件。

每个 events[i] = [indexi, timei] 表示在时间 timei 时，按下了下标为 indexi 的按钮。

数组按照 time 的递增顺序排序。
按下一个按钮所需的时间是连续两次按钮按下的时间差。按下第一个按钮所需的时间就是其时间戳。
返回按下时间 最长 的按钮的 index。如果有多个按钮的按下时间相同，则返回 index 最小的按钮。



示例 1：

输入： events = [[1,2],[2,5],[3,9],[1,15]]

输出： 1

解释：

下标为 1 的按钮在时间 2 被按下。
下标为 2 的按钮在时间 5 被按下，因此按下时间为 5 - 2 = 3。
下标为 3 的按钮在时间 9 被按下，因此按下时间为 9 - 5 = 4。
下标为 1 的按钮再次在时间 15 被按下，因此按下时间为 15 - 9 = 6。
最终，下标为 1 的按钮按下时间最长，为 6。

示例 2：

输入： events = [[10,5],[1,7]]

输出： 10

解释：

下标为 10 的按钮在时间 5 被按下。
下标为 1 的按钮在时间 7 被按下，因此按下时间为 7 - 5 = 2。
最终，下标为 10 的按钮按下时间最长，为 5。
    """

    cangjie = "func solve(events: Array<Array<Int64>>): Int64 {\n"
    python = "def solve(self, events: List[List[int]]) -> int:\n"
    des = __doc__
    degree = 0
    idx = "3386"

    def solve(self, events: List[List[int]]) -> int:
        idx, max_diff = events[0]
        for (_, t1), (i, t2) in pairwise(events):
            d = t2 - t1
            if d > max_diff or d == max_diff and i < idx:
                idx, max_diff = i, d
        return idx

    def gen(self):
        events_list = []

        for _ in range(NUM):
            events_length = random.randint(1, 1000)
            events = []

            for _ in range(events_length):
                indexi = random.randint(1, 100000)
                timei = random.randint(1, 100000)
                events.append([indexi, timei])

            sorted_events = sorted(events, key=lambda x: x[1])
            events_list.append(sorted_events)

        return [events_list]


class Solution14(BenchTesting):
    """
给你一个整数数组 nums 。

如果数组 nums 的一个分割满足以下条件，我们称它是一个 美丽 分割：

数组 nums 分为三段 非空子数组：nums1 ，nums2 和 nums3 ，三个数组 nums1 ，nums2 和 nums3 按顺序连接可以得到 nums 。
子数组 nums1 是子数组 nums2 的 前缀 或者 nums2 是 nums3 的 前缀。
请你返回满足以上条件的分割 数目 。

子数组 指的是一个数组里一段连续 非空 的元素。

前缀 指的是一个数组从头开始到中间某个元素结束的子数组。



示例 1：

输入：nums = [1,1,2,1]

输出：2

解释：

美丽分割如下：

nums1 = [1] ，nums2 = [1,2] ，nums3 = [1] 。
nums1 = [1] ，nums2 = [1] ，nums3 = [2,1] 。
示例 2：

输入：nums = [1,2,3,4]

输出：0

解释：

没有美丽分割。
    """

    cangjie = "func solve(nums: Array<Int64>): Int64 {\n"
    python = "def solve(self, nums: List[int]) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3388"

    def calc_z(self, s: List[int]) -> List[int]:
        n = len(s)
        z = [0] * n
        box_l = box_r = 0  # z-box 左右边界
        for i in range(1, n):
            if i <= box_r:
                # 手动 min，加快速度
                x = z[i - box_l]
                y = box_r - i + 1
                z[i] = x if x < y else y
            while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                box_l, box_r = i, i + z[i]
                z[i] += 1
        return z

    def solve(self, nums: List[int]) -> int:
        z0 = self.calc_z(nums)
        n = len(nums)
        ans = 0
        for i in range(1, n - 1):
            z = self.calc_z(nums[i:])
            for j in range(i + 1, n):
                if i <= j - i and z0[i] >= i or z[j - i] >= j - i:
                    ans += 1
        return ans

    def gen(self):
        nums = gen_lists_int(int_min=0, int_max=50)
        return [nums]


class Solution15(BenchTesting):
    """
给你一个字符串 s 。

如果字符串 t 中的字符出现次数相等，那么我们称 t 为 好的 。

你可以执行以下操作 任意次 ：

从 s 中删除一个字符。
往 s 中添加一个字符。
将 s 中一个字母变成字母表中下一个字母。
注意 ，第三个操作不能将 'z' 变为 'a' 。

请你返回将 s 变 好 的 最少 操作次数。



示例 1：

输入：s = "acab"

输出：1

解释：

删掉一个字符 'a' ，s 变为好的。

示例 2：

输入：s = "wddw"

输出：0

解释：

s 一开始就是好的，所以不需要执行任何操作。

示例 3：

输入：s = "aaabc"

输出：2

解释：

通过以下操作，将 s 变好：

将一个 'a' 变为 'b' 。
往 s 中插入一个 'c' 。
    """

    cangjie = "func solve(s: String): Int64 {\n"
    python = "def solve(self, s: str) -> int:\n"
    des = __doc__
    degree = 2
    idx = "3389"

    def solve(self, s: str) -> int:
        from string import ascii_lowercase
        cnt = Counter(s)
        cnt = [cnt[c] for c in ascii_lowercase]

        ans = len(s)  # target = 0 时的答案
        f = [0] * 27
        for target in range(1, max(cnt) + 1):
            f[25] = min(cnt[25], abs(cnt[25] - target))
            for i in range(24, -1, -1):
                x = cnt[i]
                if x == 0:
                    f[i] = f[i + 1]
                    continue
                # 单独操作 x（变成 target 或 0）
                f[i] = f[i + 1] + min(x, abs(x - target))
                # x 变成 target 或 0，y 变成 target
                y = cnt[i + 1]
                if 0 < y < target:  # 只有当 y 需要变大时，才去执行第三种操作
                    t = target if x > target else 0
                    f[i] = min(f[i], f[i + 2] + max(x - t, target - y))
            ans = min(ans, f[0])
        return ans

    def gen(self):
        s_list = []

        for _ in range(NUM):
            import string
            chars = string.ascii_lowercase
            length = random.randint(1, 2000)

            s = [random.choice(chars) for _ in range(length)]
            s_list.append(''.join(s))

        return [s_list]


# 427 week
class Solution16(BenchTesting):
    """
给你一个整数数组 nums，它表示一个循环数组。请你遵循以下规则创建一个大小 相同 的新数组 result ：

对于每个下标 i（其中 0 <= i < nums.length），独立执行以下操作：
如果 nums[i] > 0：从下标 i 开始，向 右 移动 nums[i] 步，在循环数组中落脚的下标对应的值赋给 result[i]。
如果 nums[i] < 0：从下标 i 开始，向 左 移动 abs(nums[i]) 步，在循环数组中落脚的下标对应的值赋给 result[i]。
如果 nums[i] == 0：将 nums[i] 的值赋给 result[i]。
返回新数组 result。

注意：由于 nums 是循环数组，向右移动超过最后一个元素时将回到开头，向左移动超过第一个元素时将回到末尾。



示例 1：

输入： nums = [3,-2,1,1]

输出： [1,1,1,3]

解释：

对于 nums[0] 等于 3，向右移动 3 步到 nums[3]，因此 result[0] 为 1。
对于 nums[1] 等于 -2，向左移动 2 步到 nums[3]，因此 result[1] 为 1。
对于 nums[2] 等于 1，向右移动 1 步到 nums[3]，因此 result[2] 为 1。
对于 nums[3] 等于 1，向右移动 1 步到 nums[0]，因此 result[3] 为 3。
示例 2：

输入： nums = [-1,4,-1]

输出： [-1,-1,4]

解释：

对于 nums[0] 等于 -1，向左移动 1 步到 nums[2]，因此 result[0] 为 -1。
对于 nums[1] 等于 4，向右移动 4 步到 nums[2]，因此 result[1] 为 -1。
对于 nums[2] 等于 -1，向左移动 1 步到 nums[1]，因此 result[2] 为 4。
    """

    cangjie = "func solve(nums: Array<Int64>): Array<Int64> {\n"
    python = "def solve(self, nums: List[int]) -> List[int]:\n"
    des = __doc__
    degree = 0
    idx = "3379"

    def solve(self, nums: List[int]) -> List[int]:
        n = len(nums)
        return [nums[(i + x) % n] for i, x in enumerate(nums)]

    def gen(self):
        nums = gen_lists_int(int_min=-100, int_max=100)

        return [nums]


class Solution17(BenchTesting):
    """
给你一个数组 points，其中 points[i] = [xi, yi] 表示无限平面上一点的坐标。

你的任务是找出满足以下条件的矩形可能的 最大 面积：

矩形的四个顶点必须是数组中的 四个 点。
矩形的内部或边界上 不能 包含任何其他点。
矩形的边与坐标轴 平行 。
返回可以获得的 最大面积 ，如果无法形成这样的矩形，则返回 -1。


示例 1：

输入： points = [[1,1],[1,3],[3,1],[3,3]]

输出：4

解释：
我们可以用这 4 个点作为顶点构成一个矩形，并且矩形内部或边界上没有其他点。因此，最大面积为 4 。


示例 2：

输入： points = [[1,1],[1,3],[3,1],[3,3],[2,2]]

输出：-1

解释：
唯一一组可能构成矩形的点为 [1,1], [1,3], [3,1] 和 [3,3]，但点 [2,2] 总是位于矩形内部。因此，返回 -1 。


示例 3：

输入： points = [[1,1],[1,3],[3,1],[3,3],[1,2],[3,2]]

输出：2

解释：
点 [1,3], [1,2], [3,2], [3,3] 可以构成面积最大的矩形，面积为 2。此外，点 [1,1], [1,2], [3,1], [3,2] 也可以构成一个符合题目要求的矩形，面积相同。
    """

    cangjie = "func solve(points: Array<Array<Int64>>): Int64 {\n"
    python = "def solve(self, points: List[List[int]]) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3380"

    def solve(self, points: List[List[int]]) -> int:
        from itertools import combinations
        return max(
            [(x4 - x1) * (y4 - y1) for (x1, y1), (x2, y2), (x3, y3), (x4, y4) in combinations(sorted(points), 4) if
             x1 == x2 and x3 == x4 and y1 == y3 and y2 == y4 and all(
                 x < x1 or x4 < x or y < y1 or y4 < y for x, y in points if
                 (x, y) not in ((x1, y1), (x2, y2), (x3, y3), (x4, y4)))], default=-1)

    def gen(self):
        points_list = []
        for _ in range(NUM):
            length = random.randint(1, 10)
            points = []
            for _ in range(length):
                i = random.randint(1, 5)
                j = random.randint(1, 5)
                points.append([i, j])

            points_list.append(points)

        return [points_list]


class Solution18(BenchTesting):
    """
给你一个整数数组 nums 和一个整数 k 。

返回 nums 中一个 非空子数组 的 最大 和，要求该子数组的长度可以 被 k 整除。


示例 1：

输入： nums = [1,2], k = 1

输出： 3

解释：

子数组 [1, 2] 的和为 3，其长度为 2，可以被 1 整除。

示例 2：

输入： nums = [-1,-2,-3,-4,-5], k = 4

输出： -10

解释：

满足题意且和最大的子数组是 [-1, -2, -3, -4]，其长度为 4，可以被 4 整除。

示例 3：

输入： nums = [-5,1,2,-3,4], k = 2

输出： 4

解释：

满足题意且和最大的子数组是 [1, 2, -3, 4]，其长度为 4，可以被 2 整除。
    """

    cangjie = "func solve(nums: Array<Int64>, k: Int64): Int64 {\n"
    python = "def solve(self, nums: List[int], k: int) -> int:\n"
    des = __doc__
    degree = 1
    idx = "3381"

    def solve(self, nums: List[int], k: int) -> int:
        min_s = [inf] * k
        min_s[-1] = s = 0
        ans = -inf
        for j, x in enumerate(nums):
            s += x
            i = j % k
            ans = max(ans, s - min_s[i])
            min_s[i] = min(min_s[i], s)
        return ans

    def gen(self):
        nums = []
        ks = []

        for _ in range(NUM):
            length = random.randint(1, 200)
            k = random.randint(1, length)
            num = [random.randint(-100, 100) for _ in range(length)]

            ks.append(k)
            nums.append(num)

        return nums, ks


class Solution19(BenchTesting):
    """
在无限平面上有 n 个点。给定两个整数数组 xCoord 和 yCoord，其中 (xCoord[i], yCoord[i]) 表示第 i 个点的坐标。

你的任务是找出满足以下条件的矩形可能的 最大 面积：

矩形的四个顶点必须是数组中的 四个 点。
矩形的内部或边界上 不能 包含任何其他点。
矩形的边与坐标轴 平行 。
返回可以获得的 最大面积 ，如果无法形成这样的矩形，则返回 -1。

示例 1：

输入： xCoord = [1,1,3,3], yCoord = [1,3,1,3]

输出： 4

解释：
我们可以用这 4 个点作为顶点构成一个矩形，并且矩形内部或边界上没有其他点。因此，最大面积为 4 。

示例 2：

输入： xCoord = [1,1,3,3,2], yCoord = [1,3,1,3,2]

输出： -1

解释：
唯一一组可能构成矩形的点为 [1,1], [1,3], [3,1] 和 [3,3]，但点 [2,2] 总是位于矩形内部。因此，返回 -1 。

示例 3：

输入： xCoord = [1,1,3,3,1,3], yCoord = [1,3,1,3,2,2]

输出： 2

解释：
点 [1,3], [1,2], [3,2], [3,3] 可以构成面积最大的矩形，面积为 2。此外，点 [1,1], [1,2], [3,1], [3,2] 也可以构成一个符合题目要求的矩形，面积相同。
    """

    cangjie = "func solve(xCoord: Array<Int64>, yCoord: Array<Int64>): Int64 {\n"
    python = "def solve(self, xCoord: List[int], yCoord: List[int]) -> int:\n"
    des = __doc__
    degree = 2
    idx = "3382"

    # 树状数组模板
    class Fenwick:
        def __init__(self, n: int):
            self.tree = [0] * (n + 1)

        def add(self, i: int) -> None:
            while i < len(self.tree):
                self.tree[i] += 1
                i += i & -i

        # [1,i] 中的元素和
        def pre(self, i: int) -> int:
            res = 0
            while i > 0:
                res += self.tree[i]
                i &= i - 1
            return res

        # [l,r] 中的元素和
        def query(self, l: int, r: int) -> int:
            return self.pre(r) - self.pre(l - 1)

    def solve(self, xCoord: List[int], yCoord: List[int]) -> int:
        x_map = defaultdict(list)  # 同一列的所有点的纵坐标
        y_map = defaultdict(list)  # 同一行的所有点的横坐标
        for x, y in zip(xCoord, yCoord):
            x_map[x].append(y)
            y_map[y].append(x)

        # 预处理每个点的正下方的点
        below = {}
        for x, ys in x_map.items():
            ys.sort()
            for y1, y2 in pairwise(ys):
                below[(x, y2)] = y1

        # 预处理每个点的正左边的点
        left = {}
        for y, xs in y_map.items():
            xs.sort()
            for x1, x2 in pairwise(xs):
                left[(x2, y)] = x1

        # 离散化用
        xs = sorted(x_map)
        ys = sorted(y_map)

        # 收集询问：矩形区域（包括边界）的点的个数
        queries = []
        # 枚举 (x2,y2) 作为矩形的右上角
        for x2, list_y in x_map.items():
            for y1, y2 in pairwise(list_y):
                # 计算矩形左下角 (x1,y1)
                x1 = left.get((x2, y2), None)
                # 矩形右下角的左边的点的横坐标必须是 x1
                # 矩形左上角的下边的点的纵坐标必须是 y1
                if x1 is not None and left.get((x2, y1), None) == x1 and below.get((x1, y2), None) == y1:
                    queries.append((
                        bisect_left(xs, x1),  # 离散化
                        bisect_left(xs, x2),
                        bisect_left(ys, y1),
                        bisect_left(ys, y2),
                        (x2 - x1) * (y2 - y1),
                    ))

        # 离线询问
        grouped_queries = [[] for _ in range(len(xs))]
        for i, (x1, x2, y1, y2, _) in enumerate(queries):
            if x1 > 0:
                grouped_queries[x1 - 1].append((i, -1, y1, y2))
            grouped_queries[x2].append((i, 1, y1, y2))

        # 回答询问
        res = [0] * len(queries)
        tree = self.Fenwick(len(ys))
        for x, qs in zip(xs, grouped_queries):
            # 把横坐标为 x 的所有点都加到树状数组中
            for y in x_map[x]:
                tree.add(bisect_left(ys, y) + 1)  # 离散化
            for qid, sign, y1, y2 in qs:
                # 查询横坐标 <= x（已满足）且纵坐标在 [y1,y2] 中的点的个数
                res[qid] += sign * tree.query(y1 + 1, y2 + 1)

        ans = -1
        for cnt, q in zip(res, queries):
            if cnt == 4:
                ans = max(ans, q[4])  # q[4] 保存着矩形面积
        return ans

    def gen(self):
        xCoord_list = []
        yCoord_list = []

        for _ in range(NUM):
            length = random.randint(1, 10)
            xCoord = [random.randint(1, 5) for _ in range(length)]
            yCoord = [random.randint(1, 5) for _ in range(length)]

            xCoord_list.append(xCoord)
            yCoord_list.append(yCoord)

        return xCoord_list, yCoord_list


# 426 week
class Solution20(BenchTesting):
    """
给你一个正整数 n。

返回 大于等于 n 且二进制表示仅包含 置位 位的 最小 整数 x 。

置位 位指的是二进制表示中值为 1 的位。



示例 1：

输入： n = 5

输出： 7

解释：

7 的二进制表示是 "111"。

示例 2：

输入： n = 10

输出： 15

解释：

15 的二进制表示是 "1111"。

示例 3：

输入： n = 3

输出： 3

解释：

3 的二进制表示是 "11"。
    """

    cangjie = "func solve(n: Int64): Int64 {\n"
    python = "def solve(self, n: int) -> int:\n"
    des = __doc__
    degree = 0
    idx = "3370"

    def solve(self, n: int) -> int:
        return (1 << n.bit_length()) - 1

    def gen(self):
        n = gen_int(int_min=1, int_max=1000)

        return [n]