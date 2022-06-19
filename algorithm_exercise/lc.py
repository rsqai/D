import os
from typing import List
from collections import defaultdict
from collections import Counter
from itertools import permutations
from itertools import combinations
from functools import cmp_to_key
import heapq
import pprint


def longestCommonSubsequence(text1: str, text2: str) -> int:
    """
    最长公共子串, 返回lcs的长度。
    dp解法，dp[i][j]为text1[:i]和text2[:j]的lcs长度
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1)] * (m + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def panduanzixulie(t, s):
    """判断字符串子序列，t是否为s的子序列输出最后一个序列的起始位置"""
    tp, sp = len(t) - 1, len(s) - 1
    while sp >= 0:
        if t[tp] == s[sp]:
            tp -= 1
        # t遍历完成后直接break
        if tp < 0:
            break
        # t未遍历完成，每次sp递减1
        sp -= 1
    return sp


def zhengshuduizuixiaohe(nums1, nums2, k):
    """k对整数最小和"""
    m, n = len(nums1), len(nums2)
    ans = []
    pq = [(nums1[i] + nums2[0], i, 0) for i in range(min(k, m))]
    while pq and len(ans) < k:
        _, i, j = heapq.heappop(pq)
        ans.append([nums1[i], nums2[j]])
        if j + 1 < n:
            heapq.heappush(pq, (nums1[i] + nums2[j + 1], i, j + 1))
    return ans


def lisp(s):
    """仿LISP计算，add/sub/mul/div，错误输出error"""

    def helper(op, a, b):
        res = ""
        if op == "add":
            res = int(a) + int(b)
        elif op == "sub":
            res = int(a) - int(b)
        elif op == "mul":
            res = int(a) * int(b)
        elif op == "div":
            if int(b) == 0:
                res = "error"
            else:
                res = int(a) / int(b)
        return res

    stack, i = [], 0
    # 栈初始化
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        if s[i].isalpha():
            stack.append(s[i:i + 3])
            i += 3
        elif s[i].isdigit():
            d = s[i]
            i += 1
            while i < len(s) and s[i].isdigit():
                d += s[i]
                i += 1
            stack.append(d)
        else:
            stack.append(s[i])
            i += 1
    # 计算结果
    cal_stack = []
    for it in stack:
        if it == ")":
            b_ = cal_stack.pop()
            a_ = cal_stack.pop()
            op_ = cal_stack.pop()
            cal_stack.pop()
            cal_stack.append(helper(op_, a_, b_))
        else:
            cal_stack.append(it)
    return cal_stack[0]


def zhaopengyou(nums):
    """
    N个⼩朋友站成⼀队， 第i个⼩朋友的身⾼为height[i]， 第i个⼩朋友可以看到的第⼀个⽐⾃⼰身⾼更⾼的⼩朋友j，那么j是i的好朋友(要求j > i)
    请重新⽣成⼀个列表，对应位置的输出是每个⼩朋友的好朋友位置，如果没有看到好朋友，请在该位置⽤0代替
    :param nums:
    :return:
    """
    res = [0] * len(nums)
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[j] > nums[i]:
                res[i] = j
                break
    return res


def zimupaixu(s):
    """
    仅包含字⺟的字符串，不包含空格，统计字符串中各个字⺟（区分⼤⼩写）出现的次数，
    并按照字⺟出现次数从⼤到⼩的顺序输出各个字⺟及其出现次数。如果次数相同，按照
    ⾃然顺序进⾏排序，且⼩写字⺟在⼤写字⺟之前。
    :param s: xyxyXX
    :return: x:2;y:2;X:2
    """
    c = Counter(s)
    res = list(map(lambda x: x[0]+":"+str(x[1]), sorted(c.items(), key=lambda x: x[1])))
    return res


def luanxuzhengshujueduizhizhihe(nums):
    """
    给定一个随机的整数数组（含正负整数），找出其中的两个数，其和的绝对值为最小值
    返回两个数，按从小到大返回以及绝对值。每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍
    :param nums:
    :return:
    """
    nums.sort()
    if nums[0] >= 0:
        return nums[0], nums[1], abs(nums[0]+nums[1])
    if nums[-1] <= 0:
        return nums[-2], nums[-1], abs(nums[-2]+nums[-1])
    left, right = 0, len(nums)-1
    # todo:无需双层遍历，只需遍历一遍O(n)
    # todo:注意left要小于right+1，因为循环内要进行right-1或者left+1的操作
    # todo:而且在abs(left+right)情况中要分情况讨论，只有当下一个使得abs减小时才能+-1，否则break
    while left < right + 1:
        if nums[left]+nums[right] > 0:
            if nums[left]+nums[right-1] >= 0:
                right -= 1
            else:
                break
        elif nums[left]+nums[right] < 0:
            if nums[left+1]+nums[right] <= 0:
                left += 1
            else:
                break
        elif nums[left]+nums[right] == 0:
            break
    return nums[left], nums[right], abs(nums[left] + nums[right])


if __name__ == '__main__':
    # print(longestCommonSubsequence("abcde", "ace"))
    # print(panduanzixulie("abc", "abcaybec"))
    # print(lisp("(sub (mul 2 4) (div 9 3))"))
    # print(zhaopengyou([100, 95]))
    # print(zhaopengyou([123, 124, 125, 121, 119, 122, 126, 123]))
    # print(zimupaixu("xyxyXX"))
    print(luanxuzhengshujueduizhizhihe([-1, -3, 7, 5, 11, 15]))
    print(luanxuzhengshujueduizhizhihe([7, 5, 11, 15]))
    print(luanxuzhengshujueduizhizhihe([-7, -5, -11, -15]))
    print(luanxuzhengshujueduizhizhihe([0, 0, 0, 0]))
    pass
