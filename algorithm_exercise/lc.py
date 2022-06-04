import os
from typing import List
import heapq
import pprint


def longestCommonSubsequence(text1: str, text2: str) -> int:
    """
    最长公共子串, 返回lcs的长度。
    dp解法，dp[i][j]为text1[:i]和text2[:j]的lcs长度
    """
    m, n = len(text1), len(text2)
    dp = [[0]*(n+1)]*(m+1)
    for i in range(1, m+1):
        for j in range(1, n+1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def panduanzixulie(t, s):
    """判断字符串子序列，t是否为s的子序列输出最后一个序列的起始位置"""
    tp, sp = len(t)-1, len(s)-1
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


if __name__ == '__main__':
    # print(longestCommonSubsequence("abcde", "ace"))
    # print(panduanzixulie("abc", "abcaybec"))
    pass
