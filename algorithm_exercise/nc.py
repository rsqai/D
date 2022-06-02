import os
import re


def xiaoxaole(s):
    """消消乐游戏，输入英文字母字符串，两两消除"""
    stack = []
    for i, c in enumerate(s):
        # 栈为空则直接添加
        if not stack:
            stack.append(c)
        # 栈非空则分情况，与栈顶相同则弹出，与栈顶不同则压入
        else:
            if c == stack[-1]:
                stack.pop()
            else:
                stack.append(c)
    print(len(stack))


def panduanzixulie(t, s):
    """判断t是否为s的子序列,找出最后一个序列的起始位置"""
    tp, sp = len(t) - 1, len(s) - 1
    # 因为要找出最后一个序列的起始位置，因此直接倒序查找
    while tp >= 0 and sp >= 0:
        if t[tp] == s[sp]:
            tp -= 1
            sp -= 1
        else:
            sp -= 1
    print(sp+1 if tp == -1 else -1)


if __name__ == '__main__':
    # xiaoxaole("abcccddeeffcg")
    # xiaoxaole("abbacddccc00")
    panduanzixulie("abc", "abcaybec")
    pass
