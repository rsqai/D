import os
import re


def merge_sort(a):
    """
    归并排序
    :param a:
    :return:
    """

    def merge(left_array, right_array):
        left_index, right_index, merge_array = 0, 0, list()
        while left_index < len(left_array) and right_index < len(right_array):
            if left_array[left_index] <= right_array[right_index]:
                merge_array.append(left_array[left_index])
                left_index += 1
            else:
                merge_array.append(right_array[right_index])
                right_index += 1
        merge_array = merge_array + left_array[left_index:] + right_array[right_index:]
        return merge_array

    def merge_sort_(arr):
        if len(arr) == 1:
            return arr
        left_arr = merge_sort_(arr[:len(arr) // 2])
        right_arr = merge_sort_(arr[len(arr) // 2:])
        return merge(left_arr, right_arr)

    return merge_sort_(a)


def quick_sort(array, i, j):
    """
    快速排序
    :param array:
    :param i:
    :param j:
    :return:
    """
    # 递归终止条件
    if i >= j:
        return
    low = i
    high = j
    key = array[low]
    while i < j:
        while i < j and array[j] > key:
            j -= 1
        # 替补pivot的位置
        array[i] = array[j]
        while i < j and array[i] <= key:
            i += 1
        # j已经替补了pivot，i替补到j位置
        array[j] = array[i]
    # 此时i==j了，下面使用i或者j均可
    array[j] = key
    # 递归drill down，此时i==j了，下面使用i或者j均可
    quick_sort(array, low, i - 1)
    quick_sort(array, i + 1, high)
    return array


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
    # panduanzixulie("abc", "abcaybec")
    print(merge_sort([10, 17, 50, 7, 30, 24, 27, 45, 15, 5, 36, 21]))
    print(quick_sort([10, 17, 50, 7, 30, 24, 27, 45, 15, 5, 36, 21], 0, 11))
    pass
