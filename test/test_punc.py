import string
import re
from collections import Counter
# a = 'ad.,\\/]qweq'
# exclude = set(string.punctuation) # 表示标点符号的字典
# b = ''.join(ch for ch in a if ch not in exclude)
#
# print(a)
# print(exclude)
# print(b)

a = 'i loooove you so much i loooove you you you you'
b = 'i loooove you i loooove you you you but'
a = a.split()
b = b.split()

a_C = Counter(a)
b_C = Counter(b)

common = a_C & b_C
num_same = sum(common.values())
print(a_C)
print(b_C)
print(common)
print(num_same)