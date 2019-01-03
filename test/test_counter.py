from collections import Counter

def test(acount):
    acount['a'] += 1

def test1():
    char_counter = Counter()
    char_counter['b'] = 1
    test(char_counter)

    test(char_counter)
    for k, v in char_counter.items():
        print(k)
    print(char_counter)

test1()
