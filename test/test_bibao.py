def outer(some_func):
    def inner():
        print("before some_func")
        ret = some_func()  # 1
        return ret + 1
    return inner

@outer
def foo():
    return 1

# foo = outer(foo)

print(foo())