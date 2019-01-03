def square_sum(fn):
    def square(*args):
        print("1---", args)
        n = args[0]
        # return n*(n-1)*(2*n-1)/6
        print("2==", n * (n - 1) * (2 * n - 1) / 6)
        print(fn.__name__)
        # fn(n * (n - 1) * (2 * n - 1) / 6)
        print("*" * 15)
        return fn(n * (n - 1) * (2 * n - 1) / 6)

    return square
@square_sum
def sum_a(a):

    print("3=", a)
    print(11111)

a = sum_a(10)