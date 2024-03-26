def dec(f):
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


@dec  # 当使用这个装饰器，def dec(f):其中的f就变成了下列的double
def double(x):
    print(x*2)
    return x * 2


double(1)

# --------------完全等价于以下


def dec(f):
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


# 当使用这个装饰器，def dec(f):其中的f就变成了下列的double
def double(x):
    print(x*2)
    return x * 2


dec(double)
double(1)
