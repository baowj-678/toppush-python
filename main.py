import scipy.io as sio
from python.toppush import topPush as topPush
from python_with_c.toppush import topPush as topPushWithC
import numpy as np
import time


def load_data():
    data = sio.loadmat("data/spambase.mat")
    x = np.array(data['Xtr'].toarray())
    y = np.array(data['ytr'])
    xte = data['Xte'].toarray()
    ans = data['ans']
    return x, y, xte, ans


def functional_test():
    print("  start functional test ".center(100, '-'))
    x, y, xte, ans = load_data()
    python_functional_test(x, y, xte, ans)
    python_with_c_functional_test(x, y, xte, ans)


def python_functional_test(x, y, xte, ans):
    print(" start python functional test ".center(100, '-'))

    w = topPush(x, y)
    pdt = np.dot(xte, w)
    diff = np.abs(ans - pdt).max()
    if diff < 1e-8:
        print("the max diff is: {} less than {}".format(diff, 1e-8))
        print("\033[1;34m" + " python functional test passed ".center(100, '-') + "\033[0m")
    else:
        print("the max diff is: {} larger than {}".format(diff, 1e-8))
        print("\033[1;31m" + " python functional test failed ".center(100, '-') + "\033[0m")


def python_with_c_functional_test(x, y, xte, ans):
    print(" start python-with-c functional test ".center(100, '-'))
    w = topPush(x, y)
    pdt = np.dot(xte, w)
    diff = np.abs(ans - pdt).max()
    if diff < 1e-8:
        print("the max diff is: {} less than {}".format(diff, 1e-8))
        print("\033[1;34m" + " python-with-c functional test passed ".center(100, '-') + "\033[0m")
    else:
        print("the max diff is: {} larger than {}".format(diff, 1e-8))
        print("\033[1;31m" + " python-with-c functional test failed ".center(100, '-') + "\033[0m")


def speed_test():
    print(" start speed test ".center(100, '-'))
    x, y, xte, ans = load_data()
    python_speed_test(x, y)
    python_with_c_speed_test(x, y)

def python_speed_test(x, y):
    print(" start python speed test ".center(100, '-'))
    # warm up
    for i in range(10):
        w = topPush(x, y)
    # test toppush
    begin = time.time_ns()
    for i in range(10):
        w = topPush(x, y)
    end = time.time_ns()
    topPushTime = end - begin
    print(f"topPush time: {topPushTime / 1e9} s")


def python_with_c_speed_test(x, y):
    print(" start python-with-c functional test ".center(100, '-'))
    # warm up
    for i in range(10):
        w = topPushWithC(x, y)
    # test toppushWithC
    begin = time.time_ns()
    for i in range(10):
        w = topPushWithC(x, y)
    end = time.time_ns()
    topPushWithCTime = end - begin
    print(f"topPushWithC time: {topPushWithCTime / 1e9} s")


if __name__ == '__main__':
    functional_test()
    speed_test()
