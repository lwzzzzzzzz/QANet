import threading
import time

count = 0

def show1():
    global count
    time.sleep(1)
    for i in range(5):
        count += 1
    print(count)

def show2():
    global count
    for i in range(5):
        count *= 2
    print(count)


t1 = threading.Thread(target=show1)
t1.start()
t2 = threading.Thread(target=show2)
t2.start()
