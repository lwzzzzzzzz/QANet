import threading
import time

total=5
lock=threading.RLock()

def sale():
    global total
    lock.acquire()
    time.sleep(.5)
    print(total)
    time.sleep(.5)
    total-=1
    lock.release()


if __name__ == '__main__':
    threads=[]

    for i in range(5):
        t=threading.Thread(target=sale,args=())
        threads.append(t)

    for t in threads:
        t.start()
