# coding:utf-8
import threading
import time

def action(arg, event):
    while not event.isSet():
        print('Thread %s is ready' % arg)
        time.sleep(1)
    event.wait()
    while event.isSet():
        print('Thread %s is running' % arg)
        time.sleep(1)

event = threading.Event()
for i in range(3):
    t =threading.Thread(target=action,args=(i,event))
    t.start()

time.sleep(2)
print('-----set-----')
event.set()
time.sleep(2)
print('-----clear----')
event.clear()
print('main_thread end!')

# setDeamon=Flase
# t.setDaemon(True)  # 设置线程为后台线程
