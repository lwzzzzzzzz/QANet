import threading

def test_run():
    print('your thread is running')

if __name__ == "__main__":
    t = threading.Thread(target=test_run(), args=())
    t.start()