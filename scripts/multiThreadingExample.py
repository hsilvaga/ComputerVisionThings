import time 
import threading

start = time.perf_counter()

def doSomething():
    print("sleeping")
    time.sleep(1)
    print("Done sleeping")


threads = []
for _ in range(10):
    t = threading.Thread(target=doSomething)
    t.start()
    threads.append(t)

for thread in threads:
    thread.join()

finish = (time.perf_counter())

print("Time taken: ", finish - start)