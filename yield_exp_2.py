import sys
import time 

def fab_memory(max): 
   n, a, b = 0, 0, 1 
   L = [] 
   while n < max: 
       L.append(b) 
       a, b = b, a + b 
       n = n + 1 
   return L

def fab_yield(max): 
    n, a, b = 0, 0, 1 
    while n < max: 
        yield b 
        a, b = b, a + b 
        n = n + 1

rounds = 100000

start = time.perf_counter_ns()
fibonacci = fab_memory(rounds)
end = time.perf_counter_ns()
print('内存消耗型共用时:', end - start)

f = iter(fab_memory(rounds))
while True:
    start = time.perf_counter_ns()
    try:
        next(f)
    except StopIteration:
        end = time.perf_counter_ns()
        print('迭代器与生成器共用时:', end - start)
        break

x = fab_yield(rounds)
while True:
    start = time.perf_counter_ns()
    try:
        next(x)
    except:
        end = time.perf_counter_ns()
        print('使用yield共用时:', end - start)
        break