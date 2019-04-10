import timeit
 
# 列表list
start1 = timeit.default_timer()

L1 = [i**2 for i in range(1000000)]

end1 = timeit.default_timer()

print(end1-start1)

# for循环
start2 = timeit.default_timer()

L2 = []

for n in range(1000000):
    L2.append(n**2)

end2 = timeit.default_timer()

print(end2-start2)
