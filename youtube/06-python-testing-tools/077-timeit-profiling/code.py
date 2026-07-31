import timeit

# time a small snippet accurately
t = timeit.timeit("sum(range(100))", number=10000)
print(f"{t:.4f}s for 10k runs")

# compare two approaches
loop = timeit.timeit("r=[]\nfor i in range(100): r.append(i*i)", number=10000)
comp = timeit.timeit("[i*i for i in range(100)]", number=10000)
print(f"loop: {loop:.3f}  comprehension: {comp:.3f}")

# cProfile finds the slow spots in a whole program
import cProfile
def work():
    return sum(i*i for i in range(100000))
cProfile.run("work()")
