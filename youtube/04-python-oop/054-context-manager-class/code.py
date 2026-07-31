class Timer:
    def __enter__(self):
        import time
        self.start = time.time()
        return self

    def __exit__(self, *args):
        import time
        print(f"elapsed: {time.time() - self.start:.3f}s")

with Timer():
    total = sum(range(1_000_000))
print(total)

# __exit__ always runs, even on error
class Guard:
    def __enter__(self): return self
    def __exit__(self, exc_type, *a):
        print("cleanup ran; error was", exc_type)
        return True   # suppress the exception

with Guard():
    raise ValueError("boom")
print("survived")
