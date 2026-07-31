class Timer:
    def __enter__(self):
        print("starting")
        return self

    def __exit__(self, *exc):
        print("done")
        return False


with Timer():
    print("working")
