import time

class Timer():
    def __enter__(self):
        self.start = time.time()
        return self.start
    
    def __exit__(self, *args):
        self.end = time.time()
        self.innterval = self.end - self.start
        print(f"执行时间： {self.innterval:.2f} 秒")