from time import time


class Timer:
    def __init__(self) -> None:
        self.time_dict = dict()
        self.last_time = None

    def clear(self):
        self.time_dict = dict()
        self.last_time = None

    def prepare(self):
        self.last_time = time()

    def time(self, describe: str):
        self.time_dict.setdefault(describe, []).append(time() - self.last_time)
        self.last_time = time()

    def to_str(self, str_format='{describe}={avg_time:.2f}s', sep='|', steps=0):
        ret = sep.join([str_format.format(describe=k, avg_time=sum(t_list[-steps:]) / len(t_list[-steps:]),
                                          count=len(t_list[-steps:])) for k, t_list in self.time_dict.items()])
        return ret
