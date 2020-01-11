import os
import multiprocessing

def get_cpuCount() -> int:
    return os.cpu_count()

def get_threadCount() -> int:
    return multiprocessing.cpu_count()