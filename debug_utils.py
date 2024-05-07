import time

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
        return result
    return wrapper


def get_size_of_tensor(x):
    return (x.indices().nelement() * 8 + x.values().nelement() * 4) / 1024**2
