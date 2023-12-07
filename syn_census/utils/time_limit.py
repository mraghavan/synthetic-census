import multiprocessing

class Func(object):
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
    
    def __call__(self):
        return self.func(*self.args, **self.kwargs)

def wrapper(func, result_queue):
    try:
        result = func()
        result_queue.put(result)
    except Exception as e:
        result_queue.put(e)

def run_with_timeout(func, timeout_seconds):
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=wrapper, args=(func, result_queue))

    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"Function exceeded the time limit of {timeout_seconds} seconds.")

    result = result_queue.get()
    
    if isinstance(result, Exception):
        raise result  # If an exception occurred in the function, raise it.
    
    return result
