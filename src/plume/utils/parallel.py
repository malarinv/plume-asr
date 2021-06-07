from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm


def parallel_apply(fn, iterable, workers=8, pool="thread", verbose=True):
    # warm-up (doesn't work there fn conditionals that doesn't follow hot path)
    # fn(iterable[0])
    if pool == "thread":
        with ThreadPoolExecutor(max_workers=workers) as exe:
            if verbose:
                print(f"parallelly applying {fn}")
                return [
                    res
                    for res in tqdm(
                        exe.map(fn, iterable),
                        position=0,
                        leave=True,
                        total=len(iterable),
                    )
                ]
            else:
                return [res for res in exe.map(fn, iterable)]
    elif pool == "process":
        with ProcessPoolExecutor(max_workers=workers) as exe:
            if verbose:
                print(f"parallelly applying {fn}")
                with tqdm(total=len(iterable)) as progress:
                    futures = []
                    for i in iterable:
                        future = exe.submit(fn, i)
                        future.add_done_callback(lambda p: progress.update())
                        futures.append(future)
                    results = []
                    for future in futures:
                        result = future.result()
                        results.append(result)
                    return result
            else:
                return [res for res in exe.map(fn, iterable)]
    elif pool == "none":
        if verbose:
            return list(map(fn, tqdm(iterable)))
        else:
            return list(map(fn, iterable))
    else:
        raise Exception(f"unsupported pool type - {pool}")
