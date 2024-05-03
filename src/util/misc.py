import errno
import os

import numpy as np


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ensure_dir(path):
    try:
        if not os.path.isdir(path):
            os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Could not create dir " + path)


def display_size(size: int) -> str:
    for unit in ["b", "Kb", "Mb", "Gb"]:
        if size > 1024:
            size = round(size / 1024, 2)
        else:
            break
    return f"{size}{unit}"


def list_avg(l):
    sum = 0
    for i in l:
        sum += i
    return float(sum) / len(l)


def list_stats(list, round_decimal=False):
    stats = {
        "min": min(list),
        "max": max(list),
        "avg": list_avg(list),
        "median": np.percentile(list, 50),
        "pc75": np.percentile(list, 75),
        "pc90": np.percentile(list, 90),
        "all": sum(list),
        "length": len(list)
    }
    if round_decimal is not False:
        for k, v in stats.items():
            if isinstance(v, float):
                stats[k] = round(v, round_decimal)
    return stats


def dict_key_add(d, key, item=None, collect_as_list=False):
    if key not in d:
        if item is None:
            d[key] = 1
        else:
            if collect_as_list:
                d[key] = [item]
            else:
                d[key] = {item}
    else:
        if item is None:
            d[key] += 1
        else:
            if collect_as_list:
                d[key].append(item)
            else:
                d[key].add(item)


def dict_key_inc(d, key, inc: int = 1):
    if key not in d:
        if inc is None:
            d[key] = 1
        else:
            d[key] = inc
    else:
        if inc is None:
            d[key] += 1
        else:
            d[key] += inc


def generator_list_batch(l, batch_size):
    n = len(l)
    for ndx in range(0, n, batch_size):
        yield l[ndx: ndx + batch_size]
