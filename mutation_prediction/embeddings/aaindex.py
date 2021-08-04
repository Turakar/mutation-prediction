import re

import numpy as np

from mutation_prediction import data as data


def _aaindex_to_float(v):
    if v == "NA":
        return float("nan")
    else:
        return float(v)


def read_aaindex1(keys=None):
    scores = []
    with open("data/embeddings/aaindex1") as fd:
        try:
            last_key = None
            while True:
                line: str = next(fd)
                if line[0] == "H":
                    last_key = line[1:].strip()
                if line[0] == "I" and (keys is None or last_key in keys):
                    acid_order = line[1:].replace(" ", "").replace("/", "").replace("\n", "")
                    data1: str = next(fd)
                    data2: str = next(fd)
                    values1 = re.split(r"\s+", data1.strip())
                    values2 = re.split(r"\s+", data2.strip())
                    vector = np.zeros((data.num_acids(),), dtype=np.float32)
                    for i, (v1, v2) in enumerate(zip(values1, values2)):
                        vector[data.acid_to_index(acid_order[i * 2])] = _aaindex_to_float(v1)
                        vector[data.acid_to_index(acid_order[i * 2 + 1])] = _aaindex_to_float(v2)
                    if not np.isnan(vector).any():
                        scores.append(vector)
        except StopIteration:
            pass
    return np.asarray(scores).T


def read_aaindex1_keys():
    found_keys = []
    with open("data/embeddings/aaindex1") as fd:
        try:
            last_key = None
            while True:
                line: str = next(fd)
                if line[0] == "H":
                    last_key = line[1:].strip()
                if line[0] == "I":
                    acid_order = line[1:].replace(" ", "").replace("/", "").replace("\n", "")
                    data1: str = next(fd)
                    data2: str = next(fd)
                    values1 = re.split(r"\s+", data1.strip())
                    values2 = re.split(r"\s+", data2.strip())
                    vector = np.zeros((data.num_acids(),), dtype=np.float32)
                    for i, (v1, v2) in enumerate(zip(values1, values2)):
                        vector[data.acid_to_index(acid_order[i * 2])] = _aaindex_to_float(v1)
                        vector[data.acid_to_index(acid_order[i * 2 + 1])] = _aaindex_to_float(v2)
                    if not np.isnan(vector).any():
                        found_keys.append(last_key)
        except StopIteration:
            pass
    return found_keys


def read_aaindex3():
    scores = []
    with open("data/embeddings/aaindex3") as fd:
        try:
            while True:
                line: str = next(fd)
                if line[0] == "M":
                    match = re.match(r"M\s+rows\s+=\s+([A-Z]{20}),\s+cols\s+=\s+([A-Z]{20})", line)
                    rows = match.group(1)
                    cols = match.group(2)
                    matrix = np.zeros((len(rows), len(cols)), dtype=np.float32)
                    for a in rows:
                        row: str = next(fd)
                        columns = re.sub(r"\s+", " ", row.strip()).split(" ")
                        i = data.acid_to_index(a)
                        for part, b in zip(columns, cols):
                            value = _aaindex_to_float(part)
                            j = data.acid_to_index(b)
                            matrix[i, j] = matrix[j, i] = value
                    if not np.isnan(matrix).any():
                        scores.append(matrix)
        except StopIteration:
            pass
    scores = np.asarray(scores).transpose((1, 2, 0))
    return scores
