import os
import random

from tests.spark.unit.dataset_utils import datasets

dss = [("used_cars_dataset_tmp", datasets()["used_cars_dataset_tmp"])]
# dss = [('used_cars_dataset', datasets()['used_cars_dataset'])]
# dss = datasets().items()

for name, ds in dss:
    print(f"Name: {name}")
    path = ds['path']
    base_dir = os.path.dirname(path)
    filename, ext = os.path.splitext(os.path.basename(path))
    train_path = os.path.join(base_dir, f"{filename}_train{ext}")
    test_path = os.path.join(base_dir, f"{filename}_test{ext}")

    with open(path, "r") as f:
        data = f.readlines()

    header = data[0]
    data = data[1:]
    data_rand = [(line, random.random()) for line in data]
    train_part = [line for line, r in data_rand if r <= 0.8]
    test_part = [line for line, r in data_rand if r > 0.8]

    hash_intersection = set(hash(l) for l in train_part).intersection(set(hash(l) for l in test_part))
    # assert len(hash_intersection) == 0, f"Len: {len(hash_intersection)}"
    if len(hash_intersection) != 0:
        print(f"Warning! Intersections length: {len(hash_intersection)}. "
              f"Train part length: {len(train_part)}. "
              f"Test part length: {len(test_part)}.")

    with open(train_path, "w") as f:
        part = ''.join(train_part)
        f.write(header)
        f.write(part)

    with open(test_path, "w") as f:
        part = ''.join(test_part)
        f.write(header)
        f.write(part)
