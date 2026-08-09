import os
import sys
import json
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm
from PIL import Image
import numpy as np


def process(route):
    rgb_full_dir = os.path.join(route, "rgb_full")
    measurements_full_dir = os.path.join(route, "measurements_full")

    frames = len(os.listdir(os.path.join(route, "measurements")))

    if os.path.exists(rgb_full_dir) and os.path.exists(measurements_full_dir):
        if len(os.listdir(rgb_full_dir)) == frames and len(os.listdir(measurements_full_dir)) == frames:
            print('The folder %s was already processed and is complete...' % route)
            return
        else:
            print('The folder %s was already processed and but is not complete...' % route)
            
    try:
        if not os.path.exists(rgb_full_dir):
            os.mkdir(rgb_full_dir)
        if not os.path.exists(measurements_full_dir):
            os.mkdir(measurements_full_dir)
        for i in range(frames):
            img_front = Image.open(os.path.join(route, "rgb_front/%04d.jpg" % i))
            img_left = Image.open(os.path.join(route, "rgb_left/%04d.jpg" % i))
            img_right = Image.open(os.path.join(route, "rgb_right/%04d.jpg" % i))
            img_rear = Image.open(os.path.join(route, "rgb_rear/%04d.jpg" % i))
            new = Image.new(img_front.mode, (800, 2400))
            new.paste(img_front, (0, 0))
            new.paste(img_left, (0, 600))
            new.paste(img_right, (0, 1200))
            new.paste(img_rear, (0, 1800))
            new.save(os.path.join(rgb_full_dir, "%04d.jpg" % i))

            measurements = json.load(
                open(os.path.join(route, "measurements/%04d.json" % i))
            )
            actors_data = json.load(
                open(os.path.join(route, "actors_data/%04d.json" % i))
            )
            affordances = np.load(
                os.path.join(route, "affordances/%04d.npy" % i), allow_pickle=True
            )

            measurements["actors_data"] = actors_data
            measurements["stop_sign"] = affordances.item()["stop_sign"]
            json.dump(
                measurements,
                open(os.path.join(measurements_full_dir, "%04d.json" % i), "w"),
            )
    except Exception as e:
        print(e)
        print('The folder %s has an existing problem, and we will proceed to remove it...' % route)
        os.system('rm -rf %s' % route)


if __name__ == "__main__":
    dataset_root = Path(sys.argv[1]).expanduser()
    index_file   = dataset_root / "dataset_index.txt"

    routes = [
        dataset_root / line.split()[0].strip()
        for line in index_file.read_text().splitlines()
        if line.strip()
    ]

    n_proc = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())) // 2
    print(f"Spawning {n_proc} worker processes …")

    with Pool(n_proc) as pool:
        list(tqdm(pool.imap_unordered(process, routes), total=len(routes)))
