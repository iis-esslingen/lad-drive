#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def collect_copy_tasks(
    route_dir: Path, output_root: Path
) -> List[Tuple[Path, Path]]:
    tasks = []

    # Copy scene_graph with specific extensions
    sg_dir = route_dir / "scene_graph"
    if sg_dir.exists():
        for path in sg_dir.rglob("*"):
            if path.suffix in [".json", ".txt", ".pt"] and path.is_file():
                dest = output_root / path.relative_to(route_dir)
                tasks.append((path, dest))

    # Copy other dirs fully
    for subdir in ["lidar", "lidar_odd", "rgb_full", "measurements_full"]:
        full_dir = route_dir / subdir
        if full_dir.exists():
            for path in full_dir.rglob("*"):
                if path.is_file():
                    dest = output_root / path.relative_to(route_dir)
                    tasks.append((path, dest))

    # Single file: measurements_all.json
    file_path = route_dir / "measurements_all.json"
    if file_path.exists():
        dest = output_root / file_path.relative_to(route_dir)
        tasks.append((file_path, dest))

    return tasks


def copy_file(src_dst: Tuple[Path, Path]) -> None:
    src, dst = src_dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main(dataset_root: str, output_root: str, num_workers: int = 4):
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    index_file = dataset_root / "dataset_index.txt"

    all_tasks = []
    with index_file.open("r") as f:
        for line in f:
            route_path = line.strip().split()[0]
            route_dir = dataset_root / route_path
            out_dir = output_root / route_path
            tasks = collect_copy_tasks(route_dir, out_dir)
            all_tasks.extend(tasks)

    print(f"Collected {len(all_tasks)} files to copy. Starting multiprocessing …")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(copy_file, all_tasks), total=len(all_tasks), desc="Copying"))

    print(f"✅ Done copying to {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel copy of selected dataset files.")
    parser.add_argument("dataset_root", type=str, help="Path to dataset directory (with dataset_index.txt)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()
    main(args.dataset_root, args.output, num_workers=args.workers)
