from tqdm import tqdm
import argparse
import tarfile
from pathlib import Path
import random


def shard_files(input_dir: Path, output_dir: Path, prefix: str, shard_size: int):
    # Get list of all files in the input directory
    all_files = list(input_dir.glob("*"))

    # Shuffle the list of files
    random.shuffle(all_files)

    # Group them in shards of the appropriate size
    total_files = len(all_files)
    num_shards = total_files // shard_size + (1 if total_files % shard_size else 0)

    for shard_num in tqdm(range(num_shards)):
        start_index = shard_num * shard_size
        end_index = start_index + shard_size
        shard_files = all_files[start_index:end_index]

        # Tar the shard
        tar_filename = output_dir / f"{prefix}_shard_{shard_num + 1:03}.tar"
        with tarfile.open(tar_filename, "w") as tar:
            for file in shard_files:
                tar.add(file, arcname=file.name)

    print(f"Processed {num_shards} shards to {output_dir}")


if __name__ == "__main__":
    """Split and tar files from input directory"""
    parser = argparse.ArgumentParser(
        description="Shard and tar files from input directory. Replaces the utility of tarp, which is very difficult to install."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        help="The directory containing files to be sharded and tarred",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="The directory where the tarred shards should be saved",
    )
    parser.add_argument(
        "--prefix", type=str, help="Prefix for naming the tarred shards"
    )
    parser.add_argument(
        "--shard_size", type=int, help="Number of files per shard", default=1000
    )

    args = parser.parse_args()

    shard_files(args.input_dir, args.output_dir, args.prefix, args.shard_size)
