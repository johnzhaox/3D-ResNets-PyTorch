import os
import argparse
import random
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dir_path',
        default=None,
        type=Path,
        help='Directory path of videos')
    parser.add_argument(
        'dst_path',
        default=None,
        type=Path,
        help='Directory path of annotation')
    parser.add_argument(
        'test_rate',
        default=0.2,
        type=float,
        help='Directory path of annotation')
    args = parser.parse_args()

    if args.dst_path.exists():
        os.system("rm -rf %s" % args.dst_path)
        args.dst_path.mkdir()

    class_dir_paths = [_ for _ in args.dir_path.iterdir() if _.is_dir()]
    class_labels = [_.name for _ in class_dir_paths]
    for class_dir in class_dir_paths:
        data_folders = [_ for _ in class_dir.iterdir() if _.is_dir()]
        random.shuffle(data_folders)
        n_test = int(len(data_folders) * args.test_rate)
        test_data_folders = data_folders[:n_test]
        train_data_folders = data_folders[n_test:]

        with open(args.dst_path / (class_dir.name + "_split.csv"), "w") as f:
            for data_folder in train_data_folders:
                for data_path in data_folder.glob("*.avi"):
                    f.write(data_path.name + " 1\n")
            for data_folder in test_data_folders:
                for data_path in data_folder.glob("*.avi"):
                    f.write(data_path.name + " 2\n")





