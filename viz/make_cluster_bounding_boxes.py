from pathlib import Path
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    args = parser.parse_args()

