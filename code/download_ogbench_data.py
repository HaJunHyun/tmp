"""Download OGBench datasets from a HuggingFace mirror.

The official host (rail.eecs.berkeley.edu) is unreachable from this instance,
so `ogbench.make_env_and_datasets(...)`'s built-in downloader times out.
This script fetches the same .npz files from the HF mirror
`zhouzypaul/ogbench_datasets` into ~/.ogbench/data, where ogbench looks for them.

Usage:
    python download_ogbench_data.py antmaze-medium-navigate-v0 scene-play-v0
    python download_ogbench_data.py --list
"""
import argparse, os, shutil, sys

from huggingface_hub import hf_hub_download, list_repo_files

REPO = 'zhouzypaul/ogbench_datasets'
DEST = os.path.expanduser('~/.ogbench/data')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('datasets', nargs='*', help='dataset names, e.g. antmaze-medium-navigate-v0')
    p.add_argument('--list', action='store_true', help='list datasets available on the mirror')
    p.add_argument('--dest', default=DEST)
    args = p.parse_args()

    files = list_repo_files(REPO, repo_type='dataset')
    names = sorted({f[: -len('.npz')].replace('-val', '') for f in files if f.endswith('.npz')})

    if args.list or not args.datasets:
        print('\n'.join(names))
        return

    os.makedirs(args.dest, exist_ok=True)
    for name in args.datasets:
        if name not in names:
            sys.exit(f'{name} not on the mirror; run --list to see what is available')
        for fname in (f'{name}.npz', f'{name}-val.npz'):
            dst = os.path.join(args.dest, fname)
            if os.path.exists(dst):
                print(f'skip (exists): {dst}')
                continue
            src = hf_hub_download(REPO, fname, repo_type='dataset')
            shutil.copyfile(src, dst)
            print(f'{dst}  ({os.path.getsize(dst) / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()
