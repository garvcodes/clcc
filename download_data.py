"""
Download raw data from GEO (GSE131907).

This script downloads the lung cancer scRNA-seq dataset from NCBI GEO.
The dataset is from Kim et al. (2020) - "Single-cell RNA sequencing
demonstrates the molecular and cellular reprogramming of metastatic
lung adenocarcinoma"

Files downloaded:
- GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz (~2.5GB compressed, ~12GB uncompressed)
- GSE131907_Lung_Cancer_cell_annotation.txt.gz (~5MB compressed)
- GSE131907_Lung_Cancer_Feature_Summary.xlsx (~20KB)

Usage:
    python download_data.py
    python download_data.py --output-dir data/raw
    python download_data.py --skip-extract  # Keep files compressed
"""

import os
import sys
import gzip
import shutil
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError
import time

# GEO supplementary file URLs
GEO_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131907/suppl"

FILES = {
    "GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz": {
        "url": f"{GEO_BASE_URL}/GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz",
        "size_mb": 2500,  # Approximate compressed size
        "description": "Raw UMI count matrix (genes x cells)",
    },
    "GSE131907_Lung_Cancer_cell_annotation.txt.gz": {
        "url": f"{GEO_BASE_URL}/GSE131907_Lung_Cancer_cell_annotation.txt.gz",
        "size_mb": 5,
        "description": "Cell annotation metadata",
    },
    "GSE131907_Lung_Cancer_Feature_Summary.xlsx": {
        "url": f"{GEO_BASE_URL}/GSE131907_Lung_Cancer_Feature_Summary.xlsx",
        "size_mb": 0.02,
        "description": "Feature summary (Excel)",
    },
}


def print_progress(block_num, block_size, total_size):
    """Print download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Progress: {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
        sys.stdout.flush()


def download_file(url: str, output_path: Path, description: str) -> bool:
    """Download a file with progress indicator."""
    print(f"\nDownloading: {description}")
    print(f"  URL: {url}")
    print(f"  Output: {output_path}")

    try:
        start_time = time.time()
        urlretrieve(url, output_path, reporthook=print_progress)
        elapsed = time.time() - start_time
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n  Complete! ({size_mb:.1f} MB in {elapsed:.1f}s)")
        return True
    except URLError as e:
        print(f"\n  ERROR: Failed to download - {e}")
        return False
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return False


def extract_gzip(gz_path: Path, output_path: Path) -> bool:
    """Extract a gzip file."""
    print(f"  Extracting {gz_path.name}...")
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Extracted: {output_path.name} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ERROR extracting: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download GSE131907 lung cancer scRNA-seq data from GEO"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to save downloaded files (default: data/raw)"
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Keep files compressed (don't extract .gz files)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GSE131907 Lung Cancer scRNA-seq Data Download")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"\nFiles to download:")
    for filename, info in FILES.items():
        print(f"  - {filename} (~{info['size_mb']} MB)")

    print("\n" + "-" * 60)

    # Download each file
    success_count = 0
    for filename, info in FILES.items():
        output_path = output_dir / filename

        # Check if already exists (uncompressed version)
        uncompressed_name = filename.replace('.gz', '')
        uncompressed_path = output_dir / uncompressed_name

        if args.skip_existing:
            if uncompressed_path.exists() and not filename.endswith('.gz'):
                print(f"\nSkipping {filename} (already exists)")
                success_count += 1
                continue
            if uncompressed_path.exists() and filename.endswith('.gz'):
                print(f"\nSkipping {filename} (uncompressed version exists)")
                success_count += 1
                continue
            if output_path.exists():
                print(f"\nSkipping {filename} (already exists)")
                success_count += 1
                continue

        # Download
        if download_file(info['url'], output_path, info['description']):
            success_count += 1

            # Extract if it's a gzip file
            if filename.endswith('.gz') and not args.skip_extract:
                if extract_gzip(output_path, uncompressed_path):
                    # Remove compressed file after extraction
                    output_path.unlink()
                    print(f"  Removed compressed file: {filename}")

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Successfully processed: {success_count}/{len(FILES)} files")

    print(f"\nFiles in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({size_mb:.1f} MB)")

    if success_count == len(FILES):
        print("\nData download complete! You can now run preprocessing:")
        print("  python 01_preprocess_raw_data.py --patient P0006")
    else:
        print("\nSome downloads failed. Please check your internet connection and try again.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
