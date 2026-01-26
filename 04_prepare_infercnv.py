"""
Download and prepare gene position annotations for CNV inference.

This script downloads GENCODE GTF and extracts gene positions (chr, start, end)
which are required for infercnvpy to order genes by chromosomal location.

The gene positions are cached and reused by 05_run_infercnv.py.

Usage:
    python 04_prepare_infercnv.py
"""

import os
import gzip
import argparse
import pandas as pd
from urllib.request import urlretrieve

GENCODE_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz"


def download_gencode_gtf(output_dir: str = "data/reference") -> str:
    """Download GENCODE GTF file."""
    os.makedirs(output_dir, exist_ok=True)
    gz_path = os.path.join(output_dir, "gencode.v44.annotation.gtf.gz")

    if os.path.exists(gz_path):
        print(f"GENCODE GTF already exists: {gz_path}")
        return gz_path

    print(f"Downloading GENCODE v44 GTF (~50MB)...")
    urlretrieve(GENCODE_URL, gz_path)
    print(f"Downloaded to: {gz_path}")
    return gz_path


def get_gene_positions_gencode(
    cache_file: str = "data/reference/gene_positions_hg38.tsv"
) -> pd.DataFrame:
    """
    Get gene positions from GENCODE GTF file.

    Args:
        cache_file: Path to cache the parsed results

    Returns:
        DataFrame with columns: gene, chr, start, end
    """
    # Check cache first
    if os.path.exists(cache_file):
        print(f"Loading cached gene positions from {cache_file}")
        return pd.read_csv(cache_file, sep='\t')

    # Download GTF
    gtf_path = download_gencode_gtf(os.path.dirname(cache_file))

    print(f"Parsing GENCODE GTF for gene positions...")

    gene_positions = []
    valid_chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

    with gzip.open(gtf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue

            # Only keep gene entries
            if fields[2] != 'gene':
                continue

            chrom = fields[0]
            if chrom not in valid_chroms:
                continue

            start = int(fields[3])
            end = int(fields[4])

            # Parse attributes to get gene_name
            attributes = fields[8]
            gene_name = None
            for attr in attributes.split(';'):
                attr = attr.strip()
                if attr.startswith('gene_name'):
                    gene_name = attr.split('"')[1]
                    break

            if gene_name:
                gene_positions.append({
                    'gene': gene_name,
                    'chr': chrom.replace('chr', ''),
                    'start': start,
                    'end': end
                })

    print(f"Parsed {len(gene_positions):,} genes from GTF")

    # Create DataFrame and remove duplicates
    df = pd.DataFrame(gene_positions)
    df = df.drop_duplicates(subset='gene', keep='first')

    # Sort by chromosome and position
    chrom_order = {str(i): i for i in range(1, 23)}
    chrom_order['X'] = 23
    chrom_order['Y'] = 24
    df['chr_num'] = df['chr'].map(chrom_order)
    df = df.sort_values(['chr_num', 'start']).drop('chr_num', axis=1)
    df = df.reset_index(drop=True)

    print(f"After deduplication: {len(df):,} unique genes")

    # Cache results
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    df.to_csv(cache_file, sep='\t', index=False)
    print(f"Cached gene positions to {cache_file}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Download and prepare gene position annotations for CNV inference'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/reference/gene_positions_hg38.tsv',
        help='Output path for gene positions file'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Preparing gene position annotations")
    print("=" * 60)

    df = get_gene_positions_gencode(cache_file=args.output)

    print("\n" + "=" * 60)
    print("Gene positions ready!")
    print("=" * 60)
    print(f"\nTotal genes: {len(df):,}")
    print(f"Chromosomes: {sorted(df['chr'].unique(), key=lambda x: int(x) if x.isdigit() else 23 if x == 'X' else 24)}")
    print(f"\nOutput file: {args.output}")
    print(f"\nNext step: python 05_run_infercnv.py --patient P0006")


if __name__ == "__main__":
    main()
