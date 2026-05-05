#!/usr/bin/env python3
"""
Standalone noise ceiling computation for source-space RSA.

Loads per-subject morphed category STCs (produced by compute_source_rsa.py)
and runs the searchlight noise ceiling across subjects.

Run this after all per-subject source-RSA jobs have completed:

    python compute_source_rsa_noise_ceiling.py \\
        --subjects 1 2 3 4 5 \\
        --models resnet50_ecoset_crop \\
        --layers layer3 \\
        --output-dir /share/klab/psulewski/psulewski/pyavs/source_rsa \\
        --n-jobs -1
"""

import argparse
import os
import sys

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from pyavs.utils.logging import get_logger

# Import the noise ceiling function from the per-subject script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rsa_analysis.compute_source_rsa import compute_noise_ceiling_stc

logger = get_logger('scripts.rsa_analysis.compute_source_rsa_noise_ceiling')


def main():
    parser = argparse.ArgumentParser(
        description="Group noise ceiling for source-space RSA"
    )
    parser.add_argument(
        '--subjects', '-s',
        nargs='+', type=int, default=list(range(1, 6)),
        help='Subject IDs (default: 1 2 3 4 5)',
    )
    parser.add_argument(
        '--models',
        nargs='+', type=str, default=['resnet50_ecoset_crop'],
        help='Model names',
    )
    parser.add_argument(
        '--layers',
        nargs='+', type=str, default=['layer3'],
        help='Model layers (must match --models count)',
    )
    parser.add_argument(
        '--subjects-dir',
        type=str, default='/share/klab/datasets/avs/rawdir/',
        help='FreeSurfer subjects directory',
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str, default='/share/klab/psulewski/psulewski/pyavs/source_rsa/',
        help='Output directory (same root used by compute_source_rsa.py)',
    )
    parser.add_argument(
        '--morph-to',
        type=str, default='fsaverage',
        help='Common surface subject (default: fsaverage)',
    )
    parser.add_argument(
        '--n-jobs', '-j',
        type=int, default=-1,
        help='Parallel jobs (default: -1)',
    )
    args = parser.parse_args()

    if len(args.models) != len(args.layers):
        raise ValueError(
            f"Number of models ({len(args.models)}) must match "
            f"number of layers ({len(args.layers)})"
        )

    model_specs = list(zip(args.models, args.layers))

    logger.info("=" * 70)
    logger.info("Source-Space RSA — Group Noise Ceiling")
    logger.info("=" * 70)
    logger.info(f"Subjects:   {args.subjects}")
    logger.info(f"Models:     {model_specs}")
    logger.info(f"Output:     {args.output_dir}")

    for model_name, layer in model_specs:
        compute_noise_ceiling_stc(
            subjects=args.subjects,
            model_name=model_name,
            layer=layer,
            output_dir=args.output_dir,
            subjects_dir=args.subjects_dir,
            morph_to=args.morph_to,
            spatial_radius=0.02,
            n_jobs=args.n_jobs,
            bound='upper',
        )

    logger.info("Done.")


if __name__ == '__main__':
    main()
