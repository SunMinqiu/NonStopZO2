#!/usr/bin/env python3
"""
Script to analyze ZOTrainer full checkpoint times from log files.
Extracts times from lines like:
  [ZOTrainer] Full checkpoint at step 1000, total took 20.802s
"""

import re
import argparse
import statistics
from pathlib import Path


def extract_checkpoint_times(log_file: str) -> list[tuple[int, float]]:
    """
    Extract checkpoint step numbers and times from a log file.

    Returns:
        List of (step, time_seconds) tuples
    """
    pattern = r'\[ZOTrainer\] Full checkpoint at step (\d+), total took ([\d.]+)s'
    results = []

    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                step = int(match.group(1))
                time_sec = float(match.group(2))
                results.append((step, time_sec))

    return results


def analyze_times(checkpoints: list[tuple[int, float]]) -> dict:
    """Calculate statistics for checkpoint times."""
    if not checkpoints:
        return None

    times = [t for _, t in checkpoints]

    return {
        'count': len(times),
        'total': sum(times),
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'min': min(times),
        'max': max(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze ZOTrainer full checkpoint times from log files'
    )
    parser.add_argument(
        'log_file',
        type=str,
        help='Path to the log file to analyze'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show individual checkpoint times'
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: File not found: {args.log_file}")
        return 1

    checkpoints = extract_checkpoint_times(args.log_file)

    if not checkpoints:
        print("No ZOTrainer full checkpoint entries found in the log file.")
        return 0

    stats = analyze_times(checkpoints)

    print(f"=" * 50)
    print(f"ZOTrainer Full Checkpoint Time Analysis")
    print(f"Log file: {args.log_file}")
    print(f"=" * 50)
    print(f"Total checkpoints:  {stats['count']}")
    print(f"Average time:       {stats['mean']:.3f}s")
    print(f"Median time:        {stats['median']:.3f}s")
    print(f"Standard deviation: {stats['stdev']:.3f}s")
    print(f"Min time:           {stats['min']:.3f}s")
    print(f"Max time:           {stats['max']:.3f}s")

    print(f"Total time spent:   {stats['total']:.3f}s")
    print(f"=" * 50)

    if args.verbose:
        print("\nIndividual checkpoint times:")
        print("-" * 30)
        for step, time_sec in checkpoints:
            print(f"  Step {step:>6}: {time_sec:.3f}s")

    return 0


if __name__ == '__main__':
    exit(main())
