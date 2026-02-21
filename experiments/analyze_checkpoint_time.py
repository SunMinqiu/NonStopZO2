#!/usr/bin/env python3
"""
Script to analyze checkpoint times from log files.
Extracts times from lines like:
  [Checkpoint Timing] Differential checkpoint save took 3.844s at step 100
  [Checkpoint Timing] Full checkpoint save took 20.802s at step 1000
"""

import re
import argparse
import statistics
from pathlib import Path


def extract_checkpoint_times(log_file: str) -> list[tuple[int, float, str]]:
    """
    Extract checkpoint step numbers, times, and types from a log file.

    Returns:
        List of (step, time_seconds, checkpoint_type) tuples
    """
    # Pattern to match: [Checkpoint Timing] <type> checkpoint save took <time>s at step <step>
    pattern = r'\[Checkpoint Timing\] (\w+) checkpoint save took ([\d.]+)s at step (\d+)'
    results = []

    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                checkpoint_type = match.group(1)  # e.g., "Differential", "Full"
                time_sec = float(match.group(2))
                step = int(match.group(3))
                results.append((step, time_sec, checkpoint_type))

    return results


def analyze_times(checkpoints: list[tuple[int, float, str]]) -> dict:
    """Calculate statistics for checkpoint times."""
    if not checkpoints:
        return None

    times = [t for _, t, _ in checkpoints]

    return {
        'count': len(times),
        'total': sum(times),
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'min': min(times),
        'max': max(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def analyze_by_type(checkpoints: list[tuple[int, float, str]]) -> dict:
    """Calculate statistics for checkpoint times grouped by type."""
    if not checkpoints:
        return {}

    # Group checkpoints by type
    by_type = {}
    for step, time_sec, checkpoint_type in checkpoints:
        if checkpoint_type not in by_type:
            by_type[checkpoint_type] = []
        by_type[checkpoint_type].append((step, time_sec))

    # Calculate stats for each type
    stats_by_type = {}
    for checkpoint_type, data in by_type.items():
        times = [t for _, t in data]
        stats_by_type[checkpoint_type] = {
            'count': len(times),
            'total': sum(times),
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'stdev': statistics.stdev(times) if len(times) > 1 else 0.0,
        }

    return stats_by_type


def main():
    parser = argparse.ArgumentParser(
        description='Analyze checkpoint times from log files'
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
    parser.add_argument(
        '--by-type',
        action='store_true',
        help='Show statistics grouped by checkpoint type'
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: File not found: {args.log_file}")
        return 1

    checkpoints = extract_checkpoint_times(args.log_file)

    if not checkpoints:
        print("No checkpoint entries found in the log file.")
        return 0

    stats = analyze_times(checkpoints)

    print(f"=" * 50)
    print(f"Checkpoint Time Analysis")
    print(f"Log file: {args.log_file}")
    print(f"=" * 50)
    print(f"Total checkpoints:  {stats['count']}")
    print(f"Average time:       {stats['mean']:.4f}s")
    print(f"Median time:        {stats['median']:.4f}s")
    print(f"Standard deviation: {stats['stdev']:.4f}s")
    print(f"Min time:           {stats['min']:.4f}s")
    print(f"Max time:           {stats['max']:.4f}s")
    print(f"Total time spent:   {stats['total']:.4f}s")
    print(f"=" * 50)

    # Show statistics by checkpoint type
    if args.by_type:
        stats_by_type = analyze_by_type(checkpoints)
        print(f"\nStatistics by Checkpoint Type:")
        print(f"=" * 50)
        for checkpoint_type, type_stats in sorted(stats_by_type.items()):
            print(f"\n{checkpoint_type} Checkpoints:")
            print(f"  Count:              {type_stats['count']}")
            print(f"  Average time:       {type_stats['mean']:.4f}s")
            print(f"  Median time:        {type_stats['median']:.4f}s")
            print(f"  Standard deviation: {type_stats['stdev']:.4f}s")
            print(f"  Min time:           {type_stats['min']:.4f}s")
            print(f"  Max time:           {type_stats['max']:.4f}s")
            print(f"  Total time spent:   {type_stats['total']:.4f}s")
        print(f"=" * 50)

    if args.verbose:
        print("\nIndividual checkpoint times:")
        print("-" * 50)
        for step, time_sec, checkpoint_type in checkpoints:
            print(f"  Step {step:>6} [{checkpoint_type:>15}]: {time_sec:.4f}s")

    return 0


if __name__ == '__main__':
    exit(main())
