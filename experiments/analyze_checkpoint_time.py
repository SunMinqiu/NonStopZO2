#!/usr/bin/env python3
"""
Script to analyze checkpoint times from log files.
Extracts times from lines like:
  [Checkpoint Timing] Differential checkpoint save took 3.844s at step 100
  [Checkpoint Timing] Full checkpoint save took 20.802s at step 1000
  [ZOTrainer] Checkpoint at step 45 took 0.561s
Also detects resume info:
  [Resume Replay] 100 updates replayed in 7.781s (device=cuda)
  [Full Resume] Total checkpoint resume time: 14.883s
"""

import re
import argparse
import statistics
from pathlib import Path


def extract_checkpoint_times(log_file: str) -> tuple[list[tuple[int, float, str]], list[dict]]:
    """
    Extract checkpoint step numbers, times, and types from a log file.
    Also extract resume information.

    Returns:
        Tuple of:
        - List of (step, time_seconds, checkpoint_type) tuples
        - List of resume info dicts
    """
    # Pattern 1: [Checkpoint Timing] <type> checkpoint save took <time>s at step <step>
    pattern1 = r'\[Checkpoint Timing\] (\w+) checkpoint save took ([\d.]+)s at step (\d+)'
    # Pattern 2: [ZOTrainer] Checkpoint at step <step> took <time>s
    pattern2 = r'\[ZOTrainer\] Checkpoint at step (\d+) took ([\d.]+)s'
    # Resume patterns
    resume_replay_pattern = r'\[Resume Replay\] (\d+) updates replayed in ([\d.]+)s \(device=(\w+)\)'
    full_resume_pattern = r'\[Full Resume\] Total checkpoint resume time: ([\d.]+)s'

    results = []
    resume_info = []

    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(pattern1, line)
            if match:
                checkpoint_type = match.group(1)
                time_sec = float(match.group(2))
                step = int(match.group(3))
                results.append((step, time_sec, checkpoint_type))
                continue

            match = re.search(pattern2, line)
            if match:
                step = int(match.group(1))
                time_sec = float(match.group(2))
                results.append((step, time_sec, 'ZOTrainer'))
                continue

            match = re.search(resume_replay_pattern, line)
            if match:
                resume_info.append({
                    'type': 'Resume Replay',
                    'updates': int(match.group(1)),
                    'time': float(match.group(2)),
                    'device': match.group(3),
                })
                continue

            match = re.search(full_resume_pattern, line)
            if match:
                resume_info.append({
                    'type': 'Full Resume',
                    'time': float(match.group(1)),
                })

    return results, resume_info


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

    checkpoints, resume_info = extract_checkpoint_times(args.log_file)

    if not checkpoints and not resume_info:
        print("No checkpoint or resume entries found in the log file.")
        return 0

    if checkpoints:
        stats = analyze_times(checkpoints)

        print(f"=" * 50)
        print(f"Checkpoint Time Analysis")
        print(f"Log file: {args.log_file}")
        print(f"=" * 50)
        print(f"Total checkpoints:  {stats['count']}")
        print(f"Average time:       {stats['mean']:.4f}s")
        print(f"Median time:        {stats['median']:.4f}s")
        print(f"Standard deviation: {stats['stdev']:.4f}s")
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
                print(f"  Total time spent:   {type_stats['total']:.4f}s")
            print(f"=" * 50)

        if args.verbose:
            print("\nIndividual checkpoint times:")
            print("-" * 50)
            for step, time_sec, checkpoint_type in checkpoints:
                print(f"  Step {step:>6} [{checkpoint_type:>15}]: {time_sec:.4f}s")
    else:
        print("No checkpoint entries found in the log file.")

    # Print resume information
    if resume_info:
        print(f"\n{'=' * 50}")
        print(f"Resume Information")
        print(f"{'=' * 50}")
        for info in resume_info:
            if info['type'] == 'Resume Replay':
                print(f"  [Resume Replay] {info['updates']} updates replayed in {info['time']:.3f}s (device={info['device']})")
            elif info['type'] == 'Full Resume':
                print(f"  [Full Resume] Total checkpoint resume time: {info['time']:.3f}s")
        print(f"{'=' * 50}")

    return 0


if __name__ == '__main__':
    exit(main())
