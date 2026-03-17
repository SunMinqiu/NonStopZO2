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
  [Full Resume] Total time from program start to first step: 25.123s
And queued anchor enqueue info:
  Queued anchor step 852 (lock=0.000s, state_dict=0.001s, copy=0.001s, total_enqueue=0.002s)
And train_runtime:
  'train_runtime': 144.1797
"""

import re
import argparse
import statistics
from pathlib import Path


def extract_checkpoint_times(log_file: str) -> tuple[list[tuple[int, float, str]], list[dict], list[float], float | None]:
    """
    Extract checkpoint step numbers, times, and types from a log file.
    Also extract resume information, queued anchor enqueue times, and train_runtime.

    Returns:
        Tuple of:
        - List of (step, time_seconds, checkpoint_type) tuples
        - List of resume info dicts
        - List of queued anchor total_enqueue times (seconds)
        - train_runtime in seconds (or None)
    """
    # Pattern 1: [Checkpoint Timing] <type> checkpoint save took <time>s at step <step>
    pattern1 = r'\[Checkpoint Timing\] (\w+) checkpoint save took ([\d.]+)s at step (\d+)'
    # Pattern 2: [ZOTrainer] Checkpoint at step <step> took <time>s
    pattern2 = r'\[ZOTrainer\] Checkpoint at step (\d+) took ([\d.]+)s'
    # Resume patterns
    resume_replay_pattern = r'\[Resume Replay\] (\d+) updates replayed in ([\d.]+)s \(device=(\w+)\)'
    full_resume_pattern = r'\[Full Resume\] Total checkpoint resume time: ([\d.]+)s'
    full_resume_start_pattern = r'(?:\[Full Resume\] )?Total time from program start to first step: ([\d.]+)s'
    # Queued anchor pattern
    queued_anchor_pattern = r'Queued anchor step \d+ \(.*?total_enqueue=([\d.]+)s\)'
    # Train runtime pattern: 'train_runtime': 144.1797
    train_runtime_pattern = r"'train_runtime':\s*([\d.]+)"

    results = []
    resume_info = []
    enqueue_times = []
    train_runtime = None

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
                continue

            match = re.search(full_resume_start_pattern, line)
            if match:
                resume_info.append({
                    'type': 'Full Resume Start',
                    'time': float(match.group(1)),
                })
                continue

            match = re.search(queued_anchor_pattern, line)
            if match:
                enqueue_times.append(float(match.group(1)))
                continue

            match = re.search(train_runtime_pattern, line)
            if match:
                train_runtime = float(match.group(1))

    return results, resume_info, enqueue_times, train_runtime


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


def analyze_one_file(log_file: str, verbose: bool = False, by_type: bool = False):
    """Analyze a single log file and print results."""
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"Error: File not found: {log_file}")
        return 1

    checkpoints, resume_info, enqueue_times, train_runtime = extract_checkpoint_times(log_file)

    if not checkpoints and not resume_info and not enqueue_times and train_runtime is None:
        print(f"No checkpoint, resume, or runtime entries found in {log_file}")
        return 0

    print(f"{'=' * 50}")
    print(f"Log file: {log_file}")
    print(f"{'=' * 50}")

    if train_runtime is not None:
        print(f"Train runtime (e2e): {train_runtime:.4f}s")

    if checkpoints:
        stats = analyze_times(checkpoints)
        print(f"Total checkpoints:  {stats['count']}")
        print(f"Average time:       {stats['mean']:.4f}s")
        print(f"Median time:        {stats['median']:.4f}s")
        print(f"Standard deviation: {stats['stdev']:.4f}s")
        print(f"Total time spent:   {stats['total']:.4f}s")

        # Show statistics by checkpoint type
        if by_type:
            stats_by_type = analyze_by_type(checkpoints)
            print(f"\nStatistics by Checkpoint Type:")
            for checkpoint_type, type_stats in sorted(stats_by_type.items()):
                print(f"\n{checkpoint_type} Checkpoints:")
                print(f"  Count:              {type_stats['count']}")
                print(f"  Average time:       {type_stats['mean']:.4f}s")
                print(f"  Median time:        {type_stats['median']:.4f}s")
                print(f"  Standard deviation: {type_stats['stdev']:.4f}s")
                print(f"  Total time spent:   {type_stats['total']:.4f}s")

        if verbose:
            print("\nIndividual checkpoint times:")
            print("-" * 50)
            for step, time_sec, checkpoint_type in checkpoints:
                print(f"  Step {step:>6} [{checkpoint_type:>15}]: {time_sec:.4f}s")

    # Print queued anchor enqueue information
    if enqueue_times:
        print(f"\nQueued Anchor Enqueue:")
        print(f"  Total enqueues:     {len(enqueue_times)}")
        print(f"  Total time:         {sum(enqueue_times):.4f}s")
        print(f"  Average time:       {statistics.mean(enqueue_times):.4f}s")

    # Print resume information
    if resume_info:
        print(f"\nResume Information:")
        for info in resume_info:
            if info['type'] == 'Resume Replay':
                print(f"  [Resume Replay] {info['updates']} updates replayed in {info['time']:.3f}s (device={info['device']})")
            elif info['type'] == 'Full Resume':
                print(f"  [Full Resume] Total checkpoint resume time: {info['time']:.3f}s")
            elif info['type'] == 'Full Resume Start':
                print(f"  [Full Resume] Total time from program start to first step: {info['time']:.3f}s")

    print(f"{'=' * 50}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Analyze checkpoint times from log files'
    )
    parser.add_argument(
        'log_files',
        type=str,
        nargs='+',
        help='Path(s) to the log file(s) to analyze'
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

    ret = 0
    for i, log_file in enumerate(args.log_files):
        if i > 0:
            print()
        result = analyze_one_file(log_file, verbose=args.verbose, by_type=args.by_type)
        if result != 0:
            ret = result

    return ret


if __name__ == '__main__':
    exit(main())
