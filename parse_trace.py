#!/usr/bin/env python3
"""
Trace Log Parser - Summarizes PyTorch internal trace logs

Usage:
    python parse_trace.py log.out                    # Full summary
    python parse_trace.py log.out --stats            # Statistics only
    python parse_trace.py log.out --timeline         # Timeline view
    python parse_trace.py log.out --layer Dispatcher # Filter by layer
    python parse_trace.py log.out --condensed        # Condensed view (no duplicates)
"""

import re
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Dict, Optional
import argparse

@dataclass
class TraceEvent:
    """Represents a single trace event"""
    line_num: int
    layer: str  # Dispatcher, Autograd, Memory, BLAS, etc.
    operation: str
    details: Dict[str, str]

    def __str__(self):
        details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
        return f"[{self.layer}] {self.operation} ({details_str})"

class TraceParser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.events: List[TraceEvent] = []
        self.stats = defaultdict(int)

        # Patterns for each layer
        self.patterns = {
            'Dispatcher': re.compile(r'\[TRACE\] ATen Dispatcher.*\n.*Operator: (.+)\n.*Dispatch KeySet: (.+)\n.*Highest priority key: (.+)'),
            'Autograd_Register': re.compile(r'\[TRACE\] Autograd: Registering backward function.*\n.*Backward function: (.+)\n.*Output tensor: shape=(.+), dtype=(.+)'),
            'Autograd_Execute': re.compile(r'\[TRACE\] Autograd: Executing backward function\n.*Function: (.+)'),
            'Autograd_Engine': re.compile(r'\[TRACE\] Autograd Engine: Starting backward pass\n.*Root node: (.+)'),
            'Memory_Alloc': re.compile(r'\[TRACE\] CPU Memory Allocation.*\n.*Allocating: (\d+) bytes \((.+) MB\)\n.*→ Allocated at address: (.+)'),
            'Memory_Free': re.compile(r'\[TRACE\] CPU Memory Deallocation.*\n.*Freeing memory at address: (.+)'),
            'BLAS': re.compile(r'\[TRACE\] CPU BLAS gemm<(.+)>.*\n.*Matrix dimensions: M=(\d+), N=(\d+), K=(\d+)\n.*alpha=(.+), beta=(.+)\n.*TransposeA=(.+), TransposeB=(.+)'),
            'CPP_Impl': re.compile(r'\[TRACE\] C\+\+ scaled_dot_product_attention.*\n.*query: shape=(.+), dtype=(.+)'),
            'Kernel_Dispatch': re.compile(r'\[TRACE\] CPU Kernel Dispatch.*\n.*CPU Capability detected: (.+)\n.*→ Selected: (.+)'),
            'Dynamo': re.compile(r'\[TRACE\] Dynamo: Starting graph capture.*\n.*Function: (.+)\n.*Filename: (.+)\n.*Compile ID: (.+)'),
            'AOTAutograd': re.compile(r'\[TRACE\] AOTAutograd: Generating forward/backward graphs.*\n.*Joint graph nodes: (.+)'),
            'Inductor': re.compile(r'\[TRACE\] Inductor: Starting code generation.*\n.*FX graph nodes: (.+)\n.*Example inputs: (.+)'),
        }

    def parse(self):
        """Parse the trace log file"""
        with open(self.filepath, 'r') as f:
            content = f.read()

        lines = content.split('\n')

        for i, line in enumerate(lines):
            if '[TRACE]' not in line:
                continue

            # Try to match each pattern
            for layer_name, pattern in self.patterns.items():
                # Get context (current line + next few lines)
                context = '\n'.join(lines[i:min(i+10, len(lines))])
                match = pattern.search(context)

                if match:
                    event = self._create_event(i+1, layer_name, match)
                    if event:
                        self.events.append(event)
                        self.stats[layer_name] += 1
                    break

    def _create_event(self, line_num: int, layer: str, match) -> Optional[TraceEvent]:
        """Create a TraceEvent from regex match"""
        try:
            if layer == 'Dispatcher':
                return TraceEvent(line_num, 'Dispatcher', match.group(1), {
                    'keyset': match.group(2),
                    'selected': match.group(3)
                })
            elif layer == 'Autograd_Register':
                return TraceEvent(line_num, 'Autograd', f'Register {match.group(1)}', {
                    'shape': match.group(2),
                    'dtype': match.group(3)
                })
            elif layer == 'Autograd_Execute':
                return TraceEvent(line_num, 'Autograd', f'Execute {match.group(1)}', {})
            elif layer == 'Autograd_Engine':
                return TraceEvent(line_num, 'Autograd', 'Start backward', {
                    'root': match.group(1)
                })
            elif layer == 'Memory_Alloc':
                return TraceEvent(line_num, 'Memory', 'Allocate', {
                    'size': match.group(1),
                    'mb': match.group(2),
                    'addr': match.group(3)
                })
            elif layer == 'Memory_Free':
                return TraceEvent(line_num, 'Memory', 'Free', {
                    'addr': match.group(1)
                })
            elif layer == 'BLAS':
                return TraceEvent(line_num, 'BLAS', f'GEMM<{match.group(1)}>', {
                    'M': match.group(2),
                    'N': match.group(3),
                    'K': match.group(4),
                    'alpha': match.group(5),
                    'beta': match.group(6),
                    'transA': match.group(7),
                    'transB': match.group(8)
                })
            elif layer == 'CPP_Impl':
                return TraceEvent(line_num, 'C++ Impl', 'SDPA', {
                    'shape': match.group(1),
                    'dtype': match.group(2)
                })
            elif layer == 'Kernel_Dispatch':
                return TraceEvent(line_num, 'Kernel Dispatch', match.group(2), {
                    'capability': match.group(1)
                })
            elif layer == 'Dynamo':
                return TraceEvent(line_num, 'Dynamo', f'Capture {match.group(1)}', {
                    'file': match.group(2),
                    'compile_id': match.group(3)
                })
            elif layer == 'AOTAutograd':
                return TraceEvent(line_num, 'AOTAutograd', 'Partition graph', {
                    'nodes': match.group(1)
                })
            elif layer == 'Inductor':
                return TraceEvent(line_num, 'Inductor', 'Generate code', {
                    'nodes': match.group(1),
                    'inputs': match.group(2)
                })
        except Exception as e:
            # print(f"Warning: Failed to parse {layer} at line {line_num}: {e}")
            pass

        return None

    def print_statistics(self):
        """Print summary statistics"""
        print("=" * 80)
        print("TRACE STATISTICS")
        print("=" * 80)

        # Layer counts
        print("\nEvents by Layer:")
        for layer in sorted(self.stats.keys(), key=lambda x: self.stats[x], reverse=True):
            print(f"  {layer:20s}: {self.stats[layer]:4d} events")

        print(f"\n  {'Total':20s}: {len(self.events):4d} events")

        # Operation-specific stats
        print("\nOperation Details:")

        # GEMM operations
        gemm_events = [e for e in self.events if e.layer == 'BLAS']
        if gemm_events:
            print(f"  GEMM calls: {len(gemm_events)}")

            # Count by dimensions
            dim_counter = Counter()
            for e in gemm_events:
                dims = f"M={e.details.get('M')}, N={e.details.get('N')}, K={e.details.get('K')}"
                dim_counter[dims] += 1

            print("  GEMM dimensions (top 5):")
            for dims, count in dim_counter.most_common(5):
                print(f"    {dims}: {count} times")

        # Memory allocations
        alloc_events = [e for e in self.events if e.operation == 'Allocate']
        free_events = [e for e in self.events if e.operation == 'Free']
        if alloc_events:
            total_bytes = sum(int(e.details.get('size', 0)) for e in alloc_events)
            print(f"\n  Memory allocations: {len(alloc_events)}")
            print(f"  Memory frees: {len(free_events)}")
            print(f"  Total allocated: {total_bytes:,} bytes ({total_bytes/1024/1024:.2f} MB)")
            print(f"  Potential leaks: {len(alloc_events) - len(free_events)} allocations")

        # Autograd functions
        autograd_exec = [e for e in self.events if 'Execute' in e.operation]
        if autograd_exec:
            print(f"\n  Backward functions executed: {len(autograd_exec)}")
            func_counter = Counter(e.operation for e in autograd_exec)
            print("  Top backward functions:")
            for func, count in func_counter.most_common(5):
                print(f"    {func}: {count} times")

        # Dispatcher operations
        dispatch_events = [e for e in self.events if e.layer == 'Dispatcher']
        if dispatch_events:
            op_counter = Counter(e.operation for e in dispatch_events)
            print(f"\n  Unique dispatched operators: {len(op_counter)}")
            print("  Top dispatched operations:")
            for op, count in op_counter.most_common(10):
                print(f"    {op}: {count} times")

    def print_timeline(self, max_events: int = 100):
        """Print chronological timeline of events"""
        print("=" * 80)
        print("EXECUTION TIMELINE")
        print("=" * 80)

        if len(self.events) > max_events:
            print(f"\nShowing first {max_events} events (total: {len(self.events)})")
            events_to_show = self.events[:max_events]
        else:
            events_to_show = self.events

        current_phase = None
        for i, event in enumerate(events_to_show, 1):
            # Detect phase changes
            if 'backward' in event.operation.lower() or 'Backward' in event.operation:
                if current_phase != 'BACKWARD':
                    current_phase = 'BACKWARD'
                    print(f"\n{'─' * 80}")
                    print(f"BACKWARD PASS BEGINS")
                    print(f"{'─' * 80}\n")
            elif current_phase is None:
                current_phase = 'FORWARD'
                print(f"\n{'─' * 80}")
                print(f"FORWARD PASS")
                print(f"{'─' * 80}\n")

            # Print event
            print(f"{i:3d}. Line {event.line_num:4d} | {str(event)}")

    def print_condensed(self):
        """Print condensed view - group similar events"""
        print("=" * 80)
        print("CONDENSED TRACE (Grouped by Operation)")
        print("=" * 80)

        # Group by (layer, operation)
        grouped = defaultdict(list)
        for event in self.events:
            key = (event.layer, event.operation)
            grouped[key].append(event)

        print("\nFORWARD PASS:")
        print("-" * 80)

        # Find where backward starts
        backward_start = next((i for i, e in enumerate(self.events)
                              if 'backward' in e.operation.lower()), len(self.events))

        forward_events = self.events[:backward_start]
        backward_events = self.events[backward_start:]

        self._print_grouped_events(forward_events)

        if backward_events:
            print("\n\nBACKWARD PASS:")
            print("-" * 80)
            self._print_grouped_events(backward_events)

    def _print_grouped_events(self, events: List[TraceEvent]):
        """Print events grouped by type"""
        grouped = defaultdict(list)
        for event in events:
            key = (event.layer, event.operation)
            grouped[key].append(event)

        for (layer, operation), event_list in sorted(grouped.items()):
            count = len(event_list)
            if count == 1:
                print(f"\n[{layer}] {operation}")
                for k, v in event_list[0].details.items():
                    print(f"  {k}: {v}")
            else:
                print(f"\n[{layer}] {operation} (×{count})")
                # Show unique variations
                if event_list[0].details:
                    unique_details = defaultdict(set)
                    for e in event_list:
                        for k, v in e.details.items():
                            unique_details[k].add(v)

                    for k, values in unique_details.items():
                        if len(values) == 1:
                            print(f"  {k}: {list(values)[0]}")
                        elif len(values) <= 5:
                            print(f"  {k}: {', '.join(sorted(values))}")
                        else:
                            print(f"  {k}: {len(values)} unique values")

    def filter_by_layer(self, layer: str):
        """Filter events by layer"""
        filtered = [e for e in self.events if layer.lower() in e.layer.lower()]
        print(f"=" * 80)
        print(f"FILTERED: {layer} layer ({len(filtered)} events)")
        print(f"=" * 80)

        for i, event in enumerate(filtered, 1):
            print(f"{i:3d}. Line {event.line_num:4d} | {str(event)}")

def main():
    parser = argparse.ArgumentParser(
        description='Parse and summarize PyTorch trace logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python parse_trace.py log.out                    # Full summary
  python parse_trace.py log.out --stats            # Statistics only
  python parse_trace.py log.out --condensed        # Condensed view
  python parse_trace.py log.out --timeline         # Timeline view
  python parse_trace.py log.out --layer BLAS       # Filter BLAS events
  python parse_trace.py log.out --timeline --max 50  # First 50 events
        """
    )

    parser.add_argument('logfile', help='Path to trace log file (e.g., log.out)')
    parser.add_argument('--stats', action='store_true', help='Show statistics only')
    parser.add_argument('--timeline', action='store_true', help='Show timeline view')
    parser.add_argument('--condensed', action='store_true', help='Show condensed view (group duplicates)')
    parser.add_argument('--layer', type=str, help='Filter by layer (Dispatcher, BLAS, Memory, etc.)')
    parser.add_argument('--max', type=int, default=100, help='Max events to show in timeline (default: 100)')

    args = parser.parse_args()

    # Parse the log
    print(f"Parsing {args.logfile}...")
    trace_parser = TraceParser(args.logfile)
    trace_parser.parse()
    print(f"Found {len(trace_parser.events)} trace events\n")

    # Show requested view
    if args.stats:
        trace_parser.print_statistics()
    elif args.timeline:
        trace_parser.print_timeline(max_events=args.max)
    elif args.condensed:
        trace_parser.print_condensed()
    elif args.layer:
        trace_parser.filter_by_layer(args.layer)
    else:
        # Default: show everything
        trace_parser.print_statistics()
        print("\n")
        trace_parser.print_condensed()
        print("\n")
        print("TIP: Use --timeline, --stats, --condensed, or --layer for specific views")
        print("     Run with --help for all options")

if __name__ == '__main__':
    main()
