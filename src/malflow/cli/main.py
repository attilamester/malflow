"""Malflow CLI - call graph analysis tool"""
import argparse
import sys

from malflow import __version__
from malflow.cli.commands import (cmd_dfs, cmd_edges, cmd_export, cmd_image, cmd_info, cmd_nodes)


def create_parser():
    parser = argparse.ArgumentParser(
        prog="malflow",
        description="Malflow - Static analysis tool for PE executables using Radare2"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"malflow {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    info_parser = subparsers.add_parser(
        "info",
        help="Analyze PE file and display call graph metadata"
    )
    info_parser.add_argument("-i", "--input", required=True, help="Path to PE file")
    info_parser.add_argument("-d", "--dump", action="store_true", help="Dump compressed call graph")
    info_parser.add_argument("-o", "--output", help="Output directory for dump")
    info_parser.add_argument("--rescan", action="store_true", help="Force rescan even if cache exists")

    info_parser.add_argument("--ep", action="store_true", help="Show entrypoints")
    info_parser.add_argument("--imports", action="store_true", help="Show imports")

    info_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Nodes command
    nodes_parser = subparsers.add_parser(
        "nodes",
        help="List and query nodes"
    )
    nodes_parser.add_argument("-i", "--input", required=True, help="Path to PE file or compressed call graph")
    nodes_parser.add_argument("--type", choices=["subroutine", "dll", "static_linked_lib"], help="Filter by node type")
    nodes_parser.add_argument("--label", help="Get node by label")
    nodes_parser.add_argument("--rva", help="Get node by RVA address")
    nodes_parser.add_argument("--limit", type=int, help="Limit number of results")
    nodes_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Edges command
    edges_parser = subparsers.add_parser(
        "edges",
        help="Display call graph edges"
    )
    edges_parser.add_argument("-i", "--input", required=True, help="Path to PE file or compressed call graph")
    edges_parser.add_argument("--limit", type=int, help="Limit number of results")
    edges_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export call graph to various formats"
    )
    export_parser.add_argument("-i", "--input", required=True, help="Path to PE file or compressed call graph")
    export_parser.add_argument("-o", "--output", help="Output file path")
    export_parser.add_argument(
        "-f", "--format",
        choices=["json", "yaml", "compressed", "dot"],
        default="json",
        help="Export format"
    )
    export_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Image command
    image_parser = subparsers.add_parser(
        "image",
        help="Generate image representation of call graph"
    )
    image_parser.add_argument("-i", "--input", required=True, help="Path to PE file or compressed call graph")
    image_parser.add_argument("-o", "--output", help="Output image file path (requires Pillow)")
    image_parser.add_argument("--size", default="512x512", help="Image size (e.g., 512x512)")
    image_parser.add_argument(
        "--encoder",
        choices=["complete", "mnemonic"],
        default="complete",
        help="Instruction encoder type"
    )

    # DFS command
    dfs_parser = subparsers.add_parser("dfs", help="Perform DFS traversal")
    dfs_parser.add_argument("-i", "--input", required=True, help="Path to PE file or compressed call graph")

    return parser


def main():
    """Main entry point for malflow CLI"""
    parser = create_parser()

    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()

    # Handle commands
    if args.command == "info":
        return cmd_info(args)
    elif args.command == "nodes":
        return cmd_nodes(args)
    elif args.command == "edges":
        return cmd_edges(args)
    elif args.command == "export":
        return cmd_export(args)
    elif args.command == "image":
        return cmd_image(args)
    elif args.command == "dfs":
        return cmd_dfs(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
