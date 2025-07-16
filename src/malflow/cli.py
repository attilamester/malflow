import argparse
import os.path

from malflow.core.model import CallGraph, CallGraphCompressed
from malflow.util.logger import Logger


def main():
    args = argparse.ArgumentParser(prog="malflow", description="Malflow CLI for processing PE files with Radare2.")

    args.add_argument("-i", "--input", help="Path to the PE file", required=True)
    args.add_argument("-d", "--dump", action="store_true", help="Dump the compressed call graph (optional, defaults to CLI logs only)", default=None)
    args = args.parse_args()

    if not args.input:
        print("Please provide input file path.")
        return
    try:
        cg = CallGraph(args.input, scan=True, verbose=True)
    except Exception as e:
        Logger.error(f"An error occurred while processing the file: {e}")
        return

    if args.dump:
        compressed = CallGraphCompressed(cg)
        output_path = os.path.dirname(os.path.abspath(args.input))
        compressed.dump_compressed(output_path)
        Logger.info(f"Call graph saved to {output_path}")


if __name__ == "__main__":
    main()
