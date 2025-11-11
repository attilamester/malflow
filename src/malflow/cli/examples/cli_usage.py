"""
Example Python API usage for Malflow
"""

import os

from malflow.core.model import CallGraph, CallGraphCompressed
from malflow.core.model.call_graph_image import CallGraphImage
from malflow.core.model.function import FunctionType


def basic_analysis(pe_file_path, verbose=False):
    """Basic call graph analysis"""
    print(f"Analyzing {pe_file_path}...")

    # Scan PE file
    cg = CallGraph(pe_file_path, scan=True, verbose=verbose)

    # Basic info
    print(f"\nMD5: {cg.md5}")
    print(f"Nodes: {len(cg.nodes)}")
    print(f"Edges: {len(cg.get_edges())}")
    print(f"Entrypoints: {len(cg.entrypoints)}")
    print(f"Scan time: {cg.scan_time:.2f}s")

    # List entrypoints
    print("\nEntrypoints:")
    for ep in cg.entrypoints:
        print(f"  - {ep.label} at {ep.rva.value}")

    return cg


def explore_nodes(cg):
    """Explore call graph nodes"""
    print("\n=== Node Analysis ===")

    # Filter by type
    dll_nodes = [n for n in cg.nodes.values() if n.type == FunctionType.DLL]
    subroutines = [n for n in cg.nodes.values() if n.type == FunctionType.SUBROUTINE]

    print(f"DLL imports: {len(dll_nodes)}")
    print(f"Subroutines: {len(subroutines)}")

    # Show some DLL imports
    print("\nSample DLL imports:")
    for node in dll_nodes[:10]:
        print(f"  - {node.label}")

    # Get specific node
    if cg.entrypoints:
        entry = cg.entrypoints[0]
        print(f"\nEntry point details: {entry.label}")
        print(f"  RVA: {entry.rva.value}")
        print(f"  Type: {entry.type.value}")
        print(f"  Instructions: {len(entry.instructions)}")
        print(f"  Calls: {len(entry.calls)}")

        # Show what it calls
        print(f"  Calls to:")
        for callee in list(entry.get_calls())[:5]:
            print(f"    -> {callee.label}")


def dfs_traversal(cg):
    """DFS traversal examples"""
    print("\n=== DFS Traversal ===")

    # Node-level DFS
    nodes = cg.dfs()
    print(f"DFS nodes: {len(nodes)}")
    print("First 10 nodes in DFS order:")
    for node in nodes[:10]:
        print(f"  - {node.label}")

    # Instruction-level DFS
    instructions = cg.dfs_instructions(max_instructions=1000)
    print(f"\nDFS instructions (first 1000): {len(instructions)}")
    print("First 100 instructions:")
    for instr in instructions[:100]:
        print(f"  - {instr.mnemonic}: {instr.disasm}")


def save_and_load(cg, output_dir):
    """Save and load compressed call graph"""
    print("\n=== Save/Load ===")

    # Save compressed
    compressed = CallGraphCompressed(cg)
    file_path = compressed.dump_compressed(output_dir)
    print(f"Saved to: {file_path}")

    # Load compressed
    loaded_compressed = CallGraphCompressed.load(file_path, verbose=True)
    loaded_cg = loaded_compressed.decompress()

    # Verify
    print(f"Loaded MD5: {loaded_cg.md5}")
    print(f"Loaded nodes: {len(loaded_cg.nodes)}")
    print(f"Match: {loaded_cg.md5 == cg.md5}")

    return loaded_cg


def generate_image(cg, output_path):
    """Generate call graph image"""
    print("\n=== Image Generation ===")

    cg_image = CallGraphImage(cg)

    # Generate image
    img_size = (512, 512)
    np_image, instr_count = cg_image.get_image(img_size)

    print(f"Generated {img_size[0]}x{img_size[1]} image")
    print(f"Instructions encoded: {instr_count}")

    # Save with PIL (if available)
    try:
        from PIL import Image
        img = Image.fromarray(np_image, 'RGB')
        img.save(output_path)
        print(f"Saved to: {output_path}")
    except ImportError:
        print("PIL not available - install with: pip install Pillow")

    return np_image


def main():
    """Main example"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python api_usage.py <pe_file>")
        print("Example: python api_usage.py malware.exe")
        return

    pe_file = sys.argv[1]

    # Basic analysis
    cg = basic_analysis(pe_file, verbose=False)

    # Explore nodes
    explore_nodes(cg)

    # DFS traversal
    dfs_traversal(cg)

    # Save and load

    output_dir = os.path.dirname(os.path.abspath(pe_file))
    loaded_cg = save_and_load(cg, output_dir)

    # Generate image
    output_image = pe_file + ".png"
    generate_image(cg, output_image)

    # DFS instructions
    print("\n=== DFS Instructions Summary ===")
    print(cg.dfs_instructions_summary_txt(completeness=0))

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
