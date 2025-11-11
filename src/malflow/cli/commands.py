"""Command handlers for malflow CLI"""
import os
from typing import Optional

from malflow.cli.formatters import (console, create_progress, export_json, export_yaml, format_callgraph_info,
                                    format_edges, format_entrypoints, format_imports, format_node_detail, format_nodes,
                                    print_error,
                                    print_info, print_success)
from malflow.core.model import CallGraph, CallGraphCompressed
from malflow.core.model.call_graph_image import CallGraphImage, InstructionEncoderComplete, InstructionEncoderMnemonic
from malflow.core.model.function import FunctionType
from malflow.util.logger import Logger


def load_callgraph(input_path: str, force_rescan: bool = False, verbose: bool = False) -> Optional[CallGraph]:
    """Load call graph from file or compressed cache"""
    if not os.path.exists(input_path):
        print_error(f"File not found: {input_path}")
        return None

    # Check for compressed cache
    input_dir = os.path.dirname(os.path.abspath(input_path))

    cg = None

    # Load compressed file directly
    if input_path.endswith(CallGraphCompressed.COMPRESSED_EXTENSION):
        try:
            print_info(f"Loading compressed call graph...")
            cg_compressed = CallGraphCompressed.load(input_path, verbose=verbose)
            cg = cg_compressed.decompress()
            return cg
        except Exception as e:
            print_error(f"Failed to load compressed file: {e}")
            return None
    else:
        # Suppose the file is an input file, not a dumped compressed file
        if not force_rescan:
            # Try to load from cache if not forcing rescan
            try:
                with open(input_path, "rb") as f:
                    import hashlib
                    md5 = hashlib.md5(f.read()).hexdigest()

                compressed_path = CallGraphCompressed.get_compressed_path(input_dir, md5)
                if os.path.exists(compressed_path):
                    print_info(f"Loading from cache: {compressed_path}")
                    cg_compressed = CallGraphCompressed.load(compressed_path, verbose=verbose)
                    cg = cg_compressed.decompress()
                    return cg
            except Exception as e:
                if verbose:
                    Logger.warning(f"Could not load from cache: {e}")

    # Scan the file
    try:
        with create_progress() as progress:
            task = progress.add_task("Scanning binary...", total=None)
            cg = CallGraph(input_path, scan=True, verbose=verbose)
            progress.update(task, completed=True)
        print_success(f"Scanned successfully (took {cg.scan_time:.2f}s)")
        return cg
    except Exception as e:
        print_error(f"Failed to scan file: {e}")
        return None


def cmd_info(args):
    """Info command - basic analysis with optional dump"""
    cg = load_callgraph(args.input, force_rescan=args.rescan, verbose=args.verbose)
    if not cg:
        return 1

    format_callgraph_info(cg)

    if args.ep:
        console.print()
        format_entrypoints(cg.entrypoints)

    if args.imports:
        console.print()
        format_imports(cg)

    if args.dump:
        output_path = args.output or os.path.dirname(os.path.abspath(args.input))
        compressed = CallGraphCompressed(cg)
        file_path = compressed.dump_compressed(output_path)
        print_success(f"Call graph saved to {file_path}")

    return 0


def cmd_nodes(args):
    """Nodes command - list and query nodes"""
    cg = load_callgraph(args.input, verbose=args.verbose)
    if not cg:
        return 1

    nodes = list(cg.nodes.values())

    # Filter by type
    if args.type:
        try:
            func_type = FunctionType(args.type)
            nodes = [n for n in nodes if n.type == func_type]
        except ValueError:
            print_error(f"Invalid function type: {args.type}")
            print_info(f"Valid types: {', '.join([t.value for t in FunctionType])}")
            return 1

    # Filter by label
    if args.label:
        node = cg.get_node_by_label(args.label)
        if node:
            format_node_detail(node)
            return 0
        else:
            print_error(f"Node not found: {args.label}")
            return 1

    # Filter by RVA
    if args.rva:
        try:
            rva_int = int(args.rva, 16) if args.rva.startswith('0x') else int(args.rva)
            node = cg.get_node_by_rva(rva_int)
            if node:
                format_node_detail(node)
                return 0
            else:
                print_error(f"Node not found at RVA: {args.rva}")
                return 1
        except ValueError:
            print_error(f"Invalid RVA format: {args.rva}")
            return 1

    # Display all nodes
    format_nodes(nodes, limit=args.limit)

    return 0


def cmd_edges(args):
    """Edges command - display call graph edges"""
    cg = load_callgraph(args.input, verbose=args.verbose)
    if not cg:
        return 1

    edges = cg.get_edges()
    format_edges(edges, limit=args.limit)

    return 0


def cmd_export(args):
    """Export command - export to various formats"""
    cg = load_callgraph(args.input, verbose=args.verbose)
    if not cg:
        return 1

    format_type = args.format.lower()
    output_dir_path = args.output or os.path.dirname(os.path.abspath(args.input))

    if format_type == 'json':
        data = CallGraphCompressed(cg).to_dict()

        output_path = os.path.join(output_dir_path, f"{cg.md5}.json")
        export_json(data, output_path)

    elif format_type == 'yaml':
        data = CallGraphCompressed(cg).to_dict()
        output_path = os.path.join(output_dir_path, f"{cg.md5}.yaml")
        export_yaml(data, output_path)

    elif format_type == 'dot':
        cg.dump_dot()
        print_success(f"Exported to {output_dir_path} in DOT format")

    elif format_type == 'svg':
        cg.dump_svg()
        print_success(f"Exported to {output_dir_path} in SVG format")
    else:
        print_error(f"Unknown format: {format_type}")
        print_info("Supported formats: json, yaml, compressed, dot, svg")
        return 1

    return 0


def cmd_image(args):
    """Image command - generate image representation"""
    cg = load_callgraph(args.input, verbose=False)
    if not cg:
        return 1

    img_size = (512, 512)

    # Select encoder
    if args.encoder == 'complete':
        encoder = InstructionEncoderComplete
    elif args.encoder == 'mnemonic':
        encoder = InstructionEncoderMnemonic
    else:
        print_error(f"Unknown encoder: {args.encoder}")
        return 1

    try:
        cg_image = CallGraphImage(cg)

        print_info(f"Generating {img_size[0]}x{img_size[1]} image...")
        np_image, instruction_count = cg_image.get_image(img_size, instruction_encoder=encoder,
                                                         allow_multiple_visits=False, store_call=True)

        output_dir_path = os.path.dirname(os.path.abspath(args.input))

        try:
            from PIL import Image
            img = Image.fromarray(np_image, 'RGB')
            output = os.path.join(output_dir_path, args.output or f"{cg.md5}.png")
            img.save(output)
            print_success(f"Image saved to {output}")
            print_info(f"Instructions encoded: {instruction_count}")
        except ImportError:
            print_error("PIL/Pillow is required to save images. Install with: pip install Pillow")
            return 1

        return 0
    except Exception as e:
        print_error(f"Failed to generate image: {e}")
        return 1


def cmd_dfs(args):
    """DFS command - perform DFS traversal"""
    cg = load_callgraph(args.input, verbose=False)
    if not cg:
        return 1

    print_info("Performing DFS on instructions...")
    instructions = cg.dfs_instructions(
        max_instructions=None,
        allow_multiple_visits=False,
        store_call=True, store_cg_node=False
    )

    console.print(f"[bold]Found {len(instructions)} instructions[/bold]")

    from rich.table import Table
    table = Table(title="Instructions (DFS Order)")
    table.add_column("Index", justify="right", style="cyan")
    table.add_column("Mnemonic", style="green")
    table.add_column("Disassembly", style="white")

    for i, instr in enumerate(instructions):
        table.add_row(str(i), instr.mnemonic, instr.disasm)

    console.print(table)

    return 0
