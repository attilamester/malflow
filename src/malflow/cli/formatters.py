"""Output formatting utilities for CLI"""
import json
from typing import Any, Dict, List, Optional

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

from malflow import CallGraph
from malflow.core.model.function import CGNode, FunctionType

console = Console()


def print_error(message: str):
    """Print error message"""
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_success(message: str):
    """Print success message"""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_info(message: str):
    """Print info message"""
    console.print(f"[bold blue]ℹ[/bold blue] {message}")


def print_warning(message: str):
    """Print warning message"""
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def format_callgraph_info(cg) -> None:
    """Display call graph metadata"""
    table = Table(title="Call Graph Information", title_justify="left")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("MD5", cg.md5)
    table.add_row("File Path", cg.file_path or "N/A")
    table.add_row("Nodes", str(len(cg.nodes)))
    table.add_row("Entrypoints", str(len(cg.entrypoints)))
    table.add_row("Edges", str(len(cg.get_edges())))
    table.add_row("Scan Time", f"{cg.scan_time:.2f}s" if cg.scan_time else "N/A")

    console.print(table)


def format_nodes(nodes: List[CGNode], limit: Optional[int] = None) -> None:
    """Display nodes in a table"""
    table = Table(title=f"Nodes ({len(nodes)} total)")
    table.add_column("Label", style="cyan", no_wrap=True)
    table.add_column("RVA", style="magenta")
    table.add_column("Type", style="green")
    table.add_column("Instructions", justify="right", style="yellow")
    table.add_column("Calls", justify="right", style="blue")

    display_nodes = nodes[:limit] if limit else nodes

    for node in display_nodes:
        table.add_row(
            node.label,
            node.rva.value if node.rva else "N/A",
            node.type.value,
            str(len(node.instructions)),
            str(len(node.calls))
        )

    if limit and len(nodes) > limit:
        console.print(f"[dim]Showing {limit} of {len(nodes)} nodes[/dim]")

    console.print(table)


def format_edges(edges: List[tuple], limit: Optional[int] = None) -> None:
    """Display edges in a table"""
    table = Table(title=f"Edges ({len(edges)} total)")
    table.add_column("From", style="cyan")
    table.add_column("→", style="dim")
    table.add_column("To", style="magenta")

    display_edges = edges[:limit] if limit else edges

    for (node_a, node_b) in display_edges:
        table.add_row(node_a.label, "→", node_b.label)

    if limit and len(edges) > limit:
        console.print(f"[dim]Showing {limit} of {len(edges)} edges[/dim]")

    console.print(table)


def format_entrypoints(entrypoints: List[CGNode]) -> None:
    """Display entrypoints"""
    table = Table(title="Entrypoints", title_justify="left")
    table.add_column("RVA", style="magenta")
    table.add_column("Label", style="cyan")

    for ep in entrypoints:
        table.add_row(ep.rva.value if ep.rva else "N/A", ep.label)

    console.print(table)


def format_imports(cg: CallGraph) -> None:
    """Display imported functions"""
    imports = [node for node in cg.nodes.values() if node.type == FunctionType.DLL]

    table = Table(title="Imported Functions", title_justify="left")
    table.add_column("RVA", style="magenta")
    table.add_column("Label", style="cyan")

    for imp in imports:
        table.add_row(imp.rva.value if imp.rva else "N/A", imp.label)

    console.print(table)


def export_json(data: Dict[str, Any], output_path: str) -> None:
    """Export data to JSON"""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print_success(f"Exported to {output_path}")


def export_yaml(data: Dict[str, Any], output_path: str) -> None:
    """Export data to YAML"""
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print_success(f"Exported to {output_path}")


def print_json(data: Dict[str, Any]) -> None:
    """Pretty print JSON data with syntax highlighting"""
    json_str = json.dumps(data, indent=2)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
    console.print(syntax)


def create_progress() -> Progress:
    """Create a progress bar for long operations"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    )


def format_node_detail(node: CGNode) -> None:
    """Display detailed information about a node"""
    panel_content = f"""
[cyan]Label:[/cyan] {node.label}
[cyan]RVA:[/cyan] {node.rva.value if node.rva else 'N/A'}
[cyan]Type:[/cyan] {node.type.value}
[cyan]Instructions:[/cyan] {len(node.instructions)}
[cyan]Calls:[/cyan] {len(node.calls)}
"""

    if node.calls:
        panel_content += "\n[cyan]Calls to:[/cyan]\n"
        for call in list(node.calls.values())[:10]:  # Show first 10
            panel_content += f"  • {call.label}\n"
        if len(node.calls) > 10:
            panel_content += f"  ... and {len(node.calls) - 10} more\n"

    console.print(Panel(panel_content, title=f"Node: {node.label}", border_style="blue"))
