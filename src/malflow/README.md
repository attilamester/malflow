# Static malware analysis - call graph scanner of PE executable with Radare2

* https://github.com/radareorg/radare2
* related work on [GitHub Pages](https://attilamester.github.io/call-graph/)

## Features

- 🔍 **Call Graph Extraction**: Extract complete call graphs from PE executables using Radare2
- 📊 **Multiple Export Formats**: JSON, YAML, compressed pickle, and image representations
- 🎨 **Visual Representations**: Generate call graph images with instruction encoding
- 🔎 **Advanced Querying**: Search and filter nodes by type, label, or RVA
- 📈 **DFS Traversal**: Depth-first search on nodes or instruction level

## Setup

* install dependencies
```bash
sudo apt-get install graphviz graphviz-dev libgraphviz-dev
virtualenv --python="/usr/bin/python3.8" "env"
source env/bin/activate
python3 -m pip install -r requirements.txt
```

* install Radare2
  * using release 5.8.8
  * https://github.com/radareorg/radare2/releases/tag/5.8.8


## Example usage

```bash
$ malflow --help
usage: malflow [-h] [--version] {info,nodes,edges,export,image,dfs} ...

Malflow - Static analysis tool for PE executables using Radare2

positional arguments:
  {info,nodes,edges,export,image,dfs}
                        Available commands
    info                Analyze PE file and display call graph metadata
    nodes               List and query nodes
    edges               Display call graph edges
    export              Export call graph to various formats
    image               Generate image representation of call graph
    dfs                 Perform DFS traversal

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
```

```bash
$ malflow -i /MBAZAAR/c7cb8e453252a441522c91d7fccc5065387ce2e1e0e78950f54619eda435b11d.mbazaar
2025-07-16 14:29:37,401 [INFO    ] Scanning /MBAZAAR/c7cb8e453252a441522c91d7fccc5065387ce2e1e0e78950f54619eda435b11d.mbazaar
2025-07-16 14:29:37,579 [INFO    ] [Entry] CGNode(entry0, RVA('0x00401000'), FunctionType.SUBROUTINE)
2025-07-16 14:29:37,588 [INFO    ] [Node - agC] CGNode(entry0, RVA('0x00401000'), FunctionType.SUBROUTINE)
2025-07-16 14:29:37,588 [INFO    ] [Node - agC] CGNode(sub.MSVCRT.dll_memset, RVA('0x00406010'), FunctionType.SUBROUTINE)
2025-07-16 14:29:37,588 [INFO    ] [Node - agC] CGNode(main, RVA('0x00406016'), FunctionType.SUBROUTINE)
...
2025-07-16 14:29:37,589 [INFO    ] [Call - agC] CGNode(entry0, RVA('0x00401000'), FunctionType.SUBROUTINE) -> CGNode(sub.MSVCRT.dll_memset, RVA('0x00406010'), FunctionType.SUBROUTINE)
2025-07-16 14:29:37,589 [INFO    ] [Call - agC] CGNode(entry0, RVA('0x00401000'), FunctionType.SUBROUTINE) -> CGNode(main, RVA('0x00406016'), FunctionType.SUBROUTINE)
2025-07-16 14:29:37,589 [INFO    ] [Call - agC] CGNode(entry0, RVA('0x00401000'), FunctionType.SUBROUTINE) -> CGNode(sub.KERNEL32.dll_HeapCreate, RVA('0x0040601c'), FunctionType.SUBROUTINE)
...
```

## Quickstart

```bash
# Show help
malflow --help

# Analyze a PE file
malflow analyze -i malware.exe

# Get call graph info
malflow info -i malware.exe
```

## Commands

### `info` - Analyze PE File

Scan a PE file and extract call graph information.

```bash
# Basic analysis
malflow info -i malware.exe
# Save compressed call graph
malflow info -i malware.exe --dump

# Specify output directory
malflow info -i malware.exe --dump -o ./output

# Force rescan (ignore cache)
malflow info -i malware.exe --rescan

# Verbose output
malflow info -i malware.exe -v
```

**Options:**
- `-i, --input` - Path to PE file (required)
- `-d, --dump` - Save compressed call graph
- `-o, --output` - Output directory for dump
- `--rescan` - Force rescan even if cache exists
- `--ep` - Show entrypoints
- `--imports` - Show imported functions
- `-v, --verbose` - Verbose output

### `nodes` - List and Query Nodes

List all nodes or query specific nodes by label, RVA, or type.

```bash
# List all nodes
malflow nodes -i malware.exe

# Filter by type
malflow nodes -i malware.exe --type subroutine
malflow nodes -i malware.exe --type dll

# Get specific node by label
malflow nodes -i malware.exe --label entry0

# Get node by RVA
malflow nodes -i malware.exe --rva 0x00401000

# Limit results
malflow nodes -i malware.exe --limit 20
```

**Options:**
- `-i, --input` - Path to PE file or compressed call graph (required)
- `--type` - Filter by type: subroutine, dll, static_linked_lib
- `--label` - Get node by label
- `--rva` - Get node by RVA address
- `--limit` - Limit number of results
- `-v, --verbose` - Verbose output

### `edges` - Display Call Graph Edges

Show all call relationships (caller → callee).

```bash
# Show all edges
malflow edges -i malware.exe

# Limit results
malflow edges -i malware.exe --limit 50
```

**Options:**
- `-i, --input` - Path to PE file or compressed call graph (required)
- `--limit` - Limit number of results
- `-v, --verbose` - Verbose output

### `export` - Export to Various Formats

Export call graph data to JSON, YAML, compressed pickle, or DOT format.

```bash
# Export to JSON (stdout)
malflow export -i malware.exe -f json

# Export to JSON file
malflow export -i malware.exe -f json -o output.json

# Export to YAML
malflow export -i malware.exe -f yaml -o output.yaml

# Export compressed
malflow export -i malware.exe -f compressed -o ./output
```

**Options:**
- `-i, --input` - Path to PE file or compressed call graph (required)
- `-o, --output` - Output file path
- `-f, --format` - Format: json, yaml, compressed, dot (default: json)
- `-v, --verbose` - Verbose output


### `image` - Generate Image Representation

Create visual representation of call graph encoded as an image.

```bash
# Generate image
malflow image -i malware.exe -o callgraph.png
```

**Note:** Requires Pillow

### `dfs` - Perform DFS Traversal

Execute depth-first search traversal on nodes or instructions.

```bash
malflow dfs -i malware.exe
```

## Examples

### Basic Workflow

```bash
# 1. Analyze and cache
malflow info -i malware.exe --dump --ep --imports

# 2. Explore nodes
malflow nodes -i malware.exe --type dll

# 3. Export for further analysis, in JSON / YAML / DOT / SVG
malflow export -i malware.exe -f json

# 4. Generate visualization
malflow image -i malware.exe -o callgraph.png
```

## Performance Tips

1. **Use Caching (-d)**: The first scan creates a compressed cache file. Subsequent commands will use the cache automatically.

2. **Force Rescan**: Only use `--rescan` when the binary has changed or you need fresh analysis.

## Troubleshooting

### Radare2 Not Found
```bash
# Install Radare2 (release 5.8.8 recommended)
# See: https://github.com/radareorg/radare2/releases/tag/5.8.8
```

### Missing Dependencies
```bash
# Install required system packages
sudo apt-get install graphviz graphviz-dev
```

## See Also

- [GitHub Repository](https://github.com/attilamester/malflow)
- [Related Work](https://attilamester.github.io/call-graph/)
- [Radare2 Documentation](https://github.com/radareorg/radare2)
