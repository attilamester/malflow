from dataclasses import dataclass

from cases.data.r2_scanner_data import R2ScannerData


@dataclass
class CallGraphData:
    agCd: str
    agRd: str
    ie: str
    r2_data: R2ScannerData


CALL_GRAPH_DATA = [
    CallGraphData(**{
        "agCd": """
digraph code {
rankdir=LR;
outputorder=edgesfirst;
graph [bgcolor=azure fontname="Courier" splines="curved"];
node [penwidth=4 fillcolor=white style=filled fontname="Courier Bold" fontsize=14 shape=box];
edge [arrowhead="normal" style=bold weight=2];

  "0x00400000" [label="entry0" URL=""];
  "0x00401000" [label="main" URL=""];
  "0x00400000" [label="eip" URL=""];
  "0x00401600" [label="fcn.00401600" URL=""];
  "0x00401700" [label="fcn.00401700" URL=""];
  "0x00400000" -> "0x00401000";
  "0x00401000" -> "0x00400000";
  "0x00401000" -> "0x00401600";
  "0x00401000" -> "0x00401700";
}
""",
        "agRd": """
digraph code {
}
""",
        "ie": """[Entrypoints]
vaddr=0x00400000 paddr=0x00400000 haddr=0x00400000 type=program

1 entrypoints
""",
        "r2_data": R2ScannerData(**{
            "md5": "-",
            "sha256": "-",
            "nodes": {
                "entry0", "main", "fcn.00401600", "fcn.00401700"
            },
            "links": {
                ("entry0", "main"), ("main", "entry0"), ("main", "fcn.00401600"), ("main", "fcn.00401700")
            },
            "functions": {}
        })
    })
]
