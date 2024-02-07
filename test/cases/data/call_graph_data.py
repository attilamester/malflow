from dataclasses import dataclass

from cases.data.r2_scanner_data import R2ScannerData


@dataclass
class CallGraphData:
    agCd: str
    agRd: str
    ie: str
    r2_data: R2ScannerData


# libr/core/canal.c: R_API void r_core_anal_callgraph(RCore *core, ut64 addr, int fmt)
CALL_GRAPH_DATA = [
    CallGraphData(**{
        "agCd": """
digraph code {
    "0x1" [label="entry0" URL=""];
    "0x2" [label="main" URL=""];
    "0x1" [label="eip" URL=""];
    "0x3" [label="fcn.3" URL=""];
    "0x4" [label="fcn.4" URL=""];
    "0x5" [label="fcn.5" URL=""];
    "0x6" [label="fcn.6" URL=""];
    "0x7" [label="fcn.7" URL=""];
    "0x8" [label="fcn.8" URL=""];
    "0x9" [label="fcn.9" URL=""];
    "0x1" -> "0x2"; "0x2" -> "0x1";
    "0x2" -> "0x3"; "0x2" -> "0x4"; "0x2" -> "0x5"; "0x2" -> "0x9";
    "0x3" -> "0x6";
    "0x3" -> "0x8";
    "0x6" -> "0x5";
    "0x6" -> "0x7";
    "0x8" -> "0x4";
}
""",
        "agRd": """
digraph code {
}
""",
        "ie": """[Entrypoints]
vaddr=0x1
""",
        "r2_data": R2ScannerData(**{
            "md5": "-",
            "sha256": "-",
            "nodes": {
                "entry0", "main", "fcn.3", "fcn.4", "fcn.5", "fcn.6", "fcn.7", "fcn.8", "fcn.9"
            },
            "links": {
                ("entry0", "main"), ("main", "entry0"), ("main", "fcn.3"), ("main", "fcn.4"), ("main", "fcn.5"),
                ("main", "fcn.9"), ("fcn.3", "fcn.6"), ("fcn.3", "fcn.8"), ("fcn.6", "fcn.7"), ("fcn.8", "fcn.4"),
                ("fcn.6", "fcn.5")
            },
            "functions": {},
            "dfs": ["entry0", "main", "fcn.3", "fcn.6", "fcn.5", "fcn.7", "fcn.8", "fcn.4", "fcn.9"]
        })
    }),
    CallGraphData(**{
        "agCd": """
digraph code {
    "0x1" [label="entry0" URL=""];
    "0x2" [label="entry1" URL=""];
    "0x3" [label="fcn.3" URL=""];
    "0x4" [label="fcn.4" URL=""];
    "0x5" [label="fcn.5" URL=""];
    "0x6" [label="fcn.6" URL=""];
    "0x7" [label="fcn.7" URL=""];
    "0x8" [label="fcn.8" URL=""];
    "0x9" [label="fcn.9" URL=""];
    "0x1" -> "0x3";
    "0x3" -> "0x4"; "0x3" -> "0x5";
    "0x4" -> "0x5"; "0x4" -> "0x6";
    "0x6" -> "0x7";
    "0x2" -> "0x3"; "0x2" -> "0x7"; "0x2" -> "0x8"; "0x2" -> "0x9";
}
""",
        "agRd": """
digraph code {
}
""",
        "ie": """[Entrypoints]
vaddr=0x1
vaddr=0x2
""",
        "r2_data": R2ScannerData(**{
            "md5": "-",
            "sha256": "-",
            "nodes": {
                "entry0", "entry1", "fcn.3", "fcn.4", "fcn.5", "fcn.6", "fcn.7", "fcn.8", "fcn.9"
            },
            "links": {("entry0", "fcn.3"), ("entry1", "fcn.3"), ("fcn.3", "fcn.4"), ("fcn.3", "fcn.5"),
                      ("fcn.4", "fcn.5"), ("fcn.4", "fcn.6"), ("fcn.6", "fcn.7"), ("entry1", "fcn.7"),
                      ("entry1", "fcn.8"), ("entry1", "fcn.9")},
            "functions": {},
            "dfs": ["entry0", "fcn.3", "fcn.4", "fcn.5", "fcn.6", "fcn.7", "entry1", "fcn.8", "fcn.9"]
        })
    })
]
