import json
from dataclasses import dataclass, field
from typing import Dict

from cases.data.r2_scanner_data import R2ScannerData
from malflow.core.model.instruction import InstructionReferenceType


@dataclass
class CallGraphData:
    agCd: str
    agRd: str
    ies: str
    r2_data: R2ScannerData
    s_pdfj: Dict = field(default_factory=dict)


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
    "0x3" -> "0x6"; "0x3" -> "0x8";
    "0x6" -> "0x5"; "0x6" -> "0x7";
    "0x8" -> "0x4";
}
""",
        "agRd": """
digraph code {
}
""",
        "ies": """
0x1 entry0
""",
        "s_pdfj": {
            "s 0x1 ; pdfj": json.dumps({"ops": [
                {"disasm": "mov eax, 1", "bytes": "", "type": "", "refs": []},
                {"disasm": "xor eax, eax", "bytes": "", "type": "", "refs": []},
                {"disasm": "call 0x2", "bytes": "", "type": "call", "refs": [{
                    "addr": 2, "type": InstructionReferenceType.CALL.value
                }]},
                {"disasm": "pop eax", "bytes": "", "type": "", "refs": []}
            ]}),
            "s 0x2 ; pdfj": json.dumps({"ops": [
                {"disasm": "add eax, 2", "bytes": "", "type": "", "refs": []},
                {"disasm": "call 0x1", "bytes": "", "type": "call", "refs": [{
                    "addr": 1, "type": InstructionReferenceType.CALL.value
                }]},
                {"disasm": "call 0x3", "bytes": "", "type": "call", "refs": [{
                    "addr": 3, "type": InstructionReferenceType.CALL.value
                }]},
                {"disasm": "adc eax, 2", "bytes": "", "type": "", "refs": []},
                {"disasm": "call 0x4", "bytes": "", "type": "call", "refs": [{
                    "addr": 4, "type": InstructionReferenceType.CALL.value
                }]},
                {"disasm": "call 0x5", "bytes": "", "type": "call", "refs": [{
                    "addr": 5, "type": InstructionReferenceType.CALL.value
                }]},
                {"disasm": "call 0x9", "bytes": "", "type": "call", "refs": [{
                    "addr": 9, "type": InstructionReferenceType.CALL.value
                }]},
            ]}),
            "s 0x3 ; pdfj": json.dumps({"ops": [
                {"disasm": "call 0x6", "bytes": "", "type": "call", "refs": [{
                    "addr": 6, "type": InstructionReferenceType.CALL.value
                }]},
                {"disasm": "shr eax", "bytes": "", "type": "", "refs": []},
                {"disasm": "shl eax", "bytes": "", "type": "", "refs": []},
                {"disasm": "call 0x8", "bytes": "", "type": "call", "refs": [{
                    "addr": 8, "type": InstructionReferenceType.CALL.value
                }]},
                {"disasm": "ror eax", "bytes": "", "type": "", "refs": []},
            ]}),
            "s 0x4 ; pdfj": json.dumps({"ops": [
                {"disasm": "ret", "bytes": "", "type": "", "refs": []}
            ]}),
            "s 0x5 ; pdfj": json.dumps({"ops": [
                {"disasm": "ret", "bytes": "", "type": "", "refs": []}
            ]}),
            "s 0x9 ; pdfj": json.dumps({"ops": [
                {"disasm": "popf", "bytes": "", "type": "", "refs": []}
            ]}),
            "s 0x6 ; pdfj": json.dumps({"ops": [
                {"disasm": "pushf", "bytes": "", "type": "", "refs": []},
                {"disasm": "call 0x7", "bytes": "", "type": "call", "refs": [{
                    "addr": 7, "type": InstructionReferenceType.CALL.value
                }]},
                {"disasm": "call 0x5", "bytes": "", "type": "call", "refs": [{
                    "addr": 5, "type": InstructionReferenceType.CALL.value
                }]},
            ]}),
            "s 0x7 ; pdfj": json.dumps({"ops": [
                {"disasm": "div eax, 2", "bytes": "", "type": "", "refs": []},
            ]}),
            "s 0x8 ; pdfj": json.dumps({"ops": [
                {"disasm": "imul eax, 2", "bytes": "", "type": "", "refs": []},
                {"disasm": "call 0x4", "bytes": "", "type": "call", "refs": [{
                    "addr": 4, "type": InstructionReferenceType.CALL.value
                }]},
                {"disasm": "idiv eax, 2", "bytes": "", "type": "", "refs": []},
            ]}),
        },
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
            "dfs": ["entry0", "main", "fcn.3", "fcn.6", "fcn.5", "fcn.7", "fcn.8", "fcn.4", "fcn.9"],
            "dfs_instructions": [
                ({}, [
                    # @formatter:off
                    # 0x1
                    "mov", "xor",
                        # 0x2
                        "add",
                            # 0x1
                            # 0x3
                                # 0x6
                                "pushf",
                                    # 0x7
                                    "div",
                                    # 0x5
                                    "ret",
                            "shr", "shl",
                                # 0x8
                                "imul",
                                    # 0x4
                                    "ret",
                                "idiv",
                            "ror",
                        "adc",
                            # 0x4
                            # 0x5
                            # 0x9
                            "popf",
                    "pop"
                    # @formatter:on
                ]),
                (dict(store_call=True), [
                    # @formatter:off
                    # 0x1
                    "mov", "xor", "call",
                        # 0x2
                        "add", "call",
                            # 0x1
                        "call",
                            # 0x3
                            "call",
                                # 0x6
                                "pushf", "call",
                                    # 0x7
                                    "div",
                                "call",
                                    # 0x5
                                    "ret",
                            "shr", "shl", "call",
                                # 0x8
                                "imul", "call",
                                    # 0x4
                                    "ret",
                                "idiv",
                            "ror",
                        "adc", "call",
                            # 0x4
                        "call",
                            # 0x5
                        "call",
                            # 0x9
                            "popf",
                    "pop"
                    # @formatter:on
                ]),
                (dict(allow_multiple_visits=True), [
                    "mov", "xor", "add", "mov", "xor", "pop", "pushf", "div", "ret", "shr", "shl", "imul", "ret",
                    "idiv", "ror", "adc", "ret", "ret", "popf", "pop"
                ]),
                (dict(allow_multiple_visits=True, store_call=True), [
                    "mov", "xor", "call", "add", "call", "mov", "xor", "call", "pop", "call", "call", "pushf", "call",
                    "div", "call", "ret", "shr", "shl", "call", "imul", "call", "ret", "idiv", "ror", "adc", "call",
                    "ret", "call", "ret", "call", "popf", "pop"
                ])
            ]
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
        "ies": """
0x1  entry0
0x2  entry1
""",
        "s_pdfj": {
            "s 0x1 ; pdfj": json.dumps({"ops": [
                {"disasm": "call 0x3", "bytes": "", "type": "call", "refs": [{
                    "addr": 3, "type": InstructionReferenceType.CALL.value
                }]},
                {"disasm": "mov eax, 1", "bytes": "", "type": "", "refs": []},
                {"disasm": "xor eax, eax", "bytes": "", "type": "", "refs": []},
            ]}),
            "s 0x2 ; pdfj": json.dumps({"ops": [
                {"disasm": "aaa", "bytes": "", "type": "", "refs": []},
                {"disasm": "call 0x3", "bytes": "", "type": "call", "refs": [{
                    "addr": 3, "type": InstructionReferenceType.CALL.value}]},
                {"disasm": "call 0x8", "bytes": "", "type": "call", "refs": [{
                    "addr": 8, "type": InstructionReferenceType.CALL.value}]},
                {"disasm": "aad", "bytes": "", "type": "", "refs": []},
                {"disasm": "call 0x7", "bytes": "", "type": "call", "refs": [{
                    "addr": 7, "type": InstructionReferenceType.CALL.value}]},
                {"disasm": "call 0x9", "bytes": "", "type": "call", "refs": [{
                    "addr": 9, "type": InstructionReferenceType.CALL.value}]},
            ]}),
            "s 0x3 ; pdfj": json.dumps({"ops": [
                {"disasm": "call 0x5", "bytes": "", "type": "call", "refs": [{
                    "addr": 5, "type": InstructionReferenceType.CALL.value}]},
                {"disasm": "call 0x4", "bytes": "", "type": "call", "refs": [{
                    "addr": 4, "type": InstructionReferenceType.CALL.value}]},
                {"disasm": "aam", "bytes": "", "type": "", "refs": []},
            ]}),
            "s 0x4 ; pdfj": json.dumps({"ops": [
                {"disasm": "pusha", "bytes": "", "type": "", "refs": []},
                {"disasm": "call 0x5", "bytes": "", "type": "call", "refs": [{
                    "addr": 5, "type": InstructionReferenceType.CALL.value}]},
                {"disasm": "call 0x6", "bytes": "", "type": "call", "refs": [{
                    "addr": 6, "type": InstructionReferenceType.CALL.value}]}
            ]}),
            "s 0x5 ; pdfj": json.dumps({"ops": [
                {"disasm": "cdq", "bytes": "", "type": "", "refs": []}
            ]}),
            "s 0x6 ; pdfj": json.dumps({"ops": [
                {"disasm": "clc", "bytes": "", "type": "", "refs": []},
                {"disasm": "call 0x7", "bytes": "", "type": "call", "refs": [{
                    "addr": 7, "type": InstructionReferenceType.CALL.value}]},
                {"disasm": "cmps", "bytes": "", "type": "", "refs": []}
            ]}),
            "s 0x7 ; pdfj": json.dumps({"ops": [
                {"disasm": "cmc", "bytes": "", "type": "", "refs": []}
            ]}),
            "s 0x8 ; pdfj": json.dumps({"ops": [
                {"disasm": "cwd", "bytes": "", "type": "", "refs": []}
            ]}),
            "s 0x9 ; pdfj": json.dumps({"ops": [
                {"disasm": "cwde", "bytes": "", "type": "", "refs": []}
            ]}),
        },
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
            "dfs": ["entry0", "fcn.3", "fcn.4", "fcn.5", "fcn.6", "fcn.7", "entry1", "fcn.8", "fcn.9"],
            "dfs_instructions": [
                ({}, [
                    "cdq", "pusha", "clc", "cmc", "cmps", "aam", "mov", "xor", "aaa", "cwd", "aad", "cwde"
                ]),
                (dict(store_call=True), [
                    "call", "call", "cdq", "call", "pusha", "call", "call", "clc", "call", "cmc", "cmps", "aam",
                    "mov", "xor", "aaa", "call", "call", "cwd", "aad", "call", "call", "cwde"
                ]),
                (dict(allow_multiple_visits=True), [
                    "cdq", "pusha", "cdq", "clc", "cmc", "cmps", "aam", "mov", "xor", "aaa", "aam", "cwd", "aad", "cmc",
                    "cwde"
                ]),
                (dict(allow_multiple_visits=True, store_call=True), [
                    "call", "call", "cdq", "call", "pusha", "call", "cdq", "call", "clc", "call", "cmc", "cmps", "aam",
                    "mov", "xor", "aaa", "call", "call", "call", "aam", "call", "cwd", "aad", "call", "cmc", "call",
                    "cwde"
                ])
            ]
        })
    })
]
