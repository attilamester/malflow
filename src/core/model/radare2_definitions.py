"""
Based on Radare2 5.8.8 implementation.
"""
from typing import Set


def is_symbol_flag(name: str) -> bool:
    """
    static bool is_symbol_flag(const char *name)
    """
    return "imp." in name \
        or "dbg." in name \
        or "sym." in name \
        or name.startswith("entry") \
        or name == "main"


def get_function_types() -> Set[str]:
    """
    R_API const char *r_anal_functiontype_tostring(int type)
    """
    return {"null", "fcn", "loc", "sym", "imp", "int", "root", "unk"} | {"sub", "reloc"}


def get_class_attribute_types() -> Set[str]:
    """
    static const char *attr_type_id(RAnalClassAttrType attr_type)
    """
    return {"method", "vtable", "base"}
