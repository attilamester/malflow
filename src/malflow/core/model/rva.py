class RVA:
    value: str  # e.g. "0x<hex-string>"
    addr: int

    def __init__(self, rva: str):
        self.value = rva.lower()
        if not self.value.startswith("0x"):
            raise Exception(f"Invalid RVA {rva}")
        try:
            self.addr = int(self.value[2:], 16)
        except:
            raise Exception(f"Invalid RVA {rva}")

    def __str__(self):
        return f"RVA('{self.value}')"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, RVA):
            return self.value == other.value and self.addr == other.addr
        return False
