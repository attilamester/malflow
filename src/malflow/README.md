# Static malware analysis - call graph scanner of PE executable with Radare2

* https://github.com/radareorg/radare2
* related work on [GitHub Pages](https://attilamester.github.io/call-graph/)

## Setup

* install dependencies
```bash
sudo apt-get install graphviz graphviz-dev
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
usage: malflow [-h] -i INPUT

Malflow CLI for processing PE files with Radare2.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the PE file
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
