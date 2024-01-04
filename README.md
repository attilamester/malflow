# Static malware analysis -- call graph scanner of PE executable with Radare2

* https://github.com/radareorg/radare2
* summary on [GitHub Pages](https://attilamester.github.io/call-graph/)

## Setup

```bash
sudo apt-get install graphviz graphviz-dev
virtualenv --python="/usr/bin/python3.8" "env"
source env/bin/activa
python3 -m pip install -r requirements.txt
```

## Local development

* to install the pre-commit hook:
```bash
pre-commit install
```

* to run tests:
```bash
pre-commit run
```
