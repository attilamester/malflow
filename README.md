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


## Local development

* to install the pre-commit hook:
```bash
pre-commit install
```

* to run tests:
```bash
pre-commit run
```
