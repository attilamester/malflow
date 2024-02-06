## Training a BagNet model

* follow the instructions in root `README.md` to install the required packages
    * note: `Radare2` installation is not necessary if you will not scan the binaries

* `src/.env`:

```
BODMAS_DIR_R2_SCANS=<folder where the pickle files are>
```

* sample code for loading a compressed pickle file and calling its image generation method:

```python
from core.data.bodmas import Bodmas
from core.model import CallGraphCompressed
from util import config

if __name__ == "__main__":
    config.load_env()
    compressed_path = CallGraphCompressed.get_compressed_path(Bodmas.get_dir_r2_scans(),
                                                              md5="f880e2b38aa997a3f272f31437bafc28")
    cg = CallGraphCompressed.load(compressed_path).decompress()
    cg.get_image(verbose=True)
```

* **note: `get_image` is under development**
