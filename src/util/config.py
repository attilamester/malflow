import os

from util.misc import get_project_root


def load_env(path: str = None):
    try:
        from dotenv import load_dotenv
        if not path:
            env_path = os.path.join(get_project_root(), "src", ".env")
        else:
            env_path = path
        if os.path.isfile(env_path):
            if path in __envs:
                print(f"Env: already loaded [{env_path}]")
                return
            load_dotenv(verbose=True, dotenv_path=env_path)
            print(f"Env: loaded [{env_path}]")
            __envs[path] = True
        else:
            print(f"Not existing .env [{env_path}]")
    except Exception as e:
        print(f"Could not load .env file [{e}]")

__envs = {}
