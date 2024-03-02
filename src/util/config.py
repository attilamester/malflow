import os

from util.misc import get_project_root


def load_env(subpath: str = None):
    try:
        from dotenv import load_dotenv
        if not subpath:
            env_path = os.path.join(get_project_root(), "src", ".env")
        else:
            env_path = os.path.join(get_project_root(), "src", subpath)
        if os.path.isfile(env_path):
            load_dotenv(verbose=True, dotenv_path=env_path)
            print(f"Loaded .env [{env_path}]")
        else:
            print(f"Not existing .env [{env_path}]")
    except Exception as e:
        print(f"Could not load .env file [{e}]")
