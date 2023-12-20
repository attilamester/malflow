import os


def load_env():
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
        if os.path.isfile(env_path):
            load_dotenv(verbose=True, dotenv_path=env_path)
            print(f"Loaded .env [{env_path}]")
        else:
            print(f"Not existing .env [{env_path}]")
    except Exception as e:
        print(f"Could not load .env file [{e}]")
