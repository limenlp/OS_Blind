from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_env(script_path: str, env_name: str, override: bool = False) -> str:
    """
    Load environment variables from exactly one location:
    {script_dir}/envs/{env_name}.env
    """
    script_dir = Path(script_path).resolve().parent
    env_path = script_dir / "envs" / f"{env_name}.env"
    if not env_path.exists():
        raise FileNotFoundError(
            f"Missing env file: {env_path}. "
            f"Expected environment config under script_dir/envs only."
        )

    load_dotenv(env_path, override=override)
    return str(env_path)
