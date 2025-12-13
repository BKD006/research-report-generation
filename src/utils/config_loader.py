import os
import yaml
import json
from pathlib import Path
from src.logger import GLOBAL_LOGGER as log
from src.exception.custom_exception import ResearchAnalystException

def _project_root() -> Path:
    """Determine absolute path for project root directory"""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | None = None) -> dict:
    try:
        env_path = os.getenv("CONFIG_PATH")

        if config_path is None:
            config_path = env_path or str(
                _project_root() / "src" / "config" / "configuration.yaml"
            )

        path = Path(config_path)
        if not path.is_absolute():
            path = _project_root() / path

        if not path.exists():
            log.error("Configuration file not found", path=str(path))
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        log.info(
            "Configuration loaded successfully",
            path=str(path),
            keys=list(config.keys()) if isinstance(config, dict) else [],
        )

        return config

    except Exception as e:
        log.error("Error loading configuration", error=str(e))
        raise ResearchAnalystException("Failed to load configuration file", e)

# ----------------------------------------------------------------------
# 🔹 Test Run (Standalone)
# ----------------------------------------------------------------------
# if __name__ == "__main__":
#     try:
#         config = load_config()
#         print("Config loaded successfully!")
#         print(json.dumps(config, indent=2))
#         log.info("ConfigLoader test run completed successfully")
#     except ResearchAnalystException as e:
#         log.error("ConfigLoader test run failed", error=str(e))