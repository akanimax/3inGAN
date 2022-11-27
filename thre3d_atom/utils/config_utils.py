import argparse
import traceback
from pathlib import Path
from typing import Any

import yaml


def int_or_none(v):
    if isinstance(v, str):
        return int(v) if v.lower() != "none" else None
    elif v is None:
        return None
    else:
        raise argparse.ArgumentTypeError("int value expected.")


def float_or_none(v):
    if isinstance(v, str):
        return float(v) if v.lower() != "none" else None
    elif v is None:
        return None
    else:
        raise argparse.ArgumentTypeError("float value expected.")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def validity(valid: Any) -> None:
    """ small shortcut for if not valid raise ValueError """
    if not valid:
        line = traceback.extract_stack()[-2].line
        raise ValueError(f"config initialised with wrong value: '{line}'")


def log_args_config_to_disk(
    args: argparse.Namespace, output_dir: Path, config_file_name: str = "config.yml"
) -> None:
    """ writes the arguments config as a yaml file at the given output directory """
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(str(output_dir / config_file_name), "w") as outfile:
        yaml.dump(args, outfile, default_flow_style=False)
