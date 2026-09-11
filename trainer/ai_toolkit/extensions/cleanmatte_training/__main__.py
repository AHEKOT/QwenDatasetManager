import argparse
import json
from .engine import Trainer

parser = argparse.ArgumentParser(description="Train the independent CleanMatte model")
parser.add_argument("config", help="QDM job JSON config")
args = parser.parse_args()
with open(args.config, encoding="utf-8") as handle:
    config = json.load(handle)
Trainer(config["config"]["process"][0], config["config"]["name"]).run()
