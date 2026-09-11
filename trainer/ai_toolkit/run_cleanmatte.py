"""QDM queue entry point. Does not import diffusion or legacy keyer processes."""
import json
import sys
from extensions.cleanmatte_training.engine import Trainer

if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as handle:
        config = json.load(handle)
    Trainer(config["config"]["process"][0], config["config"]["name"]).run()
