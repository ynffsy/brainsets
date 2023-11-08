from tqdm import tqdm
from pathlib import Path

import yaml
from kirby.data.dataset import Dataset


if __name__ == "__main__":
    with open(Path(__file__).parent.parent / "configs" / "data.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.CFullLoader)

    the_root = (
        config["tmp_dir"] if config["tmp_flag"]["processed"] else config["perm_dir"]
    )
    the_root = Path(the_root) / "processed"

    # Find all the directories under the root
    dirs = [x for x in the_root.iterdir() if x.is_dir()]
    for d in dirs:
        spec = {"selection": {"dandiset": d.name}}
        ds = Dataset(str(the_root), "train", [spec])
        num_tokens = 0
        for data in tqdm(ds):
            num_tokens += len(data.spikes.timestamps)
        print(f"{d.name}: {num_tokens}")
