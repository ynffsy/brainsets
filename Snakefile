from snakemake.utils import min_version
min_version("6.0")

configfile: "configs/data.yaml"

# get various paths from config file
TMP_DIR = config["tmp_dir"]
PERM_DIR = config["perm_dir"]
config["RAW_DIR"] = str(Path(TMP_DIR) / "raw") if config["tmp_flag"]["raw"] else str(Path(PERM_DIR) / "raw")
config["PROCESSED_DIR"] = str(Path(TMP_DIR) / "processed") if config["tmp_flag"]["processed"] else str(Path(PERM_DIR) / "processed")
config["COMPRESSED_DIR"] = str(Path(TMP_DIR) / "compressed") if config["tmp_flag"]["compressed"] else str(Path(PERM_DIR) / "compressed")

# include all snakefiles for all individual datasets
# includes are relative to the directory of the Snakefile in which they occur
module perich_miller:
    snakefile: "data/scripts/perich_miller/Snakefile"
    config: config

use rule * from perich_miller as perich_miller_*

module odoherty_sabes:
    snakefile: "data/scripts/odoherty_sabes/Snakefile"
    config: config

use rule * from odoherty_sabes as odoherty_sabes_*
