from snakemake.utils import min_version
min_version("6.0")

configfile: "configs/data.yaml"

# get various paths from config file
TMP_DIR = config["tmp_dir"]
PERM_DIR = config["perm_dir"]
config["RAW_DIR"] = str(Path(TMP_DIR) / "raw") if config["tmp_flag"]["raw"] else str(Path(PERM_DIR) / "raw")
config["PROCESSED_DIR"] = str(Path(TMP_DIR) / "processed") if config["tmp_flag"]["processed"] else str(Path(PERM_DIR) / "processed")
config["COMPRESSED_DIR"] = str(Path(TMP_DIR) / "compressed") if config["tmp_flag"]["compressed"] else str(Path(PERM_DIR) / "compressed")
config["UNCOMPRESSED_DIR"] = str(Path(TMP_DIR) / "uncompressed") if config["tmp_flag"]["uncompressed"] else str(Path(PERM_DIR) / "uncompressed")


# include all snakefiles for all individual datasets
# includes are relative to the directory of the Snakefile in which they occur
module allen_visual_behavior_neuropixels_module:
    snakefile: "data/scripts/allen_visual_behavior_neuropixels/Snakefile"
    config: config
use rule * from allen_visual_behavior_neuropixels_module as allen_visual_behavior_neuropixels_*
use rule all from allen_visual_behavior_neuropixels_module as allen_visual_behavior_neuropixels

module perich_miller_module:
    snakefile: "data/scripts/perich_miller/Snakefile"
    config: config
use rule * from perich_miller_module as perich_miller_*
use rule all from perich_miller_module as perich_miller

module willett_shenoy_module:
    snakefile: "data/scripts/willett_shenoy/Snakefile"
    config: config
use rule * from willett_shenoy_module as willett_shenoy_*
use rule all from willett_shenoy_module as willett_shenoy

module odoherty_sabes_module:
    snakefile: "data/scripts/odoherty_sabes/Snakefile"
    config: config
use rule * from odoherty_sabes_module as odoherty_sabes_*
use rule all from odoherty_sabes_module as odoherty_sabes


# make rules that combine multiple datasets
rule poyo_neurips:
    input:
        perich_miller_module.rules.all.input,
        odoherty_sabes_module.rules.all.input,
