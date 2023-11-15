from snakemake.utils import min_version
min_version("6.0")

local_env = snakemake.shell(f"source ./detect_environment.sh", read=True).strip()
print(f"Using environment {local_env}, as determined by detect_environment.sh")
configfile: f"configs/environments/{local_env}.yaml"

def expand_path(path):
    # Expands environment variables like $VAR
    path = os.path.expandvars(path)

    # Expands the '~' symbol to the user's home directory
    path = os.path.expanduser(path)

    return path

# get various paths from config file
config["TMP_DIR"] = expand_path(f"{config['tmp_dir']}")
config["RAW_DIR"] = expand_path(f"{config['raw_dir']}/raw")
config["PROCESSED_DIR"] = expand_path(f"{config['processed_dir']}/processed")
config["COMPRESSED_DIR"] = expand_path(f"{config['compressed_dir']}/compressed")
config["UNCOMPRESSED_DIR"] = expand_path(f"{config['uncompressed_dir']}/uncompressed")


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
