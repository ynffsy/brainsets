import argparse

import msgpack
import yaml

from kirby.taxonomy import description_helper


def load_yaml(filename):
    with open(filename, "r") as file:
        return yaml.load(file, Loader=yaml.CLoader)


def save_msgpack(data, filename):
    with open(filename, "wb") as file:
        msgpack.dump(data, file, default=description_helper.encode_datetime)


def main():
    desc = """Convert YAML to MessagePack to 
facilitate migration. Note that rerunning any part of the snakemake pipelines
will trigger a re-encoding anyway; this script is just if you don't want to wait for the
whole pipeline to re-run."""
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("yaml_file", type=str, help="Path to the YAML file")
    args = parser.parse_args()

    yaml_data = load_yaml(args.yaml_file)
    msgpack_file = args.yaml_file.replace(".yaml", ".msgpack") or "output.msgpack"
    save_msgpack(yaml_data, msgpack_file)
    print(f"Converted '{args.yaml_file}' to '{msgpack_file}'")


if __name__ == "__main__":
    main()
