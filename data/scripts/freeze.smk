COMPRESSED_DIR = config["COMPRESSED_DIR"]
UNCOMPRESSED_DIR = config["UNCOMPRESSED_DIR"]

rule freeze:
    input:
        description = f"{PROCESSED_DIR}/{DATASET}/description.mpk"
    output:
        data_tar = f"{COMPRESSED_DIR}/{DATASET}/dataset.tar.lz4",
        desc_out = f"{COMPRESSED_DIR}/{DATASET}/description.mpk"
    shell:
        f"""
        mkdir -p {COMPRESSED_DIR}/{DATASET}

        # Single lz4 archive.
        echo "Compressing"
        cd {PROCESSED_DIR}/{DATASET} && \
            tar -cf - . | lz4 -1 > {COMPRESSED_DIR}/{DATASET}/dataset.tar.lz4
        cd - > /dev/null
        echo "Splitting into shards"
        pwd

            # Multiple shards for webdataset usage.
            # python split_and_tar.py --input_dir {PROCESSED_DIR}/{DATASET}/ --output_dir {COMPRESSED_DIR}/{DATASET}
        done
        cp {{input.description}} {COMPRESSED_DIR}/{DATASET}/description.mpk
        """

rule unfreeze:
    input:
        data_tar = f"{COMPRESSED_DIR}/{DATASET}/dataset.tar.lz4",
        desc_in = f"{COMPRESSED_DIR}/{DATASET}/description.mpk"
    output:
        unfreeze_out = f"{UNCOMPRESSED_DIR}/{DATASET}/unfreeze.done"
    shell:
        f"""
        # Uncompress and untar the data.
        mkdir -p {UNCOMPRESSED_DIR}/{DATASET}/
        lz4 -d -c {COMPRESSED_DIR}/{DATASET}/dataset.tar.lz4 | tar -xf - -C {UNCOMPRESSED_DIR}/{DATASET}/
        done
        cp {{input.desc_in}} {UNCOMPRESSED_DIR}/{DATASET}/description.mpk
        touch {{output.unfreeze_out}}
        """