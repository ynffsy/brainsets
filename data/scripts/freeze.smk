COMPRESSED_DIR = config["COMPRESSED_DIR"]
UNCOMPRESSED_DIR = config["UNCOMPRESSED_DIR"]

rule freeze:
    input:
        description = f"{PROCESSED_DIR}/{DATASET}/description.mpk"
    output:
        train_tar = f"{COMPRESSED_DIR}/{DATASET}/train.tar.lz4",
        test_tar = f"{COMPRESSED_DIR}/{DATASET}/test.tar.lz4",
        valid_tar = f"{COMPRESSED_DIR}/{DATASET}/valid.tar.lz4",
        desc_out = f"{COMPRESSED_DIR}/{DATASET}/description.mpk"
    shell:
        f"""
        mkdir -p {COMPRESSED_DIR}/{DATASET}
        for split in valid test train; do
            # Single lz4 archive.
            echo $split
            echo "Compressing"
            cd {PROCESSED_DIR}/{DATASET}/$split && \
                tar -cf - . | lz4 -1 > {COMPRESSED_DIR}/{DATASET}/$split.tar.lz4
            cd - > /dev/null
            echo "Splitting into shards"
            pwd

            # Multiple shards for webdataset usage.
            python split_and_tar.py --input_dir {PROCESSED_DIR}/{DATASET}/$split --output_dir {COMPRESSED_DIR}/{DATASET} --prefix $split
        done
        cp {{input.description}} {COMPRESSED_DIR}/{DATASET}/description.mpk
        """

rule unfreeze:
    input:
        train_tar = f"{COMPRESSED_DIR}/{DATASET}/train.tar.lz4",
        test_tar = f"{COMPRESSED_DIR}/{DATASET}/test.tar.lz4",
        valid_tar = f"{COMPRESSED_DIR}/{DATASET}/valid.tar.lz4",
        desc_in = f"{COMPRESSED_DIR}/{DATASET}/description.mpk"
    output:
        unfreeze_out = f"{UNCOMPRESSED_DIR}/{DATASET}/unfreeze.done"
    shell:
        f"""
        for split in valid test train; do
            # Uncompress and untar the data.
            echo $split
            mkdir -p {UNCOMPRESSED_DIR}/{DATASET}/$split
            lz4 -d -c {COMPRESSED_DIR}/{DATASET}/$split.tar.lz4 | tar -xf - -C {UNCOMPRESSED_DIR}/{DATASET}/$split
        done
        cp {{input.desc_in}} {UNCOMPRESSED_DIR}/{DATASET}/description.mpk
        touch {{output.unfreeze_out}}
        """