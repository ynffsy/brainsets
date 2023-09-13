rule freeze:
    input:
        description = f"{PROCESSED_DIR}/{DATASET}/description.yaml"
    output:
        train_tar = f"{COMPRESSED_DIR}/{DATASET}/train.tar.lz4",
        test_tar = f"{COMPRESSED_DIR}/{DATASET}/test.tar.lz4",
        valid_tar = f"{COMPRESSED_DIR}/{DATASET}/valid.tar.lz4",
        desc_out = f"{COMPRESSED_DIR}/{DATASET}/description.yaml"
    shell:
        f"""
        mkdir -p {COMPRESSED_DIR}/{DATASET}
        for split in train valid test; do
            # Single lz4 archive.
            cd {PROCESSED_DIR}/{DATASET}/$split && \
                tar -cf - . | lz4 -1 > {COMPRESSED_DIR}/{DATASET}/$split.tar.lz4
            cd - > /dev/null
            # Multiple shards for webdataset usage.
            python ../../split_and_tar.py --input_dir "{PROCESSED_DIR}/{DATASET}/$split" --output_dir "{COMPRESSED_DIR}/{DATASET}" --prefix $split
        done
        cp {{input.description}} {COMPRESSED_DIR}/{DATASET}/description.yaml
        """

rule unfreeze:
    input:
        train_tar = f"{COMPRESSED_DIR}/{DATASET}/train.tar.lz4",
        test_tar = f"{COMPRESSED_DIR}/{DATASET}/test.tar.lz4",
        valid_tar = f"{COMPRESSED_DIR}/{DATASET}/valid.tar.lz4",
        desc_in = f"{COMPRESSED_DIR}/{DATASET}/description.yaml"
    output:
        train_dir = directory(f"{PROCESSED_DIR}/{DATASET}/train"),
        test_dir = directory(f"{PROCESSED_DIR}/{DATASET}/test"),
        valid_dir = directory(f"{PROCESSED_DIR}/{DATASET}/valid"),
        desc_out = f"{PROCESSED_DIR}/{DATASET}/description.yaml"
    shell:
        f"""
        for split in valid test train; do
            # Uncompress and untar the data.
            echo $split
            mkdir -p {PROCESSED_DIR}/{DATASET}/$split
            lz4 -d -c {COMPRESSED_DIR}/{DATASET}/$split.tar.lz4 | tar -xf - -C {PROCESSED_DIR}/{DATASET}/$split
        done
        cp {COMPRESSED_DIR}/{DATASET}/description.yaml {PROCESSED_DIR}/{DATASET}/description.yaml
        """