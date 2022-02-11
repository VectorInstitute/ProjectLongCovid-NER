import argparse
import os
import pathlib
from configparser import ConfigParser

from twitter_data_gen import TwitterDataGen
from clean_passes import remove_unused_fields, hash_ids, filter_roche_mentions, only_ids
import json
import atexit

import yaml

def args_parser():
    parser = argparse.ArgumentParser(description="Prepare full dataset")

    parser.add_argument("-k",
                        "--key-file",
                        type=str,
                        help="Location of the twitter API key file")
    parser.add_argument("-ck",
                        "--credential-key",
                        type=str,
                        help="Credential key used from the key file")
    parser.add_argument("-c",
                        "--config-file",
                        type=str,
                        help="Location of run configuration file")
    parser.add_argument("-o",
                        "--output-file",
                        type=str,
                        default="./output.json",
                        help="Location of output file.")

    parser.add_argument(
        "--exists_ok",
        default=False,
        action='store_true',
        help="OK if output file already exists, simply overwrite")

    parser.add_argument(
        "--no_infer_demographic",
        default=False,
        action='store_true',
        help="do not infer demographic information from user profiles")

    return parser.parse_args()


if __name__ == "__main__":

    args = args_parser()
    key_file = args.key_file
    credential_key = args.credential_key
    config_file = args.config_file
    output_file = args.output_file

    run_config = ConfigParser()
    run_config.read(config_file)


    data_gen = TwitterDataGen(credentials_file=key_file,
                              credentials_key=credential_key,
                              run_config=run_config,
                              output_file=output_file,
                              infer_demographic=not args.no_infer_demographic)

    with open(key_file, "r") as f:
        credentials = yaml.safe_load(f)[credential_key]

    data_gen.register_pass(filter_roche_mentions)
    data_gen.register_pass(remove_unused_fields)
    data_gen.register_pass(hash_ids)

    if pathlib.Path(output_file).exists() and not args.exists_ok:
        raise ValueError(
            "Output file {} already exists, use --exists_ok to overwrite".
            format(args.output_file))


    all_tweets = []

    @atexit.register
    def save_if_not_done_so():
        if not data_gen.output_file.exists() or args.exists_ok:
            if len(all_tweets) == 0:
                return
            with open(data_gen.output_file, "w") as f:
                json.dump(all_tweets, f, indent=4)

            print("Cleaning, saving intermediary json output to {}".format(
                data_gen.output_file))


    for batched_tweets in data_gen:
        all_tweets += batched_tweets


