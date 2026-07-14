from glanos_anonymization.core import (
    GlanosClient,
    AnonymizationConfig,
    PseudoResult,
    FetchKeyData,
    DepseudoPaths,
)
import argparse


def parse_args():
    parser = argparse.ArgumentParser(prog="glanos-anonymization")

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    pseudo_parser = subparsers.add_parser("pseudo", help="pseudo")
    pseudo_parser.add_argument(
        "--file",
        type=str,
        dest="file",
        required=True,
        help="Path of file to be anonymized.",
    )

    pseudo_parser.add_argument(
        "--token",
        type=str,
        dest="token",
        required=True,
        help="Glanos Anonymization API Token",
    )

    pseudo_parser.add_argument(
        "--baseUrl",
        type=str,
        dest="base_url",
        required=True,
        help="Anonymization base url",
    )

    pseudo_parser.add_argument(
        "--lang",
        type=str,
        dest="language",
        required=False,
        help="ISO-639 code of document language",
    )

    pseudo_parser.add_argument(
        "--mode",
        choices=["TYPE", "TYPE_PRESERVING", "XXX", "MT"],
        type=str,
        dest="anonymization_mode",
        default="TYPE",
        help="Pseudonymization/Anonymization Mode: TYPE: COMPANY1, TYPE_PRESERVING: Jane Doe, XXX: XXX (pure "
        "anonymization, no depseudonymizaton possible), MT: as TYPE_PRESERVING but optimized for machine translation",
    )

    pseudo_parser.add_argument(
        "--ssl-no-verify",
        action="store_true",
        dest="no_ssl_verify",
        default=None,
        help="Turn SSL-Verification off",
    )

    pseudo_parser.add_argument(
        "--async",
        dest="a_sync",
        action="store_true",
        default=None,
        help="Whether request should respond immediately with fetchKey for async fetching (default false). Note that the request will continue async if request timed out",
    )

    pseudo_parser.add_argument(
        "--outputFile",
        type=str,
        dest="output_file",
        required=False,
        help="Name of the output file [Required if async is off]",
    )

    pseudo_parser.add_argument(
        "--returnPseudoKey",
        dest="return_pseudo_key",
        action="store_true",
        default=None,
        help="[Only if --async=false] Whether to return the pseudo key (default false), data cannot be depseudonymized without pseudo key after pseudoKey retention time",
    )

    pseudo_parser.add_argument(
        "--returnEncryptKey",
        dest="return_encrypt_key",
        action="store_true",
        default=None,
        help="[Only if async=false] Whether to return the encryption key (default false), data may then be depseudonymized with pseudo and encryption key after meta data retention time",
    )

    pseudo_parser.add_argument(
        "--dataRetentionMS",
        dest="data_retention_ms",
        type=int,
        default=0,
        help="Milliseconds to keep document in the database after processing (default 0), -1 for infinite retention",
    )

    pseudo_parser.add_argument(
        "--pseudoKeyRetentionMS",
        dest="pseudo_key_retention_ms",
        type=int,
        default=0,
        help="Milliseconds to keep pseudoKey in the database after processing (default 0) - after that any depseudo request must contain the returned pseudoKey, -1 for infinite retention",
    )

    pseudo_parser.add_argument(
        "--metaRetentionMS",
        dest="meta_retention_ms",
        type=int,
        default=0,
        help="Milliseconds to keep meta data (including encryptKey) in the database after processing (default -1) - after that any depseudo request must contain the returned pseudoKey and the encryptKey, -1 for infinite retention",
    )

    pseudo_parser.add_argument(
        "--exclude_fields",
        choices=[
            "addresses",
            "anonymizationIds",
            "banks",
            "birthdays",
            "companies",
            "countries",
            "dates",
            "docMeta",
            "filepaths",
            "homepages",
            "idNumbers",
            "images",
            "internalHyperlinks",
            "legals",
            "locations",
            "mails",
            "measures",
            "moneyAmounts",
            "orgs",
            "percents",
            "personNames",
            "phones",
            "products",
            "simpleNumbers",
            "streets",
            "zips",
        ],
        dest="exclude_fields",
        nargs="+",
        default=[],
        help="Entities which should not be anonymized (default: none)",
    )

    fetch_parser = subparsers.add_parser("fetch", help="fetch")
    fetch_parser.add_argument(
        "--token",
        type=str,
        dest="token",
        required=True,
        help="Glanos Anonymization token",
    )

    fetch_parser.add_argument(
        "--fetchKey",
        type=str,
        dest="fetch_key",
        required=True,
        help="FetchKey from Pseudo Request",
    )

    fetch_parser.add_argument(
        "--outputFile",
        type=str,
        dest="output_file",
        required=True,
        help="Name of the output file",
    )

    fetch_parser.add_argument(
        "--baseUrl",
        type=str,
        dest="base_url",
        required=True,
        help="Anonymization base url",
    )

    fetch_parser.add_argument(
        "--ssl-no-verify",
        action="store_true",
        dest="no_ssl_verify",
        default=None,
        help="Turn SSL-Verification off",
    )

    fetch_parser.add_argument(
        "--returnPseudoKey",
        dest="return_pseudo_key",
        action="store_true",
        default=None,
        help="Whether to return the pseudo key (default false), data cannot be depseudonymized without pseudo key after pseudoKey retention time",
    )

    fetch_parser.add_argument(
        "--returnEncryptKey",
        dest="return_encrypt_key",
        action="store_true",
        default=None,
        help="Whether to return the encryption key (default false), data may then be depseudonymized with pseudo and encryption key after meta data retention time",
    )

    depseudo_parser = subparsers.add_parser("depseudo", help="depseudo")
    depseudo_parser.add_argument(
        "--file",
        type=str,
        dest="file",
        required=True,
        help="Path of file to be depseudoanonymized",
    )

    depseudo_parser.add_argument(
        "--token",
        type=str,
        dest="token",
        required=True,
        help="Glanos Anonymization API Token",
    )

    depseudo_parser.add_argument(
        "--baseUrl",
        type=str,
        dest="base_url",
        required=True,
        help="Anonymization base url",
    )

    depseudo_parser.add_argument(
        "--pseudoKeyFile",
        type=str,
        dest="pseudo_key_file",
        required=False,
        default=None,
        help="Optionally provide file with pseudoKey",
    )

    depseudo_parser.add_argument(
        "--encryptKeyFile",
        type=str,
        dest="encrypt_key_file",
        required=False,
        default=None,
        help="Optionally provide encryptKey file",
    )

    depseudo_parser.add_argument(
        "--metaKey",
        type=str,
        dest="meta_key",
        required=False,
        default=None,
        help="metaKey as returned by pseudonymization, required unless pseudoKey and encrytKey is provided",
    )

    depseudo_parser.add_argument(
        "--outputFile",
        type=str,
        dest="output_file",
        required=True,
        help="Name of the output file",
    )

    depseudo_parser.add_argument(
        "--ssl-no-verify",
        action="store_true",
        dest="no_ssl_verify",
        default=None,
        help="Turn SSL-Verification off",
    )

    args = parser.parse_args()

    match args.command:
        case "pseudo":
            if args.a_sync:
                if args.return_pseudo_key or args.return_encrypt_key:
                    print(
                        "Async mode has to be turned off to return pseudo or encryption key! Run the script with the --help option for more info."
                    )
                    exit(1)
            else:
                if not args.output_file:
                    print(
                        "An output filename has to be specified when async mode is off. Run the script with the --help option for more info."
                    )
                    exit(1)
        case "depseudo":
            if not args.meta_key:
                if not args.pseudo_key_file or not args.encrypt_key_file:
                    print(
                        "If no metaKey is provided, then both, --pseudoKeyFile and --encryptKeyFile need to be provided. Run the script with the --help option for more info."
                    )
                    exit(1)

    return args


def create_ano_config(args) -> AnonymizationConfig:
    
    ano_config = AnonymizationConfig(
        **{k: v for k, v in vars(args).items() if v is not None}
    )

    if args.a_sync is None:
        ano_config.a_sync = False
    
    return ano_config


def exec_command(args):
    ano_config = create_ano_config(args)
    glanos_client = GlanosClient(
        token=args.token, base_url=args.base_url, config=ano_config
    )

    match args.command:
        case "pseudo":
            try:
                pseudo_result = glanos_client.pseudo_sync(path=args.file)
            except Exception as e:
                print(f"Error while executing pseudo command: {e}")
                exit(1)
            else:
                if glanos_client.config.a_sync:
                    print(pseudo_result.data.model_dump_json())
                else:
                    glanos_client.manage_data(
                        pseudo_result, output_file=args.output_file
                    )
        case "fetch":
            try:
                pseudo_result = PseudoResult(
                    success=True, data=FetchKeyData(fetchKey=args.fetch_key)
                )
                fetch_result = glanos_client.fetch(pseudo_result)
            except Exception as e:
                print(e)
                exit(1)
            else:
                glanos_client.manage_data(fetch_result, args.output_file)
        case "depseudo":
            try:
                depseudo_paths = DepseudoPaths(
                    pseudo_key_path=args.pseudo_key_file,
                    meta_key_path=args.meta_key,
                    encrypt_key_path=args.encrypt_key_file,
                )
                depseudo_result = glanos_client.depseudo(args.file, depseudo_paths)
            except Exception as e:
                print(e)
                exit(1)
            else:
                glanos_client.manage_data(depseudo_result, output_file=args.output_file)


def main():
    cli_args = parse_args()
    exec_command(cli_args)


if __name__ == "__main__":
    main()
