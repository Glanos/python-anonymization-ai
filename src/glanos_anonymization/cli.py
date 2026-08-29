from glanos_anonymization.core import (
    GlanosClient,
    AnonymizationConfig,
    PseudoResult,
    FetchKeyData,
    DepseudoPaths,
    AnonymizationMode,
    ExcludeField,
    BlankOutMode,
    FoldMode,
    ImageReplaceMode,
    DownloadOption,
)
import argparse
import sys


def _choices(enum_cls) -> list[str]:
    return [member.value for member in enum_cls]


def _add_ssl_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ssl-no-verify",
        action="store_true",
        dest="no_ssl_verify",
        default=None,
        help="Turn SSL-Verification off",
    )


def _add_settings_args(parser: argparse.ArgumentParser) -> None:
    """Inline AnonymizationSettings flags for the 'pseudo' command."""

    parser.add_argument(
        "--mode",
        choices=_choices(AnonymizationMode),
        type=str,
        dest="anonymization_mode",
        default="TYPE",
        help="Pseudonymization/Anonymization Mode: TYPE: COMPANY1, TYPE_PRESERVING: Jane Doe, XXX: XXX (pure "
        "anonymization, no depseudonymizaton possible), MT: as TYPE_PRESERVING but optimized for machine translation",
    )

    parser.add_argument(
        "--exclude_fields",
        choices=_choices(ExcludeField),
        dest="exclude_fields",
        nargs="+",
        default=[],
        help="Entities which should not be anonymized (default: none)",
    )

    parser.add_argument(
        "--workflow",
        type=str,
        dest="workflow",
        default=None,
        help="If workflows are defined in the installation, choose the workflow to be used",
    )

    parser.add_argument(
        "--foldMode",
        choices=_choices(FoldMode),
        dest="fold_mode",
        default=None,
        help="Degree to which variations of the same entity are folded to the same anonymization/pseudonym "
        "(default: server default, MINOR_VAR)",
    )

    parser.add_argument(
        "--blankOutMode",
        choices=_choices(BlankOutMode),
        dest="blank_out_mode",
        default=None,
        help="XXX: replace by 3 X, XLEN: replace by as many X as characters in the original, "
        "BLACK: replace by a black block",
    )

    parser.add_argument(
        "--imageReplace",
        choices=_choices(ImageReplaceMode),
        dest="image_replace",
        default=None,
        help="How images are replaced: DUMMY, BLANK, TRANSPARENT, AUTO (default: server default, AUTO)",
    )

    parser.add_argument(
        "--addNoTranslateMarkup",
        dest="add_no_translate_markup",
        action="store_true",
        default=None,
        help="Add [notranslate]...[/notranslate] markup around all anonymizations",
    )

    parser.add_argument(
        "--custom",
        dest="custom",
        nargs="+",
        default=[],
        help="Installation-specific custom options (see Glanos support for available options)",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="glanos-anonymization")

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # --- pseudo ---------------------------------------------------------

    pseudo_parser = subparsers.add_parser(
        "pseudo", help="Anonymize/pseudonymize a document"
    )

    pseudo_input_group = pseudo_parser.add_mutually_exclusive_group(required=True)
    pseudo_input_group.add_argument(
        "--file", type=str, dest="file", help="Path of file to be anonymized."
    )
    pseudo_input_group.add_argument(
        "--text",
        type=str,
        dest="text",
        help="Plain text to anonymize (alternative to --file).",
    )

    pseudo_parser.add_argument(
        "--filename",
        type=str,
        dest="filename",
        default=None,
        help="Override the fileName sent to the API (must carry the correct extension). "
        "Required by the API even for --text uploads; defaults to 'plain_text.txt' there.",
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

    _add_ssl_arg(pseudo_parser)

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
        help="Milliseconds to keep pseudoKey in the database after processing (default 0) - after that any depseudo request must contain the returned pseudoKey, -1 for infinite retention. Only honored for --async requests followed by 'fetch'; a non-async request discards its pseudoKey server-side immediately regardless of this setting, so later 'depseudo --metaKey' needs --pseudoKeyFile (via --returnPseudoKey) instead",
    )

    pseudo_parser.add_argument(
        "--metaRetentionMS",
        dest="meta_retention_ms",
        type=int,
        default=0,
        help="Milliseconds to keep meta data (including encryptKey) in the database after processing (default 0) - after that any depseudo request must contain the returned pseudoKey and the encryptKey, -1 for infinite retention. metaKey-only 'depseudo --metaKey' (via --async + fetch) also needs this set to a positive value together with --pseudoKeyRetentionMS - otherwise the server rejects the depseudo request with 'No such id ...'",
    )

    pseudo_parser.add_argument(
        "--settingsId",
        type=str,
        dest="settings_id",
        default=None,
        help="Reference to an existing stored setting (ignored if --mode/--exclude_fields/... are also given, "
        "since inline settings take precedence)",
    )

    pseudo_parser.add_argument(
        "--assignee",
        type=str,
        dest="assignee",
        default=None,
        help="User-id (usually lower-cased email) to assign this document to for dashboard review",
    )

    pseudo_parser.add_argument(
        "--assignedGroup",
        type=str,
        dest="assigned_group",
        default=None,
        help="Group-id to assign this document to for dashboard review (ignored if --assignee is also given)",
    )

    pseudo_parser.add_argument(
        "--session",
        type=str,
        dest="session",
        default=None,
        help="Session ID to use for consistent pseudonyms across multiple requests",
    )

    pseudo_parser.add_argument(
        "--sessionMaxAge",
        type=int,
        dest="session_max_age",
        default=None,
        help="If >0 and --session is not given, creates a session with this expiration age (ms) "
        "that stores the pseudonym mapping for consistent pseudonyms across requests",
    )

    pseudo_parser.add_argument(
        "--options",
        dest="options",
        nargs="+",
        default=[],
        help="Installation-specific custom upload options (see Glanos support for available options)",
    )

    _add_settings_args(pseudo_parser)

    # --- fetch ------------------------------------------------------------

    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Fetch the pseudonymization result of an already uploaded document",
    )
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

    _add_ssl_arg(fetch_parser)

    fetch_parser.add_argument(
        "--returnPseudoKey",
        dest="return_pseudo_key",
        action="store_true",
        default=None,
        help="Whether to return the pseudo key (API default true), data cannot be depseudonymized without pseudo key after pseudoKey retention time",
    )

    fetch_parser.add_argument(
        "--returnEncryptKey",
        dest="return_encrypt_key",
        action="store_true",
        default=None,
        help="Whether to return the encryption key (default false), data may then be depseudonymized with pseudo and encryption key after meta data retention time",
    )

    fetch_parser.add_argument(
        "--no-start-retention-period",
        dest="start_retention_period",
        action="store_false",
        default=None,
        help="Suppress the start of the retention period on a successful fetch (default: retention period starts)",
    )

    fetch_parser.add_argument(
        "--downloadOptions",
        choices=_choices(DownloadOption),
        dest="download_options",
        default=None,
        help="Optional output conversion, e.g. PDF_AS_PDF",
    )

    # --- depseudo -----------------------------------------------------

    depseudo_parser = subparsers.add_parser(
        "depseudo", help="Depseudonymize a document"
    )
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
        help="Provide file with pseudoKey (sufficient on its own; --metaKey is not needed alongside it)",
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
        help="metaKey as returned by pseudonymization, required unless --pseudoKeyFile or --session is provided. "
        "Only usable if that document was pseudonymized with --async and BOTH --pseudoKeyRetentionMS and "
        "--metaRetentionMS set to a positive value (either alone is not enough) - a synchronous pseudo call "
        "discards its pseudoKey immediately regardless of these settings; use --pseudoKeyFile (from "
        "--returnPseudoKey) for sync results",
    )

    depseudo_parser.add_argument(
        "--session",
        type=str,
        dest="depseudo_session",
        required=False,
        default=None,
        help="Session ID to use instead of --metaKey/--pseudoKeyFile/--encryptKeyFile",
    )

    depseudo_parser.add_argument(
        "--filename",
        type=str,
        dest="depseudo_filename",
        required=False,
        default=None,
        help="fileName to send with the request. The API tracks documents by the fileName it "
        "originally assigned (printed as 'fileName: ...' by the pseudo/fetch command) - pass "
        "that here. Defaults to the local --file's basename, which only works if the file was "
        "not renamed after downloading it.",
    )

    depseudo_parser.add_argument(
        "--outputFile",
        type=str,
        dest="output_file",
        required=True,
        help="Name of the output file",
    )

    _add_ssl_arg(depseudo_parser)

    # --- session-remove -----------------------------------------------

    session_remove_parser = subparsers.add_parser(
        "session-remove",
        help="Remove a session by id (securely removes its pseudonym mapping)",
    )
    session_remove_parser.add_argument("--token", type=str, dest="token", required=True)
    session_remove_parser.add_argument(
        "--baseUrl", type=str, dest="base_url", required=True
    )
    session_remove_parser.add_argument(
        "--session",
        type=str,
        dest="session_id",
        required=True,
        help="ID of the session",
    )
    _add_ssl_arg(session_remove_parser)

    args = parser.parse_args(argv)

    match args.command:
        case "pseudo":
            if args.a_sync:
                if args.return_pseudo_key or args.return_encrypt_key:
                    pseudo_parser.error(
                        "Async mode has to be turned off to return pseudo or encryption key!"
                    )
            elif not args.output_file:
                pseudo_parser.error(
                    "An output filename has to be specified when async mode is off."
                )
        case "depseudo":
            if not (
                args.depseudo_session
                or args.meta_key
                or args.pseudo_key_file
                or args.encrypt_key_file
            ):
                depseudo_parser.error(
                    "Provide --session, --metaKey, --pseudoKeyFile, and/or --encryptKeyFile."
                )

    return args


def create_ano_config(args: argparse.Namespace) -> AnonymizationConfig:
    """
    Build an AnonymizationConfig from parsed CLI arguments.

    Only known AnonymizationConfig fields are forwarded; unrelated CLI
    arguments (token, base_url, file, ...) are ignored explicitly rather
    than relying on the model silently discarding unknown fields.
    """

    config_fields = set(AnonymizationConfig.model_fields)
    config_kwargs = {
        k: v
        for k, v in vars(args).items()
        if k in config_fields and k != "a_sync" and v is not None
    }

    # Only the 'pseudo' command exposes --async; other commands always
    # request the completed result, so default a_sync to False for them.
    config_kwargs["a_sync"] = bool(getattr(args, "a_sync", False))

    return AnonymizationConfig(**config_kwargs)


def exec_command(args: argparse.Namespace) -> None:
    ano_config = create_ano_config(args)

    with GlanosClient(
        token=args.token, base_url=args.base_url, config=ano_config
    ) as glanos_client:
        match args.command:
            case "pseudo":
                try:
                    if args.file:
                        pseudo_result = glanos_client.pseudo_sync(
                            path=args.file, filename=args.filename
                        )
                    else:
                        pseudo_result = glanos_client.pseudo_sync(
                            text=args.text, filename=args.filename
                        )
                except Exception as e:
                    print(f"Error while executing pseudo command: {e}", file=sys.stderr)
                    sys.exit(1)

                glanos_client.manage_data(pseudo_result, output_file=args.output_file)

            case "fetch":
                pseudo_result = PseudoResult(
                    success=True, data=FetchKeyData(fetchKey=args.fetch_key)
                )
                fetch_kwargs = {}
                if args.return_pseudo_key is not None:
                    fetch_kwargs["return_pseudo_key"] = args.return_pseudo_key
                if args.return_encrypt_key is not None:
                    fetch_kwargs["return_encrypt_key"] = args.return_encrypt_key
                if args.start_retention_period is not None:
                    fetch_kwargs["start_retention_period"] = args.start_retention_period
                if args.download_options is not None:
                    fetch_kwargs["download_options"] = DownloadOption(
                        args.download_options
                    )

                try:
                    fetch_result = glanos_client.fetch(pseudo_result, **fetch_kwargs)
                except Exception as e:
                    print(f"Error while executing fetch command: {e}", file=sys.stderr)
                    sys.exit(1)

                glanos_client.manage_data(fetch_result, args.output_file)

            case "depseudo":
                try:
                    depseudo_paths = DepseudoPaths(
                        pseudo_key_path=args.pseudo_key_file,
                        meta_key=args.meta_key,
                        encrypt_key_path=args.encrypt_key_file,
                    )
                    depseudo_result = glanos_client.depseudo(
                        args.file,
                        depseudo_paths,
                        session=args.depseudo_session,
                        filename=args.depseudo_filename,
                    )
                except Exception as e:
                    print(
                        f"Error while executing depseudo command: {e}", file=sys.stderr
                    )
                    sys.exit(1)

                glanos_client.manage_data(depseudo_result, output_file=args.output_file)

            case "session-remove":
                try:
                    result = glanos_client.remove_session(args.session_id)
                except Exception as e:
                    print(
                        f"Error while executing session-remove command: {e}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                print(result.model_dump_json())


def main(argv: list[str] | None = None) -> None:
    cli_args = parse_args(argv)
    exec_command(cli_args)


if __name__ == "__main__":
    main()
