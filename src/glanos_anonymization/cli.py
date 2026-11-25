from glanos_anonymization.core import (
    GlanosClient,
    AnonymizationSettings,
    PseudoResult,
    InnerResponseData,
    Processing,
    DepseudoPaths,
    AnonymizationMode,
    ExcludeField,
    BlankOutMode,
    FoldMode,
    ImageReplaceMode,
    DownloadOption,
    SettingsSort,
    SortOrder,
)
from collections.abc import Callable
from pathlib import Path
from typing import Any
from pydantic import BaseModel
import argparse
import base64
import os
import sys

TOKEN_ENV_VAR = "GLANOS_TOKEN"
BASE_URL_ENV_VAR = "GLANOS_BASE_URL"


def _choices(enum_cls) -> list[str]:
    return [member.value for member in enum_cls]


class _ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)


def _add_switch(
    parser: argparse.ArgumentParser, name: str, dest: str, help_text: str
) -> None:
    parser.add_argument(
        f"--{name}", dest=dest, action="store_true", default=None, help=help_text
    )
    parser.add_argument(
        f"--no{name[0].upper()}{name[1:]}",
        dest=dest,
        action="store_false",
        default=None,
        help=f"Turn --{name} off",
    )


def _add_ssl_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sslNoVerify",
        action="store_true",
        dest="no_ssl_verify",
        default=None,
        help="Turn SSL-Verification off",
    )


def _add_connection_args(parser: argparse.ArgumentParser) -> None:

    # Read from the environment when the flag is left out, so that the token
    # does not have to appear in the shell history or the process list.
    token = os.environ.get(TOKEN_ENV_VAR) or None
    base_url = os.environ.get(BASE_URL_ENV_VAR) or None
    parser.add_argument(
        "--token",
        type=str,
        dest="token",
        required=token is None,
        default=token,
        help=f"Glanos anonymization.ai API token (default: ${TOKEN_ENV_VAR})",
    )
    parser.add_argument(
        "--baseUrl",
        type=str,
        dest="base_url",
        required=base_url is None,
        default=base_url,
        help=f"Glanos anonymization.ai base URL (default: ${BASE_URL_ENV_VAR})",
    )
    _add_ssl_arg(parser)


def _add_polling_args(
    parser: argparse.ArgumentParser, defaults: tuple[str, str], applies_to: str = ""
) -> None:
    parser.add_argument(
        "--pollInterval",
        type=float,
        dest="poll_interval",
        default=None,
        help=f"{applies_to}Seconds between two checks whether the document is ready "
        f"(default: {defaults[0]})",
    )
    parser.add_argument(
        "--maxAttempts",
        type=int,
        dest="max_attempts",
        default=None,
        help=f"{applies_to}How often to check before giving up and printing the "
        f"fetchKey (default: {defaults[1]})",
    )


def _add_settings_args(parser: argparse.ArgumentParser) -> None:

    parser.add_argument(
        "--mode",
        choices=_choices(AnonymizationMode),
        type=str,
        dest="anonymization_mode",
        default=None,
        help="Pseudonymization/Anonymization Mode (default: TYPE): TYPE: COMPANY1, TYPE_PRESERVING: Jane Doe, XXX: XXX (pure "
        "anonymization, no depseudonymizaton possible), MT: as TYPE_PRESERVING but optimized for machine translation",
    )

    parser.add_argument(
        "--excludeFields",
        choices=_choices(ExcludeField),
        dest="exclude_fields",
        nargs="+",
        default=None,
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
        help="Degree to which variations of the same entity are folded to the same anonymization/pseudonym",
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
        help="How images are replaced: DUMMY, BLANK, TRANSPARENT, AUTO (default: AUTO)",
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
        default=None,
        help="Installation-specific custom options (see Glanos anonymization.ai support for available options)",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _ArgumentParser(prog="glanos-anonymization")

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        parser_class=_ArgumentParser,
    )

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
        "--fileName",
        type=str,
        dest="filename",
        default=None,
        help="Override the fileName sent to the API (must carry the correct extension). "
        "Required by the API even for --text uploads; defaults to 'plain_text.txt' there.",
    )

    _add_connection_args(pseudo_parser)

    pseudo_parser.add_argument(
        "--language",
        type=str,
        dest="language",
        required=False,
        help="ISO-639 code of document language",
    )

    pseudo_parser.add_argument(
        "--processing",
        choices=_choices(Processing),
        dest="processing",
        default=None,
        help="How to wait for the result (default: sync for --text, poll for --file): "
        "sync: one request that waits for the document; poll: submit, then check "
        "until it is ready; no_wait: submit and print the fetchKey at once. If the "
        "document is not ready in time, the fetchKey is printed: collect it with 'fetch'",
    )

    _add_polling_args(pseudo_parser, ("10", "30"), "[Only with --processing poll] ")

    pseudo_parser.add_argument(
        "--outputFile",
        type=str,
        dest="output_file",
        required=False,
        help="Name of the output file [Required unless the document is only submitted: "
        "--processing no_wait, --assignee, --assignedGroup]",
    )

    _add_switch(
        pseudo_parser,
        "returnPseudoKey",
        "return_pseudo_key",
        "[Only when waiting for the result] Return the pseudo key and write it to <outputFile>.pkey "
        "(default: on, so that the document can be restored with --pseudoKeyFile and --encryptKeyFile)",
    )
    _add_switch(
        pseudo_parser,
        "returnEncryptKey",
        "return_encrypt_key",
        "[Only when waiting for the result] Return the encryption key and write it to <outputFile>.ekey "
        "(default: on)",
    )

    pseudo_parser.add_argument(
        "--dataRetentionMS",
        dest="data_retention_ms",
        type=int,
        default=None,
        help="Milliseconds to keep the anonymized document on the server, -1 for infinite retention "
        "(API default 0: expires right after download)",
    )

    pseudo_parser.add_argument(
        "--pseudoKeyRetentionMS",
        dest="pseudo_key_retention_ms",
        type=int,
        default=None,
        help="Milliseconds to keep the pseudoKey on the server - after that any depseudo request must contain "
        "the returned pseudoKey, -1 for infinite retention (API default 0: expires right after download)",
    )

    pseudo_parser.add_argument(
        "--metaRetentionMS",
        dest="meta_retention_ms",
        type=int,
        default=None,
        help="Milliseconds to keep the meta data (including encryptKey) on the server - until then "
        "'depseudo --metaKey' works, after that only --pseudoKeyFile with --encryptKeyFile; -1 for infinite "
        "retention (API default 0: per spec 1h after download). "
        "Only for --file: --text requests get no metaKey and are restored with --pseudoKeyFile and --encryptKeyFile",
    )

    pseudo_parser.add_argument(
        "--settingsId",
        type=str,
        dest="settings_id",
        default=None,
        help="ID of a stored setting",
    )

    pseudo_parser.add_argument(
        "--assignee",
        type=str,
        dest="assignee",
        default=None,
        help="User-id (usually lower-cased email) to assign this document to for dashboard review. "
        "The document is only submitted and the fetchKey printed; collect it with 'fetch' after the review",
    )

    pseudo_parser.add_argument(
        "--assignedGroup",
        type=str,
        dest="assigned_group",
        default=None,
        help="Group-id to assign this document to for dashboard review (instead of --assignee). "
        "The document is only submitted and the fetchKey printed; collect it with 'fetch' after the review",
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
        dest="session_max_age_ms",
        default=None,
        help="If >0 and --session is not given, creates a session with this expiration age (ms) "
        "that stores the pseudonym mapping for consistent pseudonyms across requests",
    )

    pseudo_parser.add_argument(
        "--options",
        dest="options",
        nargs="+",
        default=None,
        help="Installation-specific custom upload options (see Glanos anonymization.ai support for available options)",
    )

    _add_settings_args(pseudo_parser)

    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Fetch the pseudonymization result of an already uploaded document",
    )
    _add_connection_args(fetch_parser)

    fetch_parser.add_argument(
        "--fetchKey",
        type=str,
        dest="fetch_key",
        required=True,
        help="FetchKey from Pseudo Request",
    )

    _add_polling_args(fetch_parser, ("300", "no limit"))

    fetch_parser.add_argument(
        "--outputFile",
        type=str,
        dest="output_file",
        required=True,
        help="Name of the output file",
    )

    _add_switch(
        fetch_parser,
        "returnPseudoKey",
        "return_pseudo_key",
        "Return the pseudo key and write it to <outputFile>.pkey (default: on)",
    )
    _add_switch(
        fetch_parser,
        "returnEncryptKey",
        "return_encrypt_key",
        "Return the encryption key and write it to <outputFile>.ekey (default: on)",
    )

    fetch_parser.add_argument(
        "--noStartRetentionPeriod",
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

    _add_connection_args(depseudo_parser)

    depseudo_parser.add_argument(
        "--pseudoKeyFile",
        type=str,
        dest="pseudo_key_file",
        required=False,
        default=None,
        help="File with the pseudoKey (<outputFile>.pkey of 'pseudo'). Together with --encryptKeyFile it always "
        "restores the document; without it, combine it with --metaKey while the meta data is kept",
    )

    depseudo_parser.add_argument(
        "--encryptKeyFile",
        type=str,
        dest="encrypt_key_file",
        required=False,
        default=None,
        help="File with the encryptKey (<outputFile>.ekey of 'pseudo'); use it together with --pseudoKeyFile",
    )

    depseudo_parser.add_argument(
        "--metaKey",
        type=str,
        dest="meta_key",
        required=False,
        default=None,
        help="metaKey as returned by pseudonymization. Only usable while the meta data is kept, i.e. if the "
        "document was pseudonymized with --metaRetentionMS > 0; otherwise use --pseudoKeyFile with "
        "--encryptKeyFile, or --session",
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
        "--fileName",
        type=str,
        dest="depseudo_filename",
        required=False,
        default=None,
        help="fileName to send with the request",
    )

    depseudo_parser.add_argument(
        "--outputFile",
        type=str,
        dest="output_file",
        required=True,
        help="Name of the output file",
    )

    session_remove_parser = subparsers.add_parser(
        "session-remove",
        help="Remove a session by id (securely removes its pseudonym mapping)",
    )
    _add_connection_args(session_remove_parser)
    session_remove_parser.add_argument(
        "--session",
        type=str,
        dest="session_id",
        required=True,
        help="ID of the session",
    )

    session_add_parser = subparsers.add_parser(
        "session-add",
        help="Create a session for consistent pseudonyms across multiple requests",
    )
    _add_connection_args(session_add_parser)
    session_add_parser.add_argument(
        "--maxAgeSeconds",
        type=int,
        dest="max_age_seconds",
        required=True,
        help="Lifetime of the session in SECONDS",
    )
    session_add_parser.add_argument(
        "--name", type=str, dest="session_name", default=None, help="Session name"
    )
    session_add_parser.add_argument(
        "--settingsId",
        type=str,
        dest="session_settings_id",
        required=True,
        help="ID of the stored setting the session is based on (see setting-add)",
    )
    session_add_parser.add_argument(
        "--shareUsers",
        dest="share_users",
        nargs="+",
        default=None,
        help="Users allowed to use the session",
    )
    session_add_parser.add_argument(
        "--shareGroups",
        dest="share_groups",
        nargs="+",
        default=None,
        help="Groups allowed to use the session",
    )

    session_get_parser = subparsers.add_parser(
        "session-get", help="Show a session by id"
    )
    _add_connection_args(session_get_parser)
    session_get_parser.add_argument(
        "--session",
        type=str,
        dest="session_id",
        required=True,
        help="ID of the session",
    )

    session_extend_parser = subparsers.add_parser(
        "session-extend",
        help="Set a session's expiration to now plus --maxAgeSeconds",
    )
    _add_connection_args(session_extend_parser)
    session_extend_parser.add_argument(
        "--session",
        type=str,
        dest="session_id",
        required=True,
        help="ID of the session",
    )
    session_extend_parser.add_argument(
        "--maxAgeSeconds",
        type=int,
        dest="max_age_seconds",
        required=True,
        help="New lifetime from now on, in SECONDS",
    )

    setting_get_parser = subparsers.add_parser(
        "setting-get", help="Show a stored setting by id"
    )
    _add_connection_args(setting_get_parser)
    setting_get_parser.add_argument(
        "--setting",
        type=str,
        dest="setting_id",
        required=True,
        help="ID of the setting",
    )

    setting_list_parser = subparsers.add_parser(
        "setting-list", help="List stored settings"
    )
    _add_connection_args(setting_list_parser)
    setting_list_parser.add_argument(
        "--query",
        type=str,
        dest="query",
        default=None,
        help="Fuzzy match on setting name",
    )
    setting_list_parser.add_argument(
        "--creator",
        dest="creator",
        nargs="+",
        default=None,
        help="One or more setting creators (the setting still has to be shared with the API user)",
    )
    setting_list_parser.add_argument(
        "--sort", choices=_choices(SettingsSort), dest="sort", default=None
    )
    setting_list_parser.add_argument(
        "--sortOrder", choices=_choices(SortOrder), dest="sort_order", default=None
    )
    setting_list_parser.add_argument(
        "--limit",
        type=int,
        dest="limit",
        default=None,
        help="Maximum number of matches (API default 10)",
    )
    setting_list_parser.add_argument(
        "--offset",
        type=int,
        dest="offset",
        default=None,
        help="0-based pagination offset",
    )

    setting_create_parser = subparsers.add_parser(
        "setting-add",
        help="Store a new setting (the server assigns its ID, printed in the result)",
    )
    _add_connection_args(setting_create_parser)
    setting_create_parser.add_argument(
        "--name",
        type=str,
        dest="setting_name",
        required=True,
        help="Name of the setting",
    )
    _add_settings_args(setting_create_parser)

    setting_update_parser = subparsers.add_parser(
        "setting-update",
        help="Change a stored setting",
    )
    _add_connection_args(setting_update_parser)
    setting_update_parser.add_argument(
        "--setting",
        type=str,
        dest="setting_id",
        required=True,
        help="ID of the setting to change",
    )
    setting_update_parser.add_argument(
        "--name",
        type=str,
        dest="setting_name",
        default=None,
        help="New name of the setting",
    )
    _add_settings_args(setting_update_parser)

    setting_remove_parser = subparsers.add_parser(
        "setting-remove", help="Remove a stored setting by id"
    )
    _add_connection_args(setting_remove_parser)
    setting_remove_parser.add_argument(
        "--setting",
        type=str,
        dest="setting_id",
        required=True,
        help="ID of the setting",
    )

    args = parser.parse_args(argv)

    match args.command:
        case "pseudo":
            _check_pseudo_args(pseudo_parser, args)
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


def _check_pseudo_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    polling_given = args.poll_interval is not None or args.max_attempts is not None
    keys_given = (
        args.return_pseudo_key is not None or args.return_encrypt_key is not None
    )

    if args.assignee is not None or args.assigned_group is not None:
        if args.assignee is not None and args.assigned_group is not None:
            parser.error("Pass either --assignee or --assignedGroup, not both.")
        if args.processing is not None or polling_given:
            parser.error(
                "A document for review is only submitted: --processing, --pollInterval "
                "and --maxAttempts do not apply. Collect it with 'fetch' after the review."
            )
        waits = False
    else:
        processing = args.processing or (
            Processing.SYNC if args.text is not None else Processing.POLL
        )
        if polling_given and processing != Processing.POLL:
            parser.error(
                "--pollInterval and --maxAttempts only apply to --processing poll."
            )
        waits = processing != Processing.NO_WAIT

    if not waits and keys_given:
        parser.error(
            "The document is only submitted, so its keys are requested by 'fetch': "
            "pass --returnPseudoKey/--returnEncryptKey there."
        )
    if waits and not args.output_file:
        parser.error(
            "An output filename has to be specified unless the document is only "
            "submitted (--processing no_wait, --assignee, --assignedGroup)."
        )


_SETTINGS_FLAGS = {
    "anonymization_mode": "mode",
    "exclude_fields": "excludeFields",
    "workflow": "workflow",
    "fold_mode": "foldMode",
    "blank_out_mode": "blankOutMode",
    "image_replace": "imageReplace",
    "add_no_translate_markup": "addNoTranslateMarkup",
    "custom": "custom",
}


def given_settings_fields(args: argparse.Namespace) -> dict:

    return {
        field: getattr(args, dest)
        for dest, field in _SETTINGS_FLAGS.items()
        if getattr(args, dest, None) is not None
    }


def build_inline_settings(args: argparse.Namespace) -> AnonymizationSettings | None:

    given = given_settings_fields(args)
    return AnonymizationSettings(**given) if given else None


def exec_command(args: argparse.Namespace) -> None:
    with GlanosClient(
        token=args.token,
        base_url=args.base_url,
        no_ssl_verify=bool(args.no_ssl_verify),
    ) as glanos_client:
        match args.command:
            case "pseudo":
                request = {
                    "filename": args.filename,
                    "settings": build_inline_settings(args),
                    "settings_id": args.settings_id,
                    "session": args.session,
                    "session_max_age_ms": args.session_max_age_ms,
                    "language": args.language,
                    "data_retention_ms": args.data_retention_ms,
                    "pseudo_key_retention_ms": args.pseudo_key_retention_ms,
                    "meta_retention_ms": args.meta_retention_ms,
                    "options": args.options,
                }
                if args.file:
                    request["path"] = args.file
                else:
                    request["text"] = args.text
                try:
                    if args.assignee is not None or args.assigned_group is not None:
                        result = glanos_client.anonymize_with_review(
                            assignee=args.assignee,
                            assigned_group=args.assigned_group,
                            **request,
                        )
                    else:
                        result = glanos_client.anonymize(
                            processing=args.processing,
                            poll_interval=args.poll_interval,
                            max_attempts=args.max_attempts,
                            return_pseudo_key=args.return_pseudo_key,
                            return_encrypt_key=args.return_encrypt_key,
                            **request,
                        )
                except Exception as e:
                    print(f"Error while executing pseudo command: {e}", file=sys.stderr)
                    sys.exit(1)

                manage_data(result.pseudo_result, output_file=args.output_file)

            case "fetch":
                fetch_kwargs = {}
                if args.poll_interval is not None:
                    fetch_kwargs["poll_interval"] = args.poll_interval
                if args.max_attempts is not None:
                    fetch_kwargs["max_attempts"] = args.max_attempts
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
                    fetch_result = glanos_client.fetch(args.fetch_key, **fetch_kwargs)
                except Exception as e:
                    print(f"Error while executing fetch command: {e}", file=sys.stderr)
                    sys.exit(1)

                manage_data(fetch_result.pseudo_result, args.output_file)

            case "depseudo":
                try:
                    depseudo_paths = DepseudoPaths(
                        pseudo_key_path=args.pseudo_key_file,
                        meta_key=args.meta_key,
                        encrypt_key_path=args.encrypt_key_file,
                    )
                    depseudo_result = glanos_client.deanonymize(
                        path=args.file,
                        filename=args.depseudo_filename,
                        session=args.depseudo_session,
                        depseudo_paths=depseudo_paths,
                    )
                except Exception as e:
                    print(
                        f"Error while executing depseudo command: {e}", file=sys.stderr
                    )
                    sys.exit(1)

                manage_data(depseudo_result.pseudo_result, output_file=args.output_file)

            case "session-remove":
                _print_result(
                    args.command, lambda: glanos_client.remove_session(args.session_id)
                )

            case "session-add":
                _print_result(
                    args.command,
                    lambda: glanos_client.add_session(
                        args.max_age_seconds,
                        name=args.session_name,
                        settings_id=args.session_settings_id,
                        share_users=args.share_users,
                        share_groups=args.share_groups,
                    ),
                )

            case "session-get":
                _print_result(
                    args.command, lambda: glanos_client.get_session(args.session_id)
                )

            case "session-extend":
                _print_result(
                    args.command,
                    lambda: glanos_client.extend_session(
                        args.session_id, args.max_age_seconds
                    ),
                )

            case "setting-get":
                _print_result(
                    args.command, lambda: glanos_client.get_setting(args.setting_id)
                )

            case "setting-list":
                _print_result(
                    args.command,
                    lambda: glanos_client.list_settings(
                        query=args.query,
                        creator=args.creator,
                        sort=args.sort,
                        sort_order=args.sort_order,
                        limit=args.limit,
                        offset=args.offset,
                    ),
                )

            case "setting-add":
                settings = AnonymizationSettings(
                    name=args.setting_name, **given_settings_fields(args)
                )
                _print_result(
                    args.command,
                    lambda: glanos_client.add_setting(settings),
                )

            case "setting-update":
                _print_result(
                    args.command,
                    lambda: update_setting(glanos_client, args),
                )

            case "setting-remove":
                _print_result(
                    args.command, lambda: glanos_client.remove_setting(args.setting_id)
                )


def write_to_output(
    data: str | bytes,
    output_file: str | Path,
    extension: str | None = None,
    mode: str = "w",
) -> None:

    path = Path(output_file)
    if extension:
        path = path.with_suffix(f".{extension}")

    with open(path, mode=mode) as f:
        f.write(data)


def write_key_file(output_file: str | Path, extension: str, key: str) -> Path:
    """Write a restoration key next to the output file, readable by its owner only."""

    path = Path(output_file).with_suffix(f".{extension}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        # os.open applies the mode only to new files
        os.chmod(path, 0o600)
        f.write(key)
    return path


def manage_data(
    pseudo_result: PseudoResult, output_file: str | Path | None = None
) -> None:

    data = pseudo_result.data
    if not isinstance(data, InnerResponseData):
        print(
            "The document is not processed yet. Collect it later with: "
            f"glanos-anonymization fetch --fetchKey {data.fetchKey} --outputFile <file>",
            file=sys.stderr,
        )
        print(data.model_dump_json())
        return

    content = base64.b64decode(data.data)
    write_to_output(content, output_file=output_file, mode="wb")

    print(f"metaKey: {data.metaKey}")
    print(f"fileName (in server): {data.fileName}")
    print(f"fileName (in output): {Path(output_file).name}")
    if data.session is not None:
        print(f"session: {data.session}")

    # The keys restore the document, so they are written to files only and
    # never printed.
    for label, key, extension in (
        ("pseudoKey", data.pseudoKey, "pkey"),
        ("encryptKey", data.encryptKey, "ekey"),
    ):
        if key is not None:
            print(f"{label} written to {write_key_file(output_file, extension, key)}")

    print(f"{output_file} (SZ: {len(content)})")


def update_setting(
    glanos_client: GlanosClient, args: argparse.Namespace
) -> AnonymizationSettings:

    current = glanos_client.get_setting(args.setting_id)
    changes = given_settings_fields(args)
    if args.setting_name is not None:
        changes["name"] = args.setting_name
    updated = AnonymizationSettings.model_validate(
        {**current.model_dump(mode="json"), **changes, "id": args.setting_id}
    )
    return glanos_client.update_setting(updated)


def _print_result(command: str, call: Callable[[], BaseModel]) -> None:
    try:
        result = call()
    except Exception as e:
        print(f"Error while executing {command} command: {e}", file=sys.stderr)
        sys.exit(1)

    print(result.model_dump_json())


def main(argv: list[str] | None = None) -> None:
    cli_args = parse_args(argv)
    exec_command(cli_args)


if __name__ == "__main__":
    main()
