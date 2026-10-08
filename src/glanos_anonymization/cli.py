from glanos_anonymization.core import (
    GlanosClient,
    AnonymizationSettings,
    PseudoResult,
    FetchKeyData,
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
from typing import Any
from pydantic import BaseModel
import argparse
import sys


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

    parser.add_argument(
        "--token",
        type=str,
        dest="token",
        required=True,
        help="Glanos anonymization.ai API token",
    )
    parser.add_argument(
        "--baseUrl",
        type=str,
        dest="base_url",
        required=True,
        help="Glanos anonymization.ai base URL",
    )
    _add_ssl_arg(parser)


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
        help="Degree to which variations of the same entity are folded to the same anonymization/pseudonym"
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

    pseudo_parser.add_argument(
        "--token",
        type=str,
        dest="token",
        required=True,
        help="Glanos anonymization.ai API token",
    )

    pseudo_parser.add_argument(
        "--baseUrl",
        type=str,
        dest="base_url",
        required=True,
        help="Glanos anonymization.ai base URL",
    )

    pseudo_parser.add_argument(
        "--language",
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

    _add_switch(
        pseudo_parser,
        "returnPseudoKey",
        "return_pseudo_key",
        "[Only without --async] Return the pseudo key and write it to <outputFile>.pkey "
        "(default: on, so that the document can be restored with --pseudoKeyFile and --encryptKeyFile)",
    )
    _add_switch(
        pseudo_parser,
        "returnEncryptKey",
        "return_encrypt_key",
        "[Only without --async] Return the encryption key and write it to <outputFile>.ekey (default: on)",
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
        help="ID of a stored setting"
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
        default=None,
        help="Installation-specific custom upload options (see Glanos anonymization.ai support for available options)",
    )

    _add_settings_args(pseudo_parser)

    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Fetch the pseudonymization result of an already uploaded document",
    )
    fetch_parser.add_argument(
        "--token",
        type=str,
        dest="token",
        required=True,
        help="Glanos anonymization.ai API token",
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
        help="Glanos anonymization.ai base URL",
    )

    _add_ssl_arg(fetch_parser)

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

    depseudo_parser.add_argument(
        "--token",
        type=str,
        dest="token",
        required=True,
        help="Glanos anonymization.ai API token",
    )

    depseudo_parser.add_argument(
        "--baseUrl",
        type=str,
        dest="base_url",
        required=True,
        help="Glanos anonymization.ai base URL",
    )

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
        help="fileName to send with the request"
    )

    depseudo_parser.add_argument(
        "--outputFile",
        type=str,
        dest="output_file",
        required=True,
        help="Name of the output file",
    )

    _add_ssl_arg(depseudo_parser)

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
                    "session_max_age": args.session_max_age,
                    "a_sync": args.a_sync,
                    "return_pseudo_key": args.return_pseudo_key,
                    "return_encrypt_key": args.return_encrypt_key,
                    "language": args.language,
                    "data_retention_ms": args.data_retention_ms,
                    "pseudo_key_retention_ms": args.pseudo_key_retention_ms,
                    "meta_retention_ms": args.meta_retention_ms,
                    "assignee": args.assignee,
                    "assigned_group": args.assigned_group,
                    "options": args.options,
                }
                try:
                    if args.file:
                        pseudo_result = glanos_client.pseudo_sync(
                            path=args.file, **request
                        )
                    else:
                        pseudo_result = glanos_client.pseudo_sync(
                            text=args.text, **request
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
