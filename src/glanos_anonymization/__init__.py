from typing import TYPE_CHECKING

from glanos_anonymization.core import (
    AnonymizationResult,
    AnonymizationSettings,
    AnonymizationSettingsList,
    AsyncGlanosClient,
    AuthenticationError,
    DataExpiredError,
    DeanonymizationResult,
    DepseudoPaths,
    GlanosAPIError,
    GlanosClient,
    GlanosClientError,
    InvalidInputError,
    Processing,
    ProcessingFailedError,
    QuotaExceededError,
    RateLimitedError,
    ResultNotReadyError,
    ServerError,
    SessionInfo,
    SettingsConflictError,
    SettingNotFoundError,
    SettingsSort,
    SortOrder,
)
from glanos_anonymization.tool import GlanosAnonymizationTools

if TYPE_CHECKING:
    from glanos_anonymization.middleware import GlanosAnonymizationMiddleware


def __getattr__(name: str):
    # The middleware needs the optional 'langchain' dependencies, so it is only
    # imported when it is used.
    if name == "GlanosAnonymizationMiddleware":
        from glanos_anonymization.middleware import GlanosAnonymizationMiddleware

        return GlanosAnonymizationMiddleware
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnonymizationResult",
    "AnonymizationSettings",
    "AnonymizationSettingsList",
    "AsyncGlanosClient",
    "AuthenticationError",
    "DataExpiredError",
    "DeanonymizationResult",
    "DepseudoPaths",
    "GlanosAPIError",
    "GlanosAnonymizationMiddleware",
    "GlanosAnonymizationTools",
    "GlanosClient",
    "GlanosClientError",
    "InvalidInputError",
    "Processing",
    "ProcessingFailedError",
    "QuotaExceededError",
    "RateLimitedError",
    "ResultNotReadyError",
    "ServerError",
    "SessionInfo",
    "SettingsConflictError",
    "SettingNotFoundError",
    "SettingsSort",
    "SortOrder",
]
