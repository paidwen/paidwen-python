from __future__ import annotations

import contextlib
import functools
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

import httpx

from ._contract import Problem
from ._errors import PaidwenError, RequestError
from ._journal import quiet_call

IdentityToken = Callable[[], str]

TOKEN_EXCHANGE = "urn:ietf:params:oauth:grant-type:token-exchange"
JWT_TYPE = "urn:ietf:params:oauth:token-type:jwt"
REFRESH_BEFORE = 5 * 60.0
EXPIRY_GUARD = 30.0


class Unavailable(Exception):
    def __init__(self, message: str, reason: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.reason = reason


class Credentials(Protocol):
    federated: bool

    def authorization(self) -> str: ...

    def reject(self, authorization: str) -> None: ...


class KeyCredentials:
    federated = False

    def __init__(self, api_key: str) -> None:
        self._authorization = f"Bearer {api_key}"

    def authorization(self) -> str:
        return self._authorization

    def reject(self, authorization: str) -> None:
        return None


class FederatedCredentials:
    federated = True

    def __init__(
        self, identity_token: IdentityToken, base: str, client: httpx.Client, timeout: float
    ) -> None:
        self._identity_token = identity_token
        self._base = base
        self._client = client
        self._timeout = timeout
        self._token: str | None = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def authorization(self) -> str:
        with self._lock:
            remaining = self._expires_at - time.monotonic()
            if self._token is not None and remaining > REFRESH_BEFORE:
                return f"Bearer {self._token}"
            if self._token is not None and remaining > EXPIRY_GUARD:
                with contextlib.suppress(Unavailable):
                    self._exchange()
                return f"Bearer {self._token}"
            self._exchange()
            return f"Bearer {self._token}"

    def reject(self, authorization: str) -> None:
        with self._lock:
            if authorization == f"Bearer {self._token}":
                self._token = None
                self._expires_at = 0.0

    def _exchange(self) -> None:
        subject = self._subject()
        started = time.monotonic()
        try:
            response = quiet_call(
                functools.partial(
                    self._client.post,
                    f"{self._base}/v1/oauth/token",
                    data={
                        "grant_type": TOKEN_EXCHANGE,
                        "subject_token": subject,
                        "subject_token_type": JWT_TYPE,
                    },
                    headers={"accept": "application/json"},
                    timeout=self._timeout,
                )
            )
        except httpx.HTTPError as error:
            raise Unavailable("Paidwen did not answer the token exchange", str(error)) from error
        try:
            answer: Any = response.json()
        except ValueError as error:
            raise Unavailable(
                f"Paidwen answered the token exchange with HTTP {response.status_code}", str(error)
            ) from error
        if response.status_code >= 500:
            raise Unavailable(
                f"Paidwen answered the token exchange with HTTP {response.status_code}",
                _description(answer),
            )
        if not response.is_success:
            raise RequestError(response.status_code, _problem(response.status_code, answer))
        token = answer.get("access_token") if isinstance(answer, dict) else None
        expires_in = answer.get("expires_in") if isinstance(answer, dict) else None
        if not isinstance(token, str) or not isinstance(expires_in, int):
            raise Unavailable("Paidwen answered the token exchange without a token")
        self._token = token
        self._expires_at = started + expires_in

    def _subject(self) -> str:
        try:
            value = self._identity_token()
        except Exception as error:
            raise PaidwenError(f"Paidwen could not read the identity token: {error}") from error
        if not isinstance(value, str) or not value.strip():
            raise PaidwenError("The identity token is empty.")
        return value.strip()


def identity_token_file(path: str) -> IdentityToken:
    return lambda: Path(path).read_text(encoding="utf-8")


def _problem(status: int, answer: Any) -> Problem:
    title = answer.get("error") if isinstance(answer, dict) else None
    problem: dict[str, Any] = {
        "type": "about:blank",
        "title": title if isinstance(title, str) else "invalid_request",
        "status": status,
    }
    description = _description(answer)
    if description is not None:
        problem["detail"] = description
    return cast(Problem, problem)


def _description(answer: Any) -> str | None:
    description = answer.get("error_description") if isinstance(answer, dict) else None
    return description if isinstance(description, str) and description else None
