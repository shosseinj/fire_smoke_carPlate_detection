from __future__ import annotations

import json
import logging
import re
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response


_PERSIAN_PATTERN = re.compile(r"[\u0600-\u06ff]")
_SENSITIVE_DETAIL_PATTERN = re.compile(
    r"(?i)(password\s*=|token\s*=|secret\s*=|host\s*=|traceback|postgresql://|rtsp://)"
)
LOGGER = logging.getLogger("uvicorn.error")

_EXACT_MESSAGES = {
    "Not Found": "مسیر یا منبع درخواستی یافت نشد",
    "Method Not Allowed": "روش درخواست برای این مسیر مجاز نیست",
    "Internal Server Error": "خطای داخلی در سرویس رخ داد",
    "Request not found": "درخواست یافت نشد",
    "Fullscreen source not found": "منبع انتخاب‌شده برای نمایش تمام‌صفحه یافت نشد",
    "Video wall is disabled": "دیوار ویدیویی غیرفعال است",
    "Logged out successfully": "خروج با موفقیت انجام شد",
    "Password changed successfully": "رمز عبور با موفقیت تغییر کرد",
}


def contains_persian(value: str) -> bool:
    return bool(_PERSIAN_PATTERN.search(value))


def public_error_message(value: object, *, status_code: int | None = None) -> str:
    text = str(value).strip()
    if status_code is not None and status_code >= 500:
        return "خطای داخلی در سرویس رخ داد؛ دوباره تلاش کنید"
    if contains_persian(text) and not _SENSITIVE_DETAIL_PATTERN.search(text):
        return text
    if text in _EXACT_MESSAGES:
        return _EXACT_MESSAGES[text]
    lowered = text.lower()
    if "not found" in lowered:
        return "مورد درخواستی یافت نشد"
    if "already exists" in lowered or "duplicate" in lowered:
        return "این مورد قبلاً ثبت شده است"
    if "required" in lowered:
        return "اطلاعات الزامی درخواست کامل نیست"
    if "invalid" in lowered or "incorrect" in lowered:
        return "اطلاعات ارسال‌شده معتبر نیست"
    if status_code == 401:
        return "برای انجام این درخواست باید وارد حساب کاربری شوید"
    if status_code == 403:
        return "اجازه انجام این درخواست را ندارید"
    if status_code == 404:
        return "مسیر یا منبع درخواستی یافت نشد"
    if status_code == 405:
        return "روش درخواست برای این مسیر مجاز نیست"
    if status_code == 409:
        return "درخواست با وضعیت فعلی سامانه تداخل دارد"
    if status_code == 422:
        return "اطلاعات ارسال‌شده معتبر نیست"
    return "انجام درخواست با خطا مواجه شد"


def validation_error_message(error_type: str, context: dict[str, Any] | None = None) -> str:
    context = context or {}
    if error_type == "missing":
        return "این فیلد الزامی است"
    if error_type in {"json_invalid", "json_type"}:
        return "بدنه JSON معتبر نیست"
    if error_type.startswith("string_too_short"):
        return f"متن باید حداقل {context.get('min_length', 1)} نویسه باشد"
    if error_type.startswith("string_too_long"):
        return f"متن نباید بیشتر از {context.get('max_length', '')} نویسه باشد"
    if error_type in {"greater_than", "greater_than_equal"}:
        return "مقدار از حداقل مجاز کمتر است"
    if error_type in {"less_than", "less_than_equal"}:
        return "مقدار از حداکثر مجاز بیشتر است"
    if error_type in {"enum", "literal_error"}:
        return "مقدار انتخاب‌شده مجاز نیست"
    if error_type.endswith("_parsing") or error_type.endswith("_type"):
        return "نوع یا قالب مقدار معتبر نیست"
    return "مقدار ارسال‌شده معتبر نیست"


def localize_error_detail(detail: Any, *, status_code: int | None = None) -> Any:
    if isinstance(detail, str):
        return public_error_message(detail, status_code=status_code)
    if isinstance(detail, list):
        return [localize_error_detail(item, status_code=status_code) for item in detail]
    if isinstance(detail, dict):
        localized = dict(detail)
        for key in ("detail", "message", "msg", "reason", "failure_message", "error_message"):
            if key in localized:
                localized[key] = localize_error_detail(localized[key], status_code=status_code)
        return localized
    return detail


def localize_frontend_payload(value: Any, *, human_context: bool = False) -> Any:
    if isinstance(value, str):
        return public_error_message(value) if human_context else value
    if isinstance(value, list):
        return [localize_frontend_payload(item, human_context=human_context) for item in value]
    if isinstance(value, dict):
        return {
            key: localize_frontend_payload(
                item,
                human_context=key in {
                    "detail",
                    "message",
                    "msg",
                    "failure_message",
                    "error_message",
                    "processing_error",
                    "last_error",
                    "guidance",
                    "error",
                },
            )
            for key, item in value.items()
        }
    return value


class LocalizedJSONRoute(APIRoute):
    def get_route_handler(self):
        original_handler = super().get_route_handler()

        async def localized_handler(request: Request) -> Response:
            response = await original_handler(request)
            content_type = response.headers.get("content-type", "")
            body = getattr(response, "body", None)
            if "application/json" not in content_type or not isinstance(body, bytes):
                return response
            try:
                payload = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                return response
            localized = localize_frontend_payload(payload)
            if localized == payload:
                return response
            headers = dict(response.headers)
            headers.pop("content-length", None)
            headers.pop("content-type", None)
            return JSONResponse(
                content=localized,
                status_code=response.status_code,
                headers=headers,
                background=response.background,
            )

        return localized_handler


def install_frontend_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def localized_http_exception_handler(
        _: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": localize_error_detail(exc.detail, status_code=exc.status_code)},
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def localized_validation_exception_handler(
        _: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        errors = []
        for error in exc.errors():
            localized = dict(error)
            localized["msg"] = validation_error_message(
                str(error.get("type", "")),
                error.get("ctx"),
            )
            errors.append(localized)
        return JSONResponse(status_code=422, content=jsonable_encoder({"detail": errors}))

    @app.exception_handler(Exception)
    async def localized_unhandled_exception_handler(
        _: Request,
        exc: Exception,
    ) -> JSONResponse:
        LOGGER.exception("Unhandled request error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "خطای داخلی در سرویس رخ داد؛ دوباره تلاش کنید"},
        )
