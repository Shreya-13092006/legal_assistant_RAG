"""FastAPI application factory. Run: `uvicorn lawrag.api.app:create_app --factory --port 8000`."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from lawrag.api.deps import set_state
from lawrag.api.routes import router
from lawrag.config import load_config
from lawrag.errors import LawRagError
from lawrag.logging_setup import request_id_var, setup_logging
from lawrag.services.container import Deps, build_deps


def _error(code: str, message: str, status: int, details: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message, "request_id": request_id_var.get(),
                                   "details": details or {}}}, status_code=status)


def create_app(deps: Deps | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        d = deps
        if d is None:
            overlays = [o for o in os.environ.get("LAWRAG_CONFIG", "").split(",") if o]
            cfg = load_config(overlays)
            setup_logging(cfg.logging.level, cfg.logging.json_format)
            d = build_deps(cfg)
        set_state(d)
        yield

    app = FastAPI(title="lawrag — IPC ↔ BNS Graph RAG", version="0.1.0", lifespan=lifespan,
                  description="Cited, verified answers about the IPC and the BNS. Not legal advice.")
    app.add_middleware(CORSMiddleware, allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
                       allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(LawRagError)
    async def _lawrag(_: Request, exc: LawRagError):
        return _error(exc.code, exc.message or str(exc), exc.http_status, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        errs = [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
        return _error("VALIDATION_ERROR", "invalid request", 422, {"errors": errs})

    app.include_router(router)
    if deps is not None:
        set_state(deps)
    return app
