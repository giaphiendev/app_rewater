import sys

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.routers import routers
from app.core import config

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | {level} | <level>{message}</level>",
)


async def logging_dependency(request: Request):
    logger.debug(f"{request.method} {request.url}")
    # logger.debug("Params:")
    # for name, value in request.path_params.items():
    #     logger.debug(f"\t{name}: {value}")
    # logger.debug("Headers:")
    # for name, value in request.headers.items():
    #     logger.debug(f"\t{name}: {value}")


app = FastAPI(title="Tracking API", debug=config.DEBUG)

# CORS
origins = []

# Set all CORS enabled origins
if config.BACKEND_CORS_ORIGINS:
    origins_raw = config.BACKEND_CORS_ORIGINS.split(",")
    for origin in origins_raw:
        use_origin = origin.strip()
        origins.append(use_origin)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.exception_handler(Exception)
async def unicorn_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": f"{exc.detail}", "code": exc.status_code, "data": None},
    )


app.include_router(
    routers.router, prefix="/api", dependencies=[Depends(logging_dependency)]
)


@app.get("/")
def health():
    print("health endpoint")
    return JSONResponse(
        status_code=200,
        content={"message": "OK", "data": None},
    )


if __name__ == "__main__":
    # if config.DEBUG:
    #     if os.environ.get("RUN_MAIN") or os.environ.get("WERKZEUG_RUN_MAIN"):
    #         import ptvsd

    #         ptvsd.enable_attach(address=("0.0.0.0", 3000))
    #         print("Attached!")

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, env_file=".env")
