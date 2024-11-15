import sys, os

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from sqlalchemy import text
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

# init Scheduler
scheduler = BackgroundScheduler()


# define scheduler task
def scheduled_task():
    logger.debug(f"[scheduled_task] start at: {datetime.now()}")
    from app.db.database import engine

    with engine.connect() as connection:
        connection.execute(text("TRUNCATE TABLE image_result"))
        connection.execute(text("TRUNCATE TABLE object_predicted"))

        connection.commit()
        logger.debug(f"[scheduled_task][TRUNCATE] Done")

    static_folder = "static/images"
    try:
        for filename in os.listdir(static_folder):
            file_path = os.path.join(static_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                logger.debug(f"Deleted file: {file_path}")
        logger.debug(f"All files in '{static_folder}' deleted at {datetime.now()}")
    except Exception as e:
        logger.debug(f"Error while deleting files: {e}")


# run job
scheduler.add_job(scheduled_task, "cron", hour=12, minute=0)

# start scheduler
scheduler.start()


@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()


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

# serve static
app.mount("/static", StaticFiles(directory="static"), name="static")


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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, env_file=".env")

    print(f"config.DEBUG: {config.DEBUG}")
    if config.DEBUG:
        # if os.environ.get("RUN_MAIN") or os.environ.get("WERKZEUG_RUN_MAIN"):
        import ptvsd

        ptvsd.enable_attach(address=("0.0.0.0", 3030))
        print("Attached!")

    print(f"app is running on 8000")
