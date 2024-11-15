import os
from typing import Any, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.orm.scoping import ScopedSessionMixin
from sqlalchemy.orm.session import Session


DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "root")
DB_NAME = os.environ.get("DB_NAME", "rewater_mysql")
DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_PORT = os.environ.get("DB_PORT", "3305")

DATABASE = "mysql://%s:%s@%s:%s/%s?charset=utf8" % (
    DB_USER,
    DB_PASSWORD,
    DB_HOST,
    DB_PORT,
    DB_NAME,
)
print(f"=========DATABASE: {DATABASE}============")

engine = create_engine(
    DATABASE,
    encoding="utf-8",
    pool_size=50,
    max_overflow=50,
    pool_pre_ping=True,
    # echo=True
)

SessionLocal = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
)

Base = declarative_base()
Base.query = SessionLocal.query_property()


# Dependency
def get_db():
    db = None
    try:
        db = SessionLocal()
        yield db
    finally:
        if db:
            db.close()
