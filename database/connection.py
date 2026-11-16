# database/connection.py
# Kết nối cơ sở dữ liệu Microsoft SQL Server (SSMS) qua pyodbc & SQLAlchemy
import os
import urllib.parse
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cấu hình từ .env
DB_TYPE = os.getenv("DB_TYPE", "mssql").lower()
MSSQL_SERVER = os.getenv("MSSQL_SERVER", r"localhost\SQLEXPRESS")
MSSQL_DATABASE = os.getenv("MSSQL_DATABASE", "traffic_ai_db")
MSSQL_DRIVER = os.getenv("MSSQL_DRIVER", "ODBC Driver 17 for SQL Server")
MSSQL_TRUSTED = os.getenv("MSSQL_TRUSTED_CONNECTION", "yes").lower() == "yes"
MSSQL_USER = os.getenv("MSSQL_USER", "")
MSSQL_PASSWORD = os.getenv("MSSQL_PASSWORD", "")

engine = None
active_db_type = "sqlite"

if DB_TYPE in ["mssql", "sqlserver"]:
    try:
        # Xây dựng chuỗi kết nối ODBC cho Microsoft SQL Server (SSMS)
        if MSSQL_TRUSTED:
            # Dùng Windows Authentication (mặc định của SSMS trên máy cục bộ)
            conn_str = (
                f"DRIVER={{{MSSQL_DRIVER}}};"
                f"SERVER={MSSQL_SERVER};"
                f"DATABASE={MSSQL_DATABASE};"
                f"Trusted_Connection=yes;"
                f"TrustServerCertificate=yes;"
            )
        else:
            # Dùng tài khoản SQL Server Authentication (ví dụ sa)
            conn_str = (
                f"DRIVER={{{MSSQL_DRIVER}}};"
                f"SERVER={MSSQL_SERVER};"
                f"DATABASE={MSSQL_DATABASE};"
                f"UID={MSSQL_USER};"
                f"PWD={MSSQL_PASSWORD};"
                f"TrustServerCertificate=yes;"
            )

        params = urllib.parse.quote_plus(conn_str)
        engine = create_engine(
            f"mssql+pyodbc:///?odbc_connect={params}",
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

        with engine.connect() as conn:
            db_name = conn.execute(text("SELECT DB_NAME()")).scalar()
            print(f"✅ Đã kết nối thành công tới Microsoft SQL Server (SSMS): `{db_name}` trên `{MSSQL_SERVER}`")
        active_db_type = "mssql"

    except Exception as e:
        print(f"⚠️ Không thể kết nối SQL Server ({e}).")
        print("🔄 Tự động chuyển sang sử dụng SQLite dự phòng...")
        engine = None

if engine is None:
    sqlite_path = os.getenv("SQLITE_PATH", os.path.join(BASE_DIR, "traffic_ai.db"))
    engine = create_engine(
        f"sqlite:///{sqlite_path}",
        connect_args={"check_same_thread": False},
    )
    active_db_type = "sqlite"
    print(f"ℹ️ Đang sử dụng SQLite: {sqlite_path}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
class Base(DeclarativeBase):
    """Lớp gốc của các model ORM (cú pháp SQLAlchemy 2.0: Mapped[...] + mapped_column)."""


def get_db():
    """Dependency cung cấp db session cho mỗi request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
