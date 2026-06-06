from sqlalchemy.exc import OperationalError

from receipt_extractor.services.database import Base, engine

try:
    Base.metadata.create_all(bind=engine)
    print("✅ Tabellen erstellt oder existieren bereits")
except OperationalError as e:
    print("⚠️ DB nicht erreichbar:", e)
