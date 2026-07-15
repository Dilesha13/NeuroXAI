from app.db.session import Base, engine
from app.db.models.patient import Patient
from app.db.models.eeg_record import EEGRecord
from app.db.models.inference_result import InferenceResult
from app.db.models.explanation import Explanation
from app.db.models.report import Report
from app.db.models.user import User


def init_db():
    Base.metadata.create_all(bind=engine)