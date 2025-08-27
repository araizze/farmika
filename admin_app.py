from fastapi import FastAPI
from sqladmin import Admin, ModelView
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String

DATABASE_URL = "sqlite+aiosqlite:///db.sqlite3"

engine = create_async_engine(DATABASE_URL, echo=True)
Base = declarative_base()

class Dataset(Base):
    __tablename__ = "dataset"
    id = Column(Integer, primary_key=True)
    filename = Column(String)
    uploaded_at = Column(String)

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    prompt = Column(String)
    response = Column(String)
    label = Column(String)
    created_at = Column(String)

# Класс, а не объект!
class DatasetAdmin(ModelView, model=Dataset):
    column_list = [Dataset.id, Dataset.filename, Dataset.uploaded_at]

class FeedbackAdmin(ModelView, model=Feedback):
    column_list = [
        Feedback.id, Feedback.user_id, Feedback.prompt, Feedback.response, Feedback.label, Feedback.created_at
    ]

app = FastAPI()

admin = Admin(app, engine)
admin.add_view(DatasetAdmin)
admin.add_view(FeedbackAdmin)

@app.get("/")
def root():
    return {"msg": "SQLAdmin работает!"}


# python -m uvicorn admin_app:app --reload --port 8001
