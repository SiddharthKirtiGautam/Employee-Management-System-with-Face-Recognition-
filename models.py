from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Database URL (SQLite for simplicity)
DATABASE_URL = 'sqlite:///employees.db'

# SQLAlchemy setup
engine = create_engine(DATABASE_URL, echo=True)
Base = declarative_base()

# Define the Employee model
class Employee(Base):
    __tablename__ = 'employees'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    embeddings = relationship('Embedding', back_populates='employee', cascade="all, delete-orphan")

# Define the Embedding model
class Embedding(Base):
    __tablename__ = 'embeddings'
    id = Column(Integer, primary_key=True)
    embedding = Column(LargeBinary, nullable=False)
    employee_id = Column(Integer, ForeignKey('employees.id'))
    employee = relationship('Employee', back_populates='embeddings')

# Create tables in the database
Base.metadata.create_all(engine)

# Session factory
Session = sessionmaker(bind=engine)
session = Session()
