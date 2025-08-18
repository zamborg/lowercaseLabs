from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware

from . import crud, models, schemas
from .database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    return crud.create_user(db=db, user=user)


@app.get("/users/", response_model=list[schemas.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users


@app.get("/transactions/", response_model=list[schemas.Transaction])
def read_transactions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    transactions = crud.get_transactions(db, skip=skip, limit=limit)
    return transactions


@app.post("/accounts/{account_id}/transactions/", response_model=schemas.Transaction)
def create_transaction_for_account(
    account_id: int, transaction: schemas.TransactionCreate, db: Session = Depends(get_db)
):
    return crud.create_transaction(db=db, transaction=transaction, account_id=account_id)

@app.get("/sankey/")
def get_sankey_data(db: Session = Depends(get_db)):
    return crud.get_sankey_data(db)

@app.get("/transactions/uncategorized", response_model=list[schemas.Transaction])
def read_uncategorized_transactions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    transactions = crud.get_uncategorized_transactions(db, skip=skip, limit=limit)
    return transactions
