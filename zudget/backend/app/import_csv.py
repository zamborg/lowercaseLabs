import csv
from sqlalchemy.orm import Session
from . import crud, models, schemas
from .database import SessionLocal, engine

import datetime
import hashlib

def get_or_create_account(db: Session, account_name: str, user_id: int):
    account = db.query(models.Account).filter(models.Account.name == account_name).first()
    if not account:
        account = models.Account(name=account_name, provider="CSV", type="checking", currency="USD", user_id=user_id)
        db.add(account)
        db.commit()
        db.refresh(account)
    return account

def get_or_create_category(db: Session, category_name: str, category_type: str):
    category = db.query(models.Category).filter(models.Category.name == category_name).first()
    if not category:
        category = models.Category(name=category_name, type=category_type)
        db.add(category)
        db.commit()
        db.refresh(category)
    return category

def import_transactions_from_csv(db: Session, filepath: str):
    # Create a default user if one doesn't exist
    users = crud.get_users(db)
    if not users:
        user = crud.create_user(db, schemas.UserCreate())
    else:
        user = users[0]

    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Data cleaning and transformation
            amount_minor = int(float(row['Amount']) * 100)
            posted_at = datetime.datetime.strptime(row['Date'], '%Y-%m-%d')
            merchant_name = row['Description']
            
            # Create a unique ID for the transaction
            unique_string = f"{row['Date']}-{merchant_name}-{amount_minor}-{row['Account']}"
            provider_txn_id = hashlib.md5(unique_string.encode()).hexdigest()
            hash_dedup = provider_txn_id

            # Get or create account
            account = get_or_create_account(db, row['Account'], user.id)

            # Get or create category
            category_name = row['Category']
            category_type = 'income' if amount_minor > 0 else 'expense'
            category = get_or_create_category(db, category_name, category_type)

            # Create transaction
            transaction_data = schemas.TransactionCreate(
                posted_at=posted_at,
                amount_minor=amount_minor,
                currency="USD",
                merchant_name=merchant_name,
                description_raw=merchant_name,
                is_pending=False,
                provider_txn_id=provider_txn_id,
                hash_dedup=hash_dedup,
            )
            db_transaction = crud.create_transaction(db, transaction_data, account.id)
            
            # Associate category with transaction
            txn_category = models.TxnCategory(
                txn_id=db_transaction.id,
                category_id=category.id,
                source="csv"
            )
            db.add(txn_category)
            db.commit()


if __name__ == "__main__":
    models.Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    # The user will need to provide the path to the csv file
    # For now, I will assume it is in the data directory
    import_transactions_from_csv(db, "data/transactions.csv")
    db.close()
