from sqlalchemy.orm import Session
from . import models, schemas
from sqlalchemy import func

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()

def create_user(db: Session, user: schemas.UserCreate):
    db_user = models.User()
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_transactions(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Transaction).offset(skip).limit(limit).all()

def create_transaction(db: Session, transaction: schemas.TransactionCreate, account_id: int):
    db_transaction = models.Transaction(**transaction.dict(), account_id=account_id)
    db.add(db_transaction)
    db.commit()
    db.refresh(db_transaction)
    return db_transaction

def get_sankey_data(db: Session):
    nodes = []
    links = []
    node_map = {}

    # Create an income node
    income_node_name = "Income"
    node_map[income_node_name] = len(nodes)
    nodes.append({"name": income_node_name})

    # Get all expenses by category
    expense_categories = db.query(models.Category.name, func.sum(models.Transaction.amount_minor).label('total')) \
        .join(models.TxnCategory, models.TxnCategory.category_id == models.Category.id) \
        .join(models.Transaction, models.Transaction.id == models.TxnCategory.txn_id) \
        .filter(models.Transaction.amount_minor < 0) \
        .group_by(models.Category.name) \
        .all()

    for category_name, total_amount in expense_categories:
        if category_name not in node_map:
            node_map[category_name] = len(nodes)
            nodes.append({"name": category_name})
        
        links.append({
            "source": node_map[income_node_name],
            "target": node_map[category_name],
            "value": abs(total_amount)
        })
        
    return {"nodes": nodes, "links": links}

def get_uncategorized_transactions(db: Session, skip: int = 0, limit: int = 100):
    # For now, just return all transactions
    return db.query(models.Transaction).offset(skip).limit(limit).all()
