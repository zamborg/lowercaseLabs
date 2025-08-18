from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, Float, JSON
from sqlalchemy.orm import relationship
from .database import Base
import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    # Add other user fields as needed from the spec

class Account(Base):
    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    provider = Column(String)
    type = Column(String)
    name = Column(String)
    currency = Column(String)
    is_hidden = Column(Boolean, default=False)

    user = relationship("User", back_populates="accounts")

User.accounts = relationship("Account", order_by=Account.id, back_populates="user")

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id"))
    posted_at = Column(DateTime, default=datetime.datetime.utcnow)
    amount_minor = Column(Integer)
    currency = Column(String)
    merchant_name = Column(String)
    merchant_id = Column(String, nullable=True)
    description_raw = Column(String)
    counterparty = Column(String, nullable=True)
    is_pending = Column(Boolean, default=False)
    provider_txn_id = Column(String, unique=True)
    hash_dedup = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    account = relationship("Account", back_populates="transactions")

Account.transactions = relationship("Transaction", order_by=Transaction.id, back_populates="account")

class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("categories.id"), nullable=True)
    name = Column(String)
    type = Column(String) # income, expense, transfer, savings

class Tag(Base):
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)

class TxnCategory(Base):
    __tablename__ = "txn_categories"

    id = Column(Integer, primary_key=True, index=True)
    txn_id = Column(Integer, ForeignKey("transactions.id"))
    category_id = Column(Integer, ForeignKey("categories.id"))
    source = Column(String) # ai, rule, user
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class TxnTag(Base):
    __tablename__ = "txn_tags"

    txn_id = Column(Integer, ForeignKey("transactions.id"), primary_key=True)
    tag_id = Column(Integer, ForeignKey("tags.id"), primary_key=True)

class Rule(Base):
    __tablename__ = "rules"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    match_json = Column(JSON)
    action_json = Column(JSON)
    priority = Column(Integer)
    enabled = Column(Boolean, default=True)
