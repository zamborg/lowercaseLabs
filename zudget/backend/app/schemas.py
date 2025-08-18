from pydantic import BaseModel
from typing import List, Optional
import datetime

class TransactionBase(BaseModel):
    posted_at: datetime.datetime
    amount_minor: int
    currency: str
    merchant_name: str
    description_raw: str
    is_pending: bool
    provider_txn_id: str
    hash_dedup: str

class TransactionCreate(TransactionBase):
    pass

class Transaction(TransactionBase):
    id: int
    account_id: int

    class Config:
        orm_mode = True

class AccountBase(BaseModel):
    name: str
    provider: str
    type: str
    currency: str
    is_hidden: bool

class AccountCreate(AccountBase):
    pass

class Account(AccountBase):
    id: int
    user_id: int
    transactions: List[Transaction] = []

    class Config:
        orm_mode = True

class UserBase(BaseModel):
    pass

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int
    accounts: List[Account] = []

    class Config:
        orm_mode = True

class CategoryBase(BaseModel):
    name: str
    type: str
    parent_id: Optional[int] = None

class CategoryCreate(CategoryBase):
    pass

class Category(CategoryBase):
    id: int

    class Config:
        orm_mode = True

class TagBase(BaseModel):
    name: str

class TagCreate(TagBase):
    pass

class Tag(TagBase):
    id: int
    user_id: int

    class Config:
        orm_mode = True
