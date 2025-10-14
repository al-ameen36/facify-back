from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel, Relationship

from models.core import AppBaseModel


class OrderBase(SQLModel):
    pi_payment_id: str = Field(index=True, unique=True, nullable=False)
    product_id: Optional[str] = None
    user_id: int = Field(foreign_key="user.id")
    txid: Optional[str] = None
    paid: bool = Field(default=False)
    cancelled: bool = Field(default=False)


class Order(OrderBase, AppBaseModel, table=True):
    """Order table model"""

    id: Optional[int] = Field(default=None, primary_key=True)


class OrderCreate(OrderBase):
    """Schema for creating an order"""

    pass


class OrderRead(OrderBase):
    """Schema for reading an order"""

    id: int
    created_at: datetime
    updated_at: datetime


class OrderUpdate(SQLModel):
    """Schema for updating an order"""

    txid: Optional[str] = None
    paid: Optional[bool] = None
    cancelled: Optional[bool] = None
