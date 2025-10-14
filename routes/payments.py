from datetime import datetime, timezone
from typing import Optional
import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from pydantic import BaseModel
from db import get_session
from models import Order, User
from utils.users import get_current_user
from models import SingleItemResponse
from dotenv import load_dotenv
import os

load_dotenv()

router = APIRouter(prefix="/payments", tags=["payments"])

# Pi Network API Configuration
PI_API_KEY = os.environ.get("PI_API_KEY")
PI_API_BASE_URL = os.environ.get("PLATFORM_API_URL")


# Request/Response Models
class IncompletePaymentRequest(BaseModel):
    payment: dict


class ApprovePaymentRequest(BaseModel):
    paymentId: str


class CompletePaymentRequest(BaseModel):
    paymentId: str
    txid: str


class CancelledPaymentRequest(BaseModel):
    paymentId: str


# Helper function for Pi API calls
async def pi_api_client(method: str, endpoint: str, data: Optional[dict] = None):
    """Helper function to make requests to Pi Network API"""
    headers = {"Authorization": f"Key {PI_API_KEY}"}
    url = f"{PI_API_BASE_URL}{endpoint}"

    async with httpx.AsyncClient(timeout=20.0) as client:
        if method.upper() == "GET":
            response = await client.get(url, headers=headers)
        elif method.upper() == "POST":
            response = await client.post(url, headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        return response.json()


@router.post("/incomplete")
async def handle_incomplete_payment(
    body: IncompletePaymentRequest,
    session: Session = Depends(get_session),
):
    """
    Handle incomplete payments by verifying them on the blockchain
    and marking orders as paid
    """
    payment = body.payment
    payment_id = payment.get("identifier")
    transaction = payment.get("transaction", {})
    txid = transaction.get("txid")
    tx_url = transaction.get("_link")

    if not payment_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payment identifier is required",
        )

    # Find the incomplete order
    statement = select(Order).where(Order.pi_payment_id == payment_id)
    order = session.exec(statement).first()

    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Order not found"
        )

    # Check the transaction on the Pi blockchain
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            horizon_response = await client.get(tx_url)
            horizon_response.raise_for_status()
            horizon_data = horizon_response.json()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to verify transaction: {str(e)}",
        )

    payment_id_on_block = horizon_data.get("memo")

    # Verify payment id matches
    if payment_id_on_block != order.pi_payment_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payment id doesn't match blockchain record",
        )

    # You can add additional verification here (amount, recipient, etc.)

    # Mark the order as paid
    order.txid = txid
    order.paid = True
    order.updated_at = datetime.now(timezone.utc)
    session.add(order)
    session.commit()

    # Let Pi Servers know that the payment is completed
    try:
        await pi_api_client(
            "POST", f"/v2/payments/{payment_id}/complete", {"txid": txid}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to complete payment with Pi: {str(e)}",
        )

    return SingleItemResponse(
        message=f"Handled the incomplete payment {payment_id}", data=order
    )


@router.post("/approve")
async def approve_payment(
    body: ApprovePaymentRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Approve a payment and create an order record
    """
    payment_id = body.paymentId

    if not payment_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Payment ID is required"
        )

    # Get payment details from Pi Network
    try:
        current_payment = await pi_api_client("GET", f"/v2/payments/{payment_id}")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch payment from Pi: {str(e)}",
        )

    # Check if order already exists
    existing_order = session.exec(
        select(Order).where(Order.pi_payment_id == payment_id)
    ).first()

    if existing_order:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Order already exists for this payment",
        )

    # Create order record
    metadata = current_payment.get("metadata", {})
    order = Order(
        pi_payment_id=payment_id,
        product_id=metadata.get("productId"),
        user_id=current_user.id,
    )

    session.add(order)
    session.commit()
    session.refresh(order)

    # Approve the payment with Pi Network
    try:
        await pi_api_client("POST", f"/v2/payments/{payment_id}/approve")
    except Exception as e:
        # Rollback order creation if Pi approval fails
        session.delete(order)
        session.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to approve payment with Pi: {str(e)}",
        )

    return SingleItemResponse(message=f"Approved the payment {payment_id}", data=order)


@router.post("/complete")
async def complete_payment(
    body: CompletePaymentRequest,
    session: Session = Depends(get_session),
):
    """
    Complete a payment after transaction verification
    """
    payment_id = body.paymentId
    txid = body.txid

    if not payment_id or not txid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payment ID and transaction ID are required",
        )

    # Find the order
    statement = select(Order).where(Order.pi_payment_id == payment_id)
    order = session.exec(statement).first()

    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Order not found"
        )

    if order.paid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Order is already marked as paid",
        )

    # Update order status
    order.txid = txid
    order.paid = True
    order.updated_at = datetime.now(timezone.utc)
    session.add(order)
    session.commit()

    # Notify Pi Network
    try:
        await pi_api_client(
            "POST", f"/v2/payments/{payment_id}/complete", {"txid": txid}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to complete payment with Pi: {str(e)}",
        )

    return SingleItemResponse(message=f"Completed the payment {payment_id}", data=order)


@router.post("/cancelled")
async def handle_cancelled_payment(
    body: CancelledPaymentRequest,
    session: Session = Depends(get_session),
):
    """
    Handle cancelled payments
    """
    payment_id = body.paymentId

    if not payment_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Payment ID is required"
        )

    # Find the order
    statement = select(Order).where(Order.pi_payment_id == payment_id)
    order = session.exec(statement).first()

    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Order not found"
        )

    # Mark order as cancelled
    order.cancelled = True
    order.updated_at = datetime.now(timezone.utc)
    session.add(order)
    session.commit()

    return SingleItemResponse(message=f"Cancelled the payment {payment_id}", data=order)
