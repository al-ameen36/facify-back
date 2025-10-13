from typing import Annotated, List
from datetime import timedelta
from fastapi import (
    BackgroundTasks,
    Depends,
    APIRouter,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
)
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session, select
from models import (
    MediaUsageType,
    SingleItemResponse,
    Token,
    User,
    UserRead,
    UserCreate,
    RefreshTokenRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
)
from models.core import ContentOwnerType
from models.media import Media, MediaRead, MediaUsage
from tasks.face import embed_media
from utils.media import delete_media_and_file, save_file_to_db, upload_file
from utils.users import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    get_current_user,
    verify_refresh_token,
    revoke_refresh_token,
    get_user_by_username,
    get_user_by_email,
    create_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    verify_password_reset_token,
    update_user_password,
    send_password_reset_email,
)
from db import get_session
from utils.users import send_verification_email

router = APIRouter(prefix="/user", tags=["user"])


@router.post("/register", response_model=SingleItemResponse[UserRead])
async def register_user(
    user_data: UserCreate,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    if get_user_by_username(session, user_data.username):
        raise HTTPException(status_code=400, detail="Username already registered")

    if get_user_by_email(session, user_data.email):
        raise HTTPException(status_code=400, detail="Email already registered")

    user = create_user(
        session=session,
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name,
        password=user_data.password,
    )

    token = create_access_token(user.username, expires_delta=timedelta(minutes=30))
    background_tasks.add_task(send_verification_email, user.email, token)

    return SingleItemResponse(
        data=UserRead.model_validate(user), message="User registered successfully"
    )


@router.post("/face", response_model=SingleItemResponse[list[MediaRead]])
async def upload_face_capture(
    files: List[UploadFile] = File(...),
    angles: List[str] = Form(...),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Upload multiple face images with corresponding angles.
    Replaces any existing image for that same angle.
    """

    if len(files) != len(angles):
        raise HTTPException(400, "Each uploaded file must have a matching angle")

    uploaded_media_list = []

    try:
        for file, angle in zip(files, angles):
            # Determine usage type
            usage_type = (
                MediaUsageType.PROFILE_PICTURE
                if angle == "center"
                else MediaUsageType.PROFILE_PICTURE_ANGLE
            )

            # Find existing media for same user + angle
            existing_medias = session.exec(
                select(Media)
                .join(MediaUsage)
                .where(
                    MediaUsage.owner_type == ContentOwnerType.USER,
                    MediaUsage.owner_id == current_user.id,
                    MediaUsage.usage_type == usage_type,
                    Media.tags == angle,
                )
            ).all()

            # Delete each safely
            for old_media in existing_medias:
                delete_media_and_file(session, old_media)
            session.commit()

            # Upload new file
            uploaded = upload_file(
                file=file,
                user=current_user,
                owner_id=current_user.id,
                owner_type=ContentOwnerType.USER,
                usage_type=usage_type,
            )

            # Save in DB with tag = angle
            saved_media = save_file_to_db(
                session=session,
                media=uploaded,
                owner_id=current_user.id,
                owner_type=ContentOwnerType.USER,
                usage_type=usage_type,
                embeddings=None,
                user_id=current_user.id,
                approval_status="approved",
                tags=angle,
            )
            uploaded_media_list.append(saved_media.to_media_read())

            # Trigger embedding generation (async)
            embed_media.delay(saved_media.id, saved_media.external_url)

        return SingleItemResponse(
            data=uploaded_media_list,
            message=f"Successfully uploaded {len(uploaded_media_list)} face captures (replacing existing ones if any).",
        )

    except Exception as e:
        session.rollback()
        raise HTTPException(500, f"Unexpected error: {str(e)}")


@router.get("/verify-email")
async def verify_email(token: str, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.verification_token == token)).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid verification token")

    user.is_verified = True
    user.verification_token = None
    session.add(user)
    session.commit()

    return {"message": "Email verified successfully"}


@router.post("/login", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: Session = Depends(get_session),
    response: Response = None,  # inject FastAPI response
):
    user = authenticate_user(session, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(user.username)
    refresh_token = create_refresh_token(user.username)

    user_data = user.to_user_read(session)

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_data,
    )


@router.post("/refresh", response_model=Token)
async def refresh_access_token(
    refresh_data: RefreshTokenRequest, session: Session = Depends(get_session)
):
    username = verify_refresh_token(refresh_data.refresh_token)
    if not username:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_username(session, username)
    if not user:
        revoke_refresh_token(refresh_data.refresh_token)
        raise HTTPException(status_code=401, detail="User not found")

    new_access_token = create_access_token(user.username)

    return Token(
        access_token=new_access_token,
        refresh_token=refresh_data.refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user,
    )


@router.post("/logout")
async def logout_user(refresh_data: RefreshTokenRequest):
    revoke_refresh_token(refresh_data.refresh_token)
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserRead)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Session = Depends(get_session),
):
    return current_user.to_user_read(session)


@router.post("/forgot-password")
async def forgot_password(
    request_data: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    user = get_user_by_email(session, request_data.email)
    if not user:
        # Don't reveal user existence
        return {"message": "If that email exists, a reset link has been sent."}

    token = create_access_token(user.username, expires_delta=timedelta(15))
    background_tasks.add_task(send_password_reset_email, user.email, token)

    return {"message": "If that email exists, a reset link has been sent."}


@router.post("/reset-password")
async def reset_password(
    reset_data: ResetPasswordRequest, session: Session = Depends(get_session)
):
    username = verify_password_reset_token(reset_data.token)
    if not username:
        raise HTTPException(status_code=400, detail="Invalid or expired token")

    user = get_user_by_username(session, username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    update_user_password(session, user, reset_data.new_password)
    return {"message": "Password updated successfully"}
