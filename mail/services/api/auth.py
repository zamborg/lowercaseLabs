from fastapi import Header, HTTPException, Request, status


def require_auth(request: Request, authorization: str | None = Header(default=None)) -> str:
    settings = request.app.state.settings
    if not authorization or authorization != f"Bearer {settings.dev_token}":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized")
    return request.app.state.dev_user_id
