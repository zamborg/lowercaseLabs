from services.api.settings import load_settings


def main() -> None:
    settings = load_settings()
    import uvicorn

    uvicorn.run(
        "services.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
