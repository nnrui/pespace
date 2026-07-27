class JaxImportError(ImportError):
    """Exception raised when jax is required but not installed.

    jax is not a default dependency of pespace. It need to be installed manually
    depending on the target hardware (CPU or GPU).
    """

    _MESSAGE = (
        "jax is required for backend='jax', but it is not installed by default.\n"
        "Install jax manually according to your hardware.\n"
    )

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message if message is not None else self._MESSAGE)
