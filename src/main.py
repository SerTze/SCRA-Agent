"""Entry point – ``uvicorn src.main:create_app --factory`` (preferred).

Non-factory mode (``uvicorn src.main:app``) also works via a lazy
module-level attribute that only creates the app on first access,
avoiding the double-initialization waste of eager ``app = create_app()``.
"""

from src.presentation.api import create_app


def __getattr__(name: str):
    """Lazy module attribute – avoids creating a wasted Container in factory mode."""
    if name == "app":
        global app  # noqa: PLW0603
        app = create_app()
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
