"""Script 09: Purge expired uploaded documents, chunks, and database rows."""

import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag.core.logging import setup_logging
from legalrag.sessions.cleanup import cleanup_expired_uploads


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    setup_logging(level="INFO")
    print("--- Running Upload Cleanup Task ---")
    stats = cleanup_expired_uploads()
    print("Xóa thành công:")
    print(f"  - Số tài liệu upload hết hạn: {stats['deleted_uploads']}")
    print(f"  - Số parent chunks hết hạn   : {stats['deleted_parents']}")
    print("--- Cleanup Finished ---")


if __name__ == "__main__":
    main()
