"""Start the console API on 127.0.0.1:8000."""

import uvicorn

from policyjev.api.app import create_app

HOST = "127.0.0.1"
PORT = 8000


def main() -> None:
    """Run uvicorn bound to the loopback interface only."""
    uvicorn.run(create_app(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
