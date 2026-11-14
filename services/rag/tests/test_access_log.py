import asyncio
import logging

from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from rag_common.service import create_app


class Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.fields: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.fields.append(record.fields)


def test_access_log_times_whole_streams_and_records_crashes():
    app = create_app("test")

    @app.get("/stream")
    async def stream():
        async def body():
            yield b"first "
            await asyncio.sleep(0.3)
            yield b"last"
        return StreamingResponse(body())

    @app.get("/boom")
    async def boom():
        raise RuntimeError("bug")

    records = Records()
    logging.getLogger("access").addHandler(records)
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            assert client.get("/stream", headers={"x-request-id": "r1"}).headers["x-request-id"] == "r1"
            assert client.get("/boom").status_code == 500
    finally:
        logging.getLogger("access").removeHandler(records)
    streamed, crashed = records.fields
    assert streamed["status"] == 200 and streamed["ms"] >= 300  # timed to the last byte, not the headers
    assert (crashed["path"], crashed["status"], crashed["error"]) == ("/boom", 500, "RuntimeError")
