"""Optional external image-analysis integration."""

import asyncio
import os
import tempfile
from pathlib import Path


def _run(coroutine):
    return asyncio.run(coroutine)


async def _analyze_file(path):
    from realitydefender import RealityDefender

    client = RealityDefender(api_key=os.environ["REALITY_DEFENDER_API_KEY"])
    try:
        upload = await client.upload(file_path=str(path))
        result = await client.get_result(upload["request_id"])
        result["request_id"] = upload["request_id"]
        return result
    finally:
        await client.cleanup()


def analyze_image(image_bytes, filename="upload.jpg"):
    """Return an RD result, or an unavailable result when RD is not configured."""
    if not os.environ.get("REALITY_DEFENDER_API_KEY"):
        return {"status": "unavailable", "message": "External detection service is not configured."}

    suffix = Path(filename).suffix.lower() or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as file:
        file.write(image_bytes)
        file.flush()
        result = _run(_analyze_file(Path(file.name)))

    return {
        "status": result.get("status"),
        "score": result.get("score"),
        "models": result.get("models", []),
        "request_id": result.get("request_id"),
    }