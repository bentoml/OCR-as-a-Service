from __future__ import annotations

import platform
import pytest
import typing as t
import bentoml
from bentoml.testing.utils import async_request
from bentoml._internal.utils import LazyLoader

if t.TYPE_CHECKING:
    from bentoml.client import HTTPClient, GrpcClient
    from grpc_health.v1 import health_pb2 as pb_health

    P = t.TypeVar("P")
    Generator = t.Generator[P, None, None]
else:
    pb_health = LazyLoader("pb_health", globals(), "grpc_health.v1.health_pb2")


@pytest.mark.asyncio
async def test_api_server_meta_http(host: str) -> None:
    client = bentoml.client.Client.from_url(host)
    await async_request("GET", client.server_url, assert_status=200)
    await async_request("GET", f"{client.server_url}/healthz", assert_status=200)
    await async_request("GET", f"{client.server_url}/livez", assert_status=200)
    await async_request("GET", f"{client.server_url}/docs.json", assert_status=200)
    status, _, body = await async_request("GET", f"{client.server_url}/metrics")
    assert status == 200 and body


@pytest.mark.asyncio
async def test_inference_http(host: str, text: str, categories: list[str]):
    client = t.cast("HTTPClient", bentoml.client.Client.from_url(host))
    resp = await client.async_health()
    assert resp.status == 200

    res = await client.async_summarize(text)
    assert isinstance(res, str)

    res = await client.async_categorize({"text": text, "categories": categories})
    assert isinstance(res, dict) and res["entertainment"] > 0.5


@pytest.mark.asyncio
@pytest.mark.skipif(
    platform.system() == "Windows", reason="grpc not yet supported on Windows"
)
async def test_inference_grpc(host: str, text: str, categories: list[str]):
    client = t.cast("GrpcClient", bentoml.client.Client.from_url(host))
    res = await client.health("bentoml.grpc.v1.BentoService")
    assert res.status == pb_health.HealthCheckResponse.SERVING

    res = await client.async_call("summarize", text)
    assert res.text and isinstance(res.text.value, str)

    res = await client.async_call(
        "categorize", {"text": text, "categories": categories}
    )
    assert (
        res.json
        and res.json.struct_value.fields["entertainment"].value.number_value > 0.5
    )
