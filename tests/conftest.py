from __future__ import annotations

import subprocess
import psutil
import sys
import os
import pytest
import contextlib
from pathlib import Path
import typing as t
import bentoml


if t.TYPE_CHECKING:
    from PIL import Image
    from _pytest.fixtures import FixtureRequest
    from _pytest.main import Session
    from _pytest.nodes import Item
    from _pytest.config import Config

    P = t.TypeVar("P")
    Generator = t.Generator[P, None, None]

PROJECT_PATH = Path(__file__).parent.parent
BENTO_NAME = "document-processing"


def convert_pdf_to_images(
    pdf_path: str | bytes, **convert_attrs: t.Any
) -> list[Image.Image]:
    try:
        import pdf2image
    except ImportError:
        raise RuntimeError(
            "Make sure to install all required dependencies with 'pip install -r requirements.txt'."
        )
    if not isinstance(pdf_path, (str, bytes)):
        raise TypeError(
            "pdf_path should be either a path to a PDF file or a bytes object containing a PDF file."
        )
    convert_attrs.setdefault("thread_count", 6)

    if isinstance(pdf_path, bytes):
        return pdf2image.convert_from_bytes(pdf_path, **convert_attrs)
    return pdf2image.convert_from_path(pdf_path, **convert_attrs)


def pytest_collection_modifyitems(
    session: Session, config: Config, items: list[Item]
) -> None:
    subprocess.check_call(
        [sys.executable, f"{os.path.join(PROJECT_PATH, 'warmup.py')}"]
    )


# TODO: Add containerize tests
@pytest.fixture(name="bento", scope="function")
def fixture_build_bento() -> Generator[bentoml.Bento]:
    try:
        bento = bentoml.get(BENTO_NAME)
    except bentoml.exceptions.NotFound:
        print(f"Building bento from path: {PROJECT_PATH}")
        subprocess.check_output(["bentoml", "build", "."])
        bento = bentoml.get(BENTO_NAME)
    yield bento
    bentoml.delete(BENTO_NAME)


@pytest.fixture(name="project_path", params=[PROJECT_PATH], scope="session")
def fixture_project_path(request: FixtureRequest):
    return request.param


@pytest.fixture(name="enable_grpc", params=[True, False], scope="session")
def fixture_enable_grpc(request: FixtureRequest):
    return request.param


@pytest.fixture(autouse=True, scope="session")
def bento_directory(request: FixtureRequest):
    os.chdir(PROJECT_PATH.__fspath__())
    sys.path.insert(0, PROJECT_PATH.__fspath__())
    yield
    os.chdir(request.config.invocation_dir)
    sys.path.pop(0)


@pytest.fixture(
    name="pdf_image",
    scope="session",
    params=convert_pdf_to_images(
        PROJECT_PATH.joinpath("samples", "2204.08387.pdf").__fspath__()
    ),
)
def fixture_pdf_image(request: FixtureRequest):
    return request.param


@pytest.fixture(scope="module")
def host(
    bentoml_home: str,
    deployment_mode: t.Literal["container", "distributed", "standalone"],
    clean_context: contextlib.ExitStack,
    project_path: str,
    enable_grpc: bool,
) -> t.Generator[str, None, None]:
    from bentoml.testing.server import host_bento

    if psutil.WINDOWS and enable_grpc:
        pytest.skip("gRPC is not yet supported on Windows.")

    with host_bento(
        "service:svc",
        deployment_mode=deployment_mode,
        project_path=project_path,
        bentoml_home=bentoml_home,
        clean_context=clean_context,
        use_grpc=enable_grpc,
        config_file=PROJECT_PATH.joinpath("config", "default.yaml").__fspath__(),
    ) as _host:
        yield _host
