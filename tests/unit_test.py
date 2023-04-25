from __future__ import annotations

import torch
import asyncio
import pytest
import typing as t

import detectron2.structures as Structures

if t.TYPE_CHECKING:
    import PIL.Image
    import PIL.PpmImagePlugin
    from pytest_mock import MockerFixture


@pytest.mark.asyncio
async def test_segmentation(mocker: MockerFixture, pdf_image: PIL.Image.Image):
    import service

    p_runner = mocker.patch("service.processor")
    p_runner.async_run = p_runner.object(service.processor, "async_run")
    instances = Structures.Instances(num_instances=2, image_size=(2200, 1700))
    instances.pred_classes = torch.tensor([1, 2])
    instances.scores = torch.tensor([0.9, 0.8])
    instances.pred_boxes = Structures.Boxes(
        torch.tensor([[2, 3, 4, 1], [882, 23, 23, 45]])
    )
    future = asyncio.Future()
    future.set_result({"instances": instances})
    p_runner.async_run.return_value = future

    res = await service.segmentation(pdf_image)
    assert res == ([1, 2], [0.9, 0.8], instances.pred_boxes)


@pytest.mark.asyncio
async def test_preprocess(mocker: MockerFixture, pdf_image: PIL.Image.Image):
    import service

    c_runner = mocker.patch("service.en_reader")
    c_runner.async_run = c_runner.object(service.en_reader, "async_run")
    future = asyncio.Future()
    future.set_result([0.82, "The quick brown fox jumps over the lazy dog."])
    c_runner.async_run.return_value = future

    p_runner = mocker.patch("service.processor")
    p_runner.async_run = p_runner.object(service.processor, "async_run")
    instances = Structures.Instances(num_instances=2, image_size=(2200, 1700))
    instances.pred_classes = torch.tensor([1, 2])
    instances.scores = torch.tensor([0.9, 0.8])
    instances.pred_boxes = Structures.Boxes(
        torch.tensor([[2, 3, 4, 1], [882, 23, 23, 45]])
    )
    future = asyncio.Future()
    future.set_result({"instances": instances})
    p_runner.async_run.return_value = future

    _ = []
    res = await service.preprocess(pdf_image, _)
    assert isinstance(res, list)


@pytest.mark.asyncio
async def test_image_to_text(
    mocker: MockerFixture, pdf_image: PIL.PpmImagePlugin.PpmImageFile
):
    import service

    c_runner = mocker.patch("service.en_reader")
    c_runner.async_run = c_runner.object(service.en_reader, "async_run")
    future = asyncio.Future()
    future.set_result([0.82, "The quick brown fox jumps over the lazy dog."])
    c_runner.async_run.return_value = future

    p_runner = mocker.patch("service.processor")
    p_runner.async_run = p_runner.object(service.processor, "async_run")
    instances = Structures.Instances(image_size=(2200, 1700))
    instances.pred_classes = torch.tensor([1, 2])
    instances.scores = torch.tensor([0.9, 0.8])
    instances.pred_boxes = Structures.Boxes(
        torch.tensor([[2, 3, 4, 1], [882, 23, 23, 45]])
    )
    future = asyncio.Future()
    future.set_result({"instances": instances})
    p_runner.async_run.return_value = future

    res = await service.image_to_text(pdf_image.fp)

    assert isinstance(res, dict) and isinstance(res["parsed"], str)
