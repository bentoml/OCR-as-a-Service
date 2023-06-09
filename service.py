from __future__ import annotations

import io
import os
import typing as t
import asyncio

import numpy as np
import torch

import bentoml
from warmup import convert_pdf_to_images
from warmup import download_models

if t.TYPE_CHECKING:
    import PIL.Image
    from detectron2.structures import Boxes
    from detectron2.structures import Instances

# Download Model
reader_model, predictor_model = download_models()
en_reader = reader_model.to_runner()
processor = predictor_model.to_runner()

THRESHOLD = float(os.getenv("OCR_THRESHOLD", 0.8))

svc = bentoml.Service(name="document-processing", runners=[en_reader, processor])


async def segmentation(im: PIL.Image.Image) -> tuple[list[int], list[float], Boxes]:
    output: Instances = (await processor.async_run(np.asarray(im)))["instances"]
    return (
        output.get("pred_classes").tolist(),
        output.get("scores").tolist(),
        output.get("pred_boxes"),
    )


async def preprocess(im: PIL.Image.Image, res: list[str], threshold: float = 0.8):
    async def _proc_cls_scores(
        cls: int, score: float, box: torch.Tensor, im: PIL.Image.Image
    ):
        # 4 table
        if cls != 4 and score >= threshold:
            join_char = "" if cls == 0 else " "
            text = join_char.join(
                [
                    t[1]
                    for t in await en_reader.readtext.async_run(
                        np.asarray(im.crop(box.numpy()))
                    )
                ]
            )
            # ignore annotations for table footer
            if not text.startswith("Figure") or not text.startswith("Table"):
                print("Extract text:", text)
                res.append(text)

    classes, scores, boxes = await segmentation(im)

    return await asyncio.gather(
        *[
            _proc_cls_scores(cls, score, box, im)
            for cls, score, box in zip(classes, scores, boxes)
        ]
    )


@svc.api(input=bentoml.io.File(mime_type="application/pdf"), output=bentoml.io.JSON())
async def image_to_text(file: io.BytesIO) -> dict[t.Literal["parsed"], str]:
    res = []
    with file:
        await asyncio.gather(
            *[
                preprocess(im, res, THRESHOLD)
                for im in convert_pdf_to_images(file.read())
            ]
        )
    return {"parsed": "\n".join(res)}
