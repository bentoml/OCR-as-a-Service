from __future__ import annotations

import os
import uuid
import typing as t
import asyncio
import platform
import warnings
import subprocess

import numpy as np
import torch
import easyocr
from PIL import Image
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

import dit
import bentoml

if t.TYPE_CHECKING:
    from detectron2.config import CfgNode
    from detectron2.engine import DefaultPredictor
    from detectron2.structures import Boxes
    from detectron2.structures import Instances

warnings.filterwarnings("ignore", category=UserWarning)


def convert_pdf_to_images(
    pdf_path: str | bytes, **convert_attrs: t.Any
) -> list[Image.Image]:
    try:
        subprocess.check_output(["pdfinfo", "-v"], stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        if platform.system() == "Darwin":
            raise RuntimeError(
                "Make sure to install 'poppler' on macOS with brew: 'brew install poppler'"
            )
        elif platform.system() == "Windows":
            raise RuntimeError(
                "Refer to https://github.com/Belval/pdf2image for Windows instruction."
            )
        else:
            raise RuntimeError(
                "'pdftocairo' and 'pdftoppm' should already be included in your Linux distrobution (Seems like they are not installed). Refer to your package manager and install 'poppler-utils'"
            )
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

    fn = (
        pdf2image.convert_from_bytes
        if isinstance(pdf_path, bytes)
        else pdf2image.convert_from_path
    )
    return fn(pdf_path, **convert_attrs)


def segmentation(
    im: Image.Image, predictor: DefaultPredictor, cfg: CfgNode, visualize: bool = False
) -> tuple[list[int], list[float], Boxes]:
    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    tensor = np.array(im)
    output: Instances = predictor(tensor)["instances"]
    if visualize:
        v = Visualizer(
            tensor[:, :, ::-1], md, scale=1.0, instance_mode=ColorMode.SEGMENTATION
        )
        res = v.draw_instance_predictions(output.to("cpu"))
        Image.fromarray(res.get_image()[:, :, ::-1]).save(
            f"{uuid.uuid4()}-segmented.png"
        )
    return (
        output.get("pred_classes").tolist(),
        output.get("scores").tolist(),
        output.get("pred_boxes"),
    )


async def process_im(
    im: Image.Image,
    predictor: DefaultPredictor,
    cfg: CfgNode,
    reader: easyocr.Reader,
    res: list[str],
    threshold: float = 0.8,
):
    async def _proc_cls_scores(
        cls: int, score: float, box: torch.Tensor, im: Image.Image
    ):
        if cls != 4 and score >= threshold:
            cropped = im.crop(box.numpy())
            join_char = "" if cls == 0 else " "
            text = join_char.join([t[1] for t in reader.readtext(np.asarray(cropped))])
            # ignore annotations for table footer
            if not text.startswith("Figure") or not text.startswith("Table"):
                print("Extract text:", text)
                res.append(text)

    classes, scores, boxes = segmentation(im, predictor, cfg)

    return await asyncio.gather(
        *[
            _proc_cls_scores(cls, score, box, im)
            for cls, score, box in zip(classes, scores, boxes)
        ]
    )


@torch.inference_mode()
async def main(threshold: float = 0.8, analyze: bool = False):
    # TODO: support EOL token.
    reader = easyocr.Reader(["en"])
    cfg = dit.get_cfg()
    predictor = dit.get_predictor(cfg)

    if analyze:
        intro = (
            "\nUsing EasyOCR model with LayouLMv3 Detectron2 model for PDF extraction."
        )
        print(intro)
        print("=" * len(intro))
        res = []
        await asyncio.gather(
            *[
                process_im(im, predictor, cfg, reader, res, threshold)
                for im in convert_pdf_to_images(
                    os.path.join("samples", "2204.08387.pdf"), dpi=300
                )
            ]
        )
        print("results:", res)
        print("Finished processing all pages.")
        print("=" * 30)

    tag = "en-reader"
    try:
        reader_model = bentoml.easyocr.get(tag)
        print(f"'{tag}' is previously saved: {reader_model}")
    except bentoml.exceptions.NotFound:
        reader_model = bentoml.easyocr.save_model(tag, reader)
        print(f"'{tag}' is saved: {reader_model}")

    tag = "dit-predictor"
    try:
        predictor_model = bentoml.detectron.get(tag)
        print(f"'{tag}' is previously saved: {predictor_model}")
    except bentoml.exceptions.NotFound:
        predictor_model = bentoml.detectron.save_model(tag, predictor)
        print(f"'{tag}' is saved: {predictor_model}")

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true", default=False)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    raise SystemExit(asyncio.run(main(**vars(args))))
