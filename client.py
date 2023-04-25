from __future__ import annotations

import io

import bentoml

SAMPLES = "./samples/2204.08387.pdf"


def call(host: str = "127.0.0.1") -> int:
    client = bentoml.client.Client.from_url(f"{host}:3000")
    with open(SAMPLES, "rb") as f:
        res = client.image_to_text(io.BytesIO(f.read()))
        print("Summarized text from the article:", res)
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")

    args = parser.parse_args()

    raise SystemExit(call(host=args.host))
