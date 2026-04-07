# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Agriculture Environment.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with:\n    uv sync\n"
    ) from e

from models import AgricultureAction, AgricultureObservation

try:
    from .agriculture_environment import AgricultureEnvironment
except ModuleNotFoundError:
    from server.agriculture_environment import AgricultureEnvironment


app = create_app(
    AgricultureEnvironment,
    AgricultureAction,
    AgricultureObservation,
    env_name="agriculture",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)