"""Dynamic sampling CLI; ordinary and dynamic runs use the same local loop."""

import asyncio
from typing import Literal

import chz

from search_r1.config import CLIConfig as CommonCLIConfig
from search_r1.train import cli_main


@chz.chz
class CLIConfig(CommonCLIConfig):
    sampling_mode: Literal["dynamic"] = "dynamic"


if __name__ == "__main__":
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
