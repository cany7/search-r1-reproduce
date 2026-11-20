"""Legacy stage-three CLI over the ordinary-to-dynamic fork protocol."""

import argparse
import asyncio
from pathlib import Path

import chz

from search_r1.training import runtime
from search_r1.training.checkpoints import read_config
from search_r1.training.config import from_snapshot


def build_config(parent, output, target, save_every, max_rounds, parent_step):
    # The first invocation inherits the parent's recipe. On restart the child
    # snapshot is sufficient even if the parent directory has been archived.
    settings = read_config(output if (output / "config.json").exists() else parent)
    return chz.replace(
        from_snapshot(settings),
        log_path=str(output),
        parent_run=str(parent),
        parent_step=parent_step,
        load_checkpoint_path=None,
        wandb_name=output.name,
        session_name=output.name,
        max_steps=target,
        save_every=save_every,
        eval_every=0,
        num_substeps=1,
        num_epochs=1,
        stream_minibatch_config=None,
        remove_constant_reward_groups=True,
        sampling_mode="dynamic",
        max_sampling_rounds=max_rounds,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", required=True, type=int)
    parser.add_argument("--parent-step", type=int, default=100, help="Legacy calls default to 100")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(
        runtime.main(
            build_config(
                args.parent,
                args.output,
                args.target,
                args.save_every,
                args.max_rounds,
                args.parent_step,
            )
        )
    )


if __name__ == "__main__":
    main()
