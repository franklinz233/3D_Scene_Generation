import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a 3D Gaussian scene from a single RGB image.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the OmegaConf YAML configuration.",
    )
    parser.add_argument(
        "--input",
        default="test_data/figure/castle.png",
        help="Input RGB image. Defaults to the bundled castle example.",
    )
    parser.add_argument(
        "--resize-long-edge",
        type=int,
        default=None,
        help="Override scene.input.resize_long_edge from the config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from scenedreamer.config import load_cfg
    from scenedreamer.engine import SceneDreamerPipeline

    cfg = load_cfg(args.config)

    if args.input:
        cfg.scene.input.rgb = args.input
    if args.resize_long_edge is not None:
        cfg.scene.input.resize_long_edge = args.resize_long_edge

    SceneDreamerPipeline(cfg)()
