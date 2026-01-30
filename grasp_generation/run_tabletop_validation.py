import argparse
import os

import numpy as np

from src.validate_tabletop import validate_one_object_tabletop


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tabletop validation from an existing validated .npy file."
    )
    parser.add_argument(
        "--validated_path",
        required=True,
        help="Path to *_success_validated.npy",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run tabletop validation in debug (single sample + video).",
    )
    args = parser.parse_args()

    validated_path = args.validated_path
    if not os.path.exists(validated_path):
        raise FileNotFoundError(f"Validated file not found: {validated_path}")

    tabletop_success_indices = validate_one_object_tabletop(
        validated_path,
        debug=args.debug,
    )

    output_dir = os.path.dirname(validated_path)
    object_code = os.path.basename(validated_path).split("_success_validated.npy")[0]
    tabletop_validated_path = os.path.join(
        output_dir, f"{object_code}_tabletop_validated.npy"
    )
    np.save(
        tabletop_validated_path,
        np.load(validated_path, allow_pickle=True)[tabletop_success_indices],
        allow_pickle=True,
    )

    print(
        f"Saved tabletop validated grasps: {tabletop_validated_path} "
        f"({len(tabletop_success_indices)} / "
        f"{len(np.load(validated_path, allow_pickle=True))})"
    )


if __name__ == "__main__":
    main()
