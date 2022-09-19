# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""A test script for mid frame interpolation from two input frames.

Usage example:
 python3 -m frame_interpolation.eval.interpolator_test \
   --frame1 <filepath of the first frame> \
   --frame2 <filepath of the second frame> \
   --model_path <The filepath of the TF2 saved model to use>

The output is saved to <the directory of the input frames>/output_frame.png. If
`--output_frame` filepath is provided, it will be used instead.
"""
import os
from typing import Sequence

from . import interpolator
from . import util
from absl import app
from absl import flags
import glob
import numpy as np
from tqdm import tqdm

_DIR = flags.DEFINE_string(
    name="dir", default="./data/output_frames", help="The directory path of frames"
)
_HALF_FRAME = flags.DEFINE_bool(
    name="half_frame",
    default=False,
    help="force a single midframe, exactly halfway between two frames",
)
_MODEL_PATH = flags.DEFINE_string(
    name="model_path",
    default="./pretrained_models/film_net/Style/saved_model",
    help="The path of the TF2 saved model to use.",
)
_ALIGN = flags.DEFINE_integer(
    name="align",
    default=64,
    help="If >1, pad the input size so it is evenly divisible by this value.",
)
_BLOCK_HEIGHT = flags.DEFINE_integer(
    name="block_height",
    default=None,
    help="An int >= 1, number of patches along height, "
    "patch_height = height//block_height, should be evenly divisible.",
)
_BLOCK_WIDTH = flags.DEFINE_integer(
    name="block_width",
    default=None,
    help="An int >= 1, number of patches along width, "
    "patch_width = width//block_width, should be evenly divisible.",
)


def get_frame_num(path: os.PathLike) -> int:
    name = os.path.basename(path).replace(".png", "")
    return int(name)


def _run_interpolator() -> None:

    model_wrapper = interpolator.Interpolator(_MODEL_PATH.value, _ALIGN.value)

    frames_path = _DIR.value

    file_paths = sorted(glob.glob(_DIR.value + "/**/*.png", recursive=True))

    print(file_paths)

    prev_frame_num = 1
    prev_file_path = file_paths[0]
    for file_path in tqdm(file_paths):

        frame_num = get_frame_num(file_path)

        num_intermediary_frames = frame_num - prev_frame_num - 1

        if num_intermediary_frames > 0:
            # First batched image.
            image_1 = util.read_image(prev_file_path)
            image_batch_1 = np.expand_dims(image_1, axis=0)

            # Second batched image.
            image_2 = util.read_image(file_path)
            image_batch_2 = np.expand_dims(image_2, axis=0)

            for i in range(num_intermediary_frames):
                intermediary_frame_num = prev_frame_num + i + 1
                print(i, prev_file_path, file_path)

                # Batched time.
                if _HALF_FRAME.value:
                    fill_value = 0.5
                else:
                    fill_value = (i + 1) / (num_intermediary_frames + 1)
                batch_dt = np.full(
                    shape=(1,),
                    fill_value=fill_value,
                    dtype=np.float32,
                )

                # Invoke the model once.
                mid_frame = model_wrapper.interpolate(
                    image_batch_1, image_batch_2, batch_dt
                )[0]

                # Write interpolated mid-frame.
                mid_frame_filepath = os.path.join(
                    frames_path, f"{intermediary_frame_num:0>10d}.png"
                )
                util.write_image(mid_frame_filepath, mid_frame)

        prev_frame_num = frame_num
        prev_file_path = file_path


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    _run_interpolator()


if __name__ == "__main__":
    app.run(main)
