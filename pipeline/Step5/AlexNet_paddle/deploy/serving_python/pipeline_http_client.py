# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import requests
import json
import cv2
import base64
import os


def get_args(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description='Paddle Serving', add_help=add_help)

    parser.add_argument('--img-path', default="../../images/demo.jpg")
    args = parser.parse_args()
    return args


def cv2_to_base64(image):
    """cv2_to_base64

    Convert an numpy array to a base64 object.

    Args:
        image: Input array.

    Returns: Base64 output of the input.
    """
    return base64.b64encode(image).decode('utf8')


def main(args):
    url = "http://127.0.0.1:18080/alexnet/prediction"
    logid = 10000

    img_path = args.img_path
    with open(img_path, 'rb') as file:
        image_data1 = file.read()
    # data should be transformed to the base64 format
    image = cv2_to_base64(image_data1)
    data = {"key": ["image"], "value": [image], "logid": logid}
    # send requests
    r = requests.post(url=url, data=json.dumps(data))
    print(r.json())


if __name__ == "__main__":
    args = get_args()
    main(args)
