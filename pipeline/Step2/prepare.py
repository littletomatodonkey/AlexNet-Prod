import os
import sys


def copy_data(input_dir, output_dir, total_num=16):
    img_num = 0
    dirs = os.listdir(input_dir)
    for idx, dir_name in enumerate(dirs):
        temp_input_dir = os.path.join(input_dir, dir_name)
        temp_output_dir = os.path.join(output_dir, dir_name)
        os.makedirs(temp_output_dir, exist_ok=True)
        if img_num >= total_num:
            continue
        img_name = os.listdir(temp_input_dir)[0]
        input_image_name = os.path.join(temp_input_dir, img_name)
        output_image_name = os.path.join(temp_output_dir, img_name)
        cmd = "cp {} {}".format(input_image_name, output_image_name)
        os.system(cmd)
        img_num += 1
    return


input_dir = "/paddle/data/ILSVRC2012/train/"
output_dir = "lite_data/train"
copy_data(input_dir, output_dir)

input_dir = "/paddle/data/ILSVRC2012/val/"
output_dir = "lite_data/val"
copy_data(input_dir, output_dir)
