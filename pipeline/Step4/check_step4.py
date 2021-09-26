from reprod_log import ReprodDiffHelper

if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("AlexNet_torch/bp_align_torch.npy")
    paddle_info = diff_helper.load_info("AlexNet_paddle/bp_align_paddle.npy")

    diff_helper.compare_info(torch_info, paddle_info)

    diff_helper.report(path="bp_align_diff.log")
