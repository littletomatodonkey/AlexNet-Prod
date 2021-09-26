from reprod_log import ReprodDiffHelper

if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()
    benchmark_info = diff_helper.load_info("./train_align_benchmark.npy")
    paddle_info = diff_helper.load_info(
        "AlexNet_paddle/train_align_paddle.npy")

    diff_helper.compare_info(benchmark_info, paddle_info)

    diff_helper.report(path="train_align_diff.log")
