from reprod_log import ReprodDiffHelper

if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()
    inference_out = diff_helper.load_info(
        "AlexNet_paddle/output_inference_engine.npy")
    training_out = diff_helper.load_info(
        "AlexNet_paddle/output_training_engine.npy")

    diff_helper.compare_info(inference_out, training_out)

    diff_helper.report(path="train_infer_diff.log", diff_threshold=1e-5)
