import numpy as np
from reprod_log import ReprodLogger

if __name__ == "__main__":
    reprod_log_1 = ReprodLogger()
    reprod_log_2 = ReprodLogger()

    data_1 = np.random.rand(1, 3, 224, 224).astype(np.float32)
    data_2 = np.random.rand(1, 3, 224, 224).astype(np.float32)

    reprod_log_1.add("demo_test_1", data_1)
    reprod_log_1.add("demo_test_2", data_1)
    reprod_log_1.save("result_1.npy")

    reprod_log_2.add("demo_test_1", data_1)
    reprod_log_2.add("demo_test_2", data_2)
    reprod_log_2.save("result_2.npy")
