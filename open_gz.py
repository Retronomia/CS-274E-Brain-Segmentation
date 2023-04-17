from utils import read_json


# curr best custom linear
# experiments/0z7ss-0-wmh_sp-AutoEnc-Custom_Loss/data_summary_test.gz
# curr best unsup
# experiments/0obl5-0-wmh_usp-AutoEnc-L1_Loss/data_summary_test.gz
json = read_json(
    "experiments/6xf6u-0-wmh_sp-AutoEnc-Custom_Loss/data_summary_test.gz")
for c, v in json.items():
    print(f"{c}:")
    print(v)
    print("=======================")
