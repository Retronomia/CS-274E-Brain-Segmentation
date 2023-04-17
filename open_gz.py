from utils import read_json

#Example of opening compressed json file:
json = read_json(
    "experiments/6xf6u-0-wmh_sp-AutoEnc-Custom_Loss/data_summary_test.gz")
for c, v in json.items():
    print(f"{c}:")
    print(v)
    print("=======================")
