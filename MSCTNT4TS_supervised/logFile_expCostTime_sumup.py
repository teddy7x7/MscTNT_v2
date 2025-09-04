import re

def read_log_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def find_and_sum_cost_times(log_content):
    # pattern = r"Epoch: \d+ cost time: (\d+)"
    pattern = r"Epoch: \d+ cost time: (\d+\.\d+)"
    matches = re.findall(pattern, log_content)
    print("epoches : ", len(matches))
    print("matches : \n", matches)
    total_cost_time = sum(float(time) for time in matches)
    return total_cost_time

def main():
    # log_file_path = 'logs/LongForecasting/MscTNT4TS_Electricity_720_720.log'
    log_file_path = 'logs/LongForecasting/PatchTST_Electricity_PatchTST42_40epoch2_336_720.log'
    log_content = read_log_file(log_file_path)

    # print("log_content :\n", log_content)

    total_cost_time = find_and_sum_cost_times(log_content)
    print(f"log_file_path: {log_file_path}")
    print(f"Total cost time: {total_cost_time}")

if __name__ == "__main__":
    main()
