from pathlib import Path

data_path = Path("data/news.2017.zh.shuffled.deduped")

train_data_path = Path("data/train.txt")
test_data_path = Path("data/test.txt")

with open(data_path) as f:
    lines = f.readlines()

# split data
test_lines = lines[-2000:]  # 最后2000行
train_lines = lines[:-2000]  # 其他行

with open(train_data_path, "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open(test_data_path, "w", encoding="utf-8") as f:
    f.writelines(test_lines)
