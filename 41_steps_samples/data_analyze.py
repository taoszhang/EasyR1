import os
import json

def load_txt_file(file_path):
    results = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(line)
    return results


ids = load_txt_file("ids.txt")
advantages = load_txt_file("advantages_sum.txt")
rewards = load_txt_file("reward_tensor.txt")

# 初始化一个每个值都是列表的字典
id2adv = {}
id2red = {}
for id, adv, reward in zip(ids, advantages, rewards):
    # 具有相同id的值，放到一个列表里
    id2adv[id] = id2adv.get(id, []) + [adv]
    id2red[id] = id2red.get(id, []) + [reward]

# 统计所有id的数量，以及平均值，和全为零的数量
id_count = len(id2adv)
id_sum = 0
id_zero_count = 0
for id, adv in id2adv.items():
    # 计算每个id的平均值
    avg = sum([float(a) for a in adv]) / len(adv)
    id_sum += avg
    # 如果idvalues全为零，计数
    if all([float(a) == 0 for a in adv]):
        id_zero_count += 1

# 保存字典
with open("id2adv.json", "w") as f:
    json.dump(id2adv, f, indent=4)

with open("id2red.json", "w") as f:
    json.dump(id2red, f, indent=4)

print(f"Total id count: {id_count}")
print(f"Average id value: {id_sum / id_count}")
print(f"Total id zero count: {id_zero_count}")
