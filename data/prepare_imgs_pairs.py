import argparse
import numpy as np
import os
import re
from pathlib import Path 
import random

seed_value = 12345
random.seed(seed_value)

parser = argparse.ArgumentParser(description='MegaDepth preprocessing script')

parser.add_argument(
    '--base_path', type=str, required=True,
    help='path to Dataset'
)
parser.add_argument(
    '--scene_id', type=str, required=True,
    help='scene ID'
)

parser.add_argument(
    '--output_path', type=str, required=True,
    help='path to the output directory'
)

args = parser.parse_args()

base_path = args.base_path
# Remove the trailing / if need be.
if base_path[-1] in ['/', '\\']:
    base_path = base_path[: - 1]

output_path = args.output_path
# Remove the trailing / if need be.
if output_path[-1] in ['/', '\\']:
    output_path = output_path[: - 1]

scene_id = args.scene_id

pattern = re.compile(r"^\d")

data_path = Path(f"{base_path}/{scene_id}")
total_num = max([int(re.findall(r"\d+", file.name)[0]) for file in list(data_path.glob('*')) if pattern.match(file.name)])

scene = data_path.name
name = [] 
score = [] 

print(f"{scene_id} has total num of ids: {total_num}")

#train.npz
for i in range(total_num+1):
    next = random.randint(20,50)
    if (i+next)>total_num:
        continue
    name.append([scene, str(i).zfill(4), str(i+next).zfill(4)])
    score.append(1.)
random.shuffle(name)
np.savez(f"{output_path}/train/{args.scene_id}.npz", name = name, score = score)

#val.npz
# resample the sequence by 10 frames and choose num_choices pair
name = [] 
score = [] 
values = list(range(0, total_num + 1, 10))
num_choices=10
if len(values)<num_choices:
    values = list(range(0, total_num + 1))
random_choices = random.sample(values, num_choices)

for i in random_choices:
    next = random.randint(20,50)
    if (i+next)>total_num:
        name.append([scene, str(total_num-next).zfill(4), str(total_num).zfill(4)])
        continue
    name.append([scene, str(i).zfill(4), str(i+next).zfill(4)])
    score.append(1.)
random.shuffle(name)
np.savez(f"{output_path}/val/{args.scene_id}.npz", name = name, score = score)