import os

dir_name = "./data/train_target"
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".wav"):
        os.remove(os.path.join(dir_name, item))
