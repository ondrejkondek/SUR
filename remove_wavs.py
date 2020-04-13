import os

dir_name = "./data_projekt/test/non_target_dev"
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".wav"):
        os.remove(os.path.join(dir_name, item))
