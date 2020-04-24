import os
import sys

if (len(sys.argv) != 2):
    print("remove_wavs.py [DIR]")
    exit(1)


dir_name = sys.argv[1]
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".wav"):
        os.remove(os.path.join(dir_name, item))
