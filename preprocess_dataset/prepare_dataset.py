import os
import shutil
import yaml
from box import Box

with open("config/config.yaml") as f:
    cfg = Box(yaml.safe_load(f))

base = "/data/gauravs/"
ssg = os.path.join(base, "ssg-qa")
cholec = os.path.join(base, "TF-Cholec80/data/cholec80")
new_data = os.path.join(base, "combine_data")
if not os.path.exists(new_data):
    os.mkdir(new_data)

for i in range(50):
    i = "{:02d}".format(i)

    folder = f"VID{i}"
    new_folder = os.path.join(new_data, folder)
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
        
    imgs = os.path.join(new_folder, "imgs")
    qas = os.path.join(new_folder, "qas")

    ssg_path = os.path.join(ssg, folder)

    if os.path.exists(ssg_path):
        ssg_list = os.listdir(ssg_path)
        cholec_path = os.path.join(cholec, "frames/video01")
        cholec_list = os.listdir(cholec_path)

        for s in ssg_list:
            snum = int(s.split(".")[0])
            for c in cholec_list:
                cnum = int(c.split("_")[1].split(".")[0])
                if cnum == snum:
                    img = os.path.join(cholec_path, c)
                    qa = os.path.join(ssg_path, s)
                    shutil.copyfile(img, f"{imgs}/{snum}.png")
                    shutil.copyfile(qa, f"{qas}/{s}")