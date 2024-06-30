import multiprocessing as mp
import os, shutil, json
import pandas as pd
import tqdm


"""
Questions:

how many tools are operating?
what is the phase of image?
'is ' + tools[i] + ' used in ' + phases + '?' >> tools[i] + ' is used in ' + phases OR yes
'is ' + label + ' used in ' + phases + '?' >> label + ' is not used in ' + phases  OR no

We start with the simpler version with yes and no. Later we can introduce other option as well.
"""    

def main():
    
    # defining paths
    seq = range(1,80)
    folder = '/data/gauravs/TF-Cholec80/data/cholec80'
    tool_path = os.path.join(folder, "tool_annotations")
    phase_path = os.path.join(folder, "phase_annotations")

    os.makedirs(os.path.join(folder, "dfs"), exist_ok=True)

    dst = "/data/gauravs/surgicalGPT/our_dataset"   
    vqa_qtns = open("/data/gauravs/surgicalGPT/our_dataset/vqa/questions.lst","w")
    vqa_ans = open("/data/gauravs/surgicalGPT/our_dataset/vqa/answers.lst","w")  
    gvqa_qtns = open("/data/gauravs/surgicalGPT/our_dataset/gvqa/questions.lst","w")
    gvqa_ans = open("/data/gauravs/surgicalGPT/our_dataset/gvqa/answers.lst","w")    

    count = 0

    # iterating over files
    for num in tqdm.tqdm(seq):
        num = f"{num:02}"
        sub_tool_path = os.path.join(tool_path, f"video{num}-tool.txt")
        sub_phase_path = os.path.join(phase_path, f"video{num}-phase.txt")

        df_frame, df_tool, df_phase = list(),list(),list()

        tf = open(sub_tool_path).readlines()[1:]
        phs = open(sub_phase_path).readlines()[1:]    

        for _tf in tf:
            (frame, grasper, bipolar, hook, scissors, clippers, irrigator, specimenbag) = _tf.split("\t")

            # frame
            _n = int(frame)/25
            df_frame.append(_n)
            
            # tools
            _tools = []
            if int(grasper) == 1: _tools.append("grasper")
            if int(bipolar) == 1: _tools.append("bipolar")
            if int(hook) == 1: _tools.append("hook")
            if int(scissors) == 1: _tools.append("scissors")
            if int(clippers) == 1: _tools.append("clippers")
            if int(irrigator) == 1: _tools.append("irrigator")
            if int(specimenbag.replace("\n","")) == 1: _tools.append("specimen bag")
            df_tool.append(_tools)

            # phase 
            _ph = phs[int(frame)].split("\t")[-1].replace("\n","").strip()
            df_phase.append(space_conv_dict[_ph])
        
        df = pd.DataFrame(
            {
                "frames":df_frame,
                "tools":df_tool,
                "phases":df_phase
            }
        )
        df.to_csv(os.path.join(dst, f"dfs/df_{num}.csv"), index=False)


        # combining everything 
        for i in range(len(df)):

            # copying image
            row = df.iloc[i,:]
            row_tool = row['tools']
            row_phase = row['phases']
            row_img = f"{(int(row['frames'])+1):06}"
            
            imgsrc = os.path.join(folder, f"frames/video{num}/video{num}_{row_img}.png")
            imgdst = os.path.join(dst, f"images/{count}.png")
            shutil.copyfile(imgsrc, imgdst)
            count +=1
            
            # writing questions answers
            _vqaqtn = [
                    f"QTN{count} \t how many tools are operating?",
                    f"QTN{count} \t what is the phase of image?",
                    ]
            
            _vqaans = [
                    f"ANS{count} \t {str(len(row_tool))}", 
                    f"ANS{count} \t {row_phase}"
                    ]
            
            _gvqaqtn = [
                f"QTN{count} \t what tools are being used in this image of a surgery?",
                f"QTN{count} \t what can you tell us about the phase of surgery?",
                ]

            

            for t in row_tool:
                _vqaqtn.append(f"QTN{count} \t is {t} used in {row_phase}?")
                _vqaans.append(f"ANS{count} \t yes")
            
            for t in all_tool_list:
                if t not in row_tool:
                    _vqaqtn.append(f"QTN{count} \t is {t} used in {row_phase}?")
                    _vqaans.append(f"ANS{count} \t no")

            for q,a in zip(_vqaqtn, _vqaans):
                vqa_qtns.write(q + "\n")
                vqa_ans.write(a + "\n")


if __name__ == "__main__":

    files = os.listdir("/data/shared/CholecT50/CholecT50/labels")
    for _file in range(len(files)):
        _file_name = f"VID{int(_file).02d}.json"
        print(_file_name)
        data = json.load(open(f"/data/shared/CholecT50/CholecT50/labels/{_file_name}"))

        annotations = data["annotations"]
        categories = data["categories"]
        print(categories)
