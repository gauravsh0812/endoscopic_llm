import os

def accuracy():
    logfile = open("logs/predictions.txt").readlines()

    total = len(logfile)
    count = 0
    for l in logfile:
        _,_,tgt,pred = l.replace("\n","").split("\t")
        tgt = tgt.replace("<sos>","").replace("<eos>","").strip()
        pred = pred.replace("<sos>","").replace("<eos>","").strip()
        print(f"{tgt} \t {pred}")
        if tgt == pred:
            count += 1

    print("Accuracy: ", count/total)

accuracy()