import os

from data_utils_sentihood import *

data_dir='../data/sentihood/'
aspect2idx = {
    'general': 0,
    'price': 1,
    'transit-location': 2,
    'safety': 3,
}

(train, train_aspect_idx), (val, val_aspect_idx), (test, test_aspect_idx) = load_task(data_dir, aspect2idx)

print("len(train) = ", len(train))
print("len(val) = ", len(val))
print("len(test) = ", len(test))

train.sort(key=lambda x:x[2]+str(x[0])+x[3][0])
val.sort(key=lambda x:x[2]+str(x[0])+x[3][0])
test.sort(key=lambda x:x[2]+str(x[0])+x[3][0])

location_name = ['loc1', 'loc2']
aspect_name = ['general', 'price', 'safety', 'transit']
dir_path = [data_dir + 'bert-single/' + i + '_' + j + '/' for i in location_name for j in aspect_name]
for path in dir_path:
    if not os.path.exists(path):
        os.makedirs(path)

count=0
with open(dir_path[0]+"train.tsv","w",encoding="utf-8") as f1_general, \
    open(dir_path[1]+"train.tsv", "w", encoding="utf-8") as f1_price, \
    open(dir_path[2]+"train.tsv", "w", encoding="utf-8") as f1_safety, \
    open(dir_path[3]+"train.tsv", "w", encoding="utf-8") as f1_transit, \
    open(dir_path[4]+"train.tsv", "w", encoding="utf-8") as f2_general, \
    open(dir_path[5]+"train.tsv", "w", encoding="utf-8") as f2_price, \
    open(dir_path[6]+"train.tsv", "w", encoding="utf-8") as f2_safety, \
    open(dir_path[7]+"train.tsv", "w",encoding="utf-8") as f2_transit, \
    open(data_dir + "bert-pair/train_NLI_M.tsv", "r", encoding="utf-8") as f:
    s = f.readline().strip()
    s = f.readline().strip()
    while s:
        count+=1
        tmp=s.split("\t")
        line=tmp[0]+"\t"+tmp[1]+"\t"+tmp[3]+"\n"
        if count<=11908:               #loc1
            if count%4==1:
                f1_general.write(line)
            if count%4==2:
                f1_price.write(line)
            if count%4==3:
                f1_safety.write(line)
            if count%4==0:
                f1_transit.write(line)
        else:                          #loc2
            if count%4==1:
                f2_general.write(line)
            if count%4==2:
                f2_price.write(line)
            if count%4==3:
                f2_safety.write(line)
            if count%4==0:
                f2_transit.write(line)
        s = f.readline().strip()

count=0
with open(dir_path[0]+"dev.tsv","w",encoding="utf-8") as f1_general, \
    open(dir_path[1]+"dev.tsv", "w", encoding="utf-8") as f1_price, \
    open(dir_path[2]+"dev.tsv", "w", encoding="utf-8") as f1_safety, \
    open(dir_path[3]+"dev.tsv", "w", encoding="utf-8") as f1_transit, \
    open(dir_path[4]+"dev.tsv", "w", encoding="utf-8") as f2_general, \
    open(dir_path[5]+"dev.tsv", "w", encoding="utf-8") as f2_price, \
    open(dir_path[6]+"dev.tsv", "w", encoding="utf-8") as f2_safety, \
    open(dir_path[7]+"dev.tsv", "w",encoding="utf-8") as f2_transit, \
    open(data_dir + "bert-pair/dev_NLI_M.tsv", "r", encoding="utf-8") as f:
    s = f.readline().strip()
    s = f.readline().strip()
    while s:
        count+=1
        tmp=s.split("\t")
        line=tmp[0]+"\t"+tmp[1]+"\t"+tmp[3]+"\n"
        if count<=2988:               #loc1
            if count%4==1:
                f1_general.write(line)
            if count%4==2:
                f1_price.write(line)
            if count%4==3:
                f1_safety.write(line)
            if count%4==0:
                f1_transit.write(line)
        else:                          #loc2
            if count%4==1:
                f2_general.write(line)
            if count%4==2:
                f2_price.write(line)
            if count%4==3:
                f2_safety.write(line)
            if count%4==0:
                f2_transit.write(line)
        s = f.readline().strip()

count=0
with open(dir_path[0]+"test.tsv","w",encoding="utf-8") as f1_general, \
    open(dir_path[1]+"test.tsv", "w", encoding="utf-8") as f1_price, \
    open(dir_path[2]+"test.tsv", "w", encoding="utf-8") as f1_safety, \
    open(dir_path[3]+"test.tsv", "w", encoding="utf-8") as f1_transit, \
    open(dir_path[4]+"test.tsv", "w", encoding="utf-8") as f2_general, \
    open(dir_path[5]+"test.tsv", "w", encoding="utf-8") as f2_price, \
    open(dir_path[6]+"test.tsv", "w", encoding="utf-8") as f2_safety, \
    open(dir_path[7]+"test.tsv", "w",encoding="utf-8") as f2_transit, \
    open(data_dir + "bert-pair/test_NLI_M.tsv", "r", encoding="utf-8") as f:
    s = f.readline().strip()
    s = f.readline().strip()
    while s:
        count+=1
        tmp=s.split("\t")
        line=tmp[0]+"\t"+tmp[1]+"\t"+tmp[3]+"\n"
        if count<=5964:               #loc1
            if count%4==1:
                f1_general.write(line)
            if count%4==2:
                f1_price.write(line)
            if count%4==3:
                f1_safety.write(line)
            if count%4==0:
                f1_transit.write(line)
        else:                          #loc2
            if count%4==1:
                f2_general.write(line)
            if count%4==2:
                f2_price.write(line)
            if count%4==3:
                f2_safety.write(line)
            if count%4==0:
                f2_transit.write(line)
        s = f.readline().strip()

print("Finished!")