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

dir_path = data_dir+'bert-pair/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

sentiments=["None","Positive","Negative"]
with open(dir_path+"train_NLI_B.tsv","w",encoding="utf-8") as f:
    f.write("id\tsentence1\tsentence2\tlabel\n")
    for v in train:
        for sentiment in sentiments:
            f.write(str(v[0])+"\t")
            word=v[1][0].lower()
            if word=='location1':f.write('location - 1')
            elif word=='location2':f.write('location - 2')
            elif word[0]=='\'':f.write("\' "+word[1:])
            else:f.write(word)
            for i in range(1,len(v[1])):
                word=v[1][i].lower()
                f.write(" ")
                if word == 'location1':
                    f.write('location - 1')
                elif word == 'location2':
                    f.write('location - 2')
                elif word[0] == '\'':
                    f.write("\' " + word[1:])
                else:
                    f.write(word)
            f.write("\t")
            f.write(sentiment+" - ")
            if v[2]=='LOCATION1':f.write('location - 1 - ')
            if v[2]=='LOCATION2':f.write('location - 2 - ')
            if len(v[3])==1:
                f.write(v[3][0]+"\t")
            else:
                f.write("transit location\t")
            if v[4]==sentiment:
                f.write("1\n")
            else:
                f.write("0\n")
                
with open(dir_path+"train_QA_B.tsv","w",encoding="utf-8") as f:
    f.write("id\tsentence1\tsentence2\tlabel\n")
    for v in train:
        for sentiment in sentiments:
            f.write(str(v[0])+"\t")
            word=v[1][0].lower()
            if word=='location1':f.write('location - 1')
            elif word=='location2':f.write('location - 2')
            elif word[0]=='\'':f.write("\' "+word[1:])
            else:f.write(word)
            for i in range(1,len(v[1])):
                word=v[1][i].lower()
                f.write(" ")
                if word == 'location1':
                    f.write('location - 1')
                elif word == 'location2':
                    f.write('location - 2')
                elif word[0] == '\'':
                    f.write("\' " + word[1:])
                else:
                    f.write(word)
            f.write("\t")
            f.write("the polarity of the aspect ")
            if len(v[3])==1:
                f.write(v[3][0])
            else:
                f.write("transit location")
            if v[2]=='LOCATION1':f.write(' of location - 1 - is ')
            if v[2]=='LOCATION2':f.write(' of location - 2 - is ')
            f.write(sentiment+" .\t")
            if v[4]==sentiment:
                f.write("1\n")
            else:
                f.write("0\n")

with open(dir_path+"dev_NLI_B.tsv","w",encoding="utf-8") as f:
    f.write("id\tsentence1\tsentence2\tlabel\n")
    for v in val:
        for sentiment in sentiments:
            f.write(str(v[0])+"\t")
            word=v[1][0].lower()
            if word=='location1':f.write('location - 1')
            elif word=='location2':f.write('location - 2')
            elif word[0]=='\'':f.write("\' "+word[1:])
            else:f.write(word)
            for i in range(1,len(v[1])):
                word=v[1][i].lower()
                f.write(" ")
                if word == 'location1':
                    f.write('location - 1')
                elif word == 'location2':
                    f.write('location - 2')
                elif word[0] == '\'':
                    f.write("\' " + word[1:])
                else:
                    f.write(word)
            f.write("\t")
            f.write(sentiment+" - ")
            if v[2]=='LOCATION1':f.write('location - 1 - ')
            if v[2]=='LOCATION2':f.write('location - 2 - ')
            if len(v[3])==1:
                f.write(v[3][0]+"\t")
            else:
                f.write("transit location\t")
            if v[4]==sentiment:
                f.write("1\n")
            else:
                f.write("0\n")
                
with open(dir_path+"dev_QA_B.tsv","w",encoding="utf-8") as f:
    f.write("id\tsentence1\tsentence2\tlabel\n")
    for v in val:
        for sentiment in sentiments:
            f.write(str(v[0])+"\t")
            word=v[1][0].lower()
            if word=='location1':f.write('location - 1')
            elif word=='location2':f.write('location - 2')
            elif word[0]=='\'':f.write("\' "+word[1:])
            else:f.write(word)
            for i in range(1,len(v[1])):
                word=v[1][i].lower()
                f.write(" ")
                if word == 'location1':
                    f.write('location - 1')
                elif word == 'location2':
                    f.write('location - 2')
                elif word[0] == '\'':
                    f.write("\' " + word[1:])
                else:
                    f.write(word)
            f.write("\t")
            f.write("the polarity of the aspect ")
            if len(v[3])==1:
                f.write(v[3][0])
            else:
                f.write("transit location")
            if v[2]=='LOCATION1':f.write(' of location - 1 - is ')
            if v[2]=='LOCATION2':f.write(' of location - 2 - is ')
            f.write(sentiment+" .\t")
            if v[4]==sentiment:
                f.write("1\n")
            else:
                f.write("0\n")

with open(dir_path+"test_NLI_B.tsv","w",encoding="utf-8") as f:
    f.write("id\tsentence1\tsentence2\tlabel\n")
    for v in test:
        for sentiment in sentiments:
            f.write(str(v[0])+"\t")
            word=v[1][0].lower()
            if word=='location1':f.write('location - 1')
            elif word=='location2':f.write('location - 2')
            elif word[0]=='\'':f.write("\' "+word[1:])
            else:f.write(word)
            for i in range(1,len(v[1])):
                word=v[1][i].lower()
                f.write(" ")
                if word == 'location1':
                    f.write('location - 1')
                elif word == 'location2':
                    f.write('location - 2')
                elif word[0] == '\'':
                    f.write("\' " + word[1:])
                else:
                    f.write(word)
            f.write("\t")
            f.write(sentiment + " - ")
            if v[2]=='LOCATION1':f.write('location - 1 - ')
            if v[2]=='LOCATION2':f.write('location - 2 - ')
            if len(v[3])==1:
                f.write(v[3][0]+"\t")
            else:
                f.write("transit location\t")
            if v[4]==sentiment:
                f.write("1\n")
            else:
                f.write("0\n")

with open(dir_path+"test_QA_B.tsv","w",encoding="utf-8") as f:
    f.write("id\tsentence1\tsentence2\tlabel\n")
    for v in test:
        for sentiment in sentiments:
            f.write(str(v[0])+"\t")
            word=v[1][0].lower()
            if word=='location1':f.write('location - 1')
            elif word=='location2':f.write('location - 2')
            elif word[0]=='\'':f.write("\' "+word[1:])
            else:f.write(word)
            for i in range(1,len(v[1])):
                word=v[1][i].lower()
                f.write(" ")
                if word == 'location1':
                    f.write('location - 1')
                elif word == 'location2':
                    f.write('location - 2')
                elif word[0] == '\'':
                    f.write("\' " + word[1:])
                else:
                    f.write(word)
            f.write("\t")
            f.write("the polarity of the aspect ")
            if len(v[3])==1:
                f.write(v[3][0])
            else:
                f.write("transit location")
            if v[2]=='LOCATION1':f.write(' of location - 1 - is ')
            if v[2]=='LOCATION2':f.write(' of location - 2 - is ')
            f.write(sentiment+" .\t")
            if v[4]==sentiment:
                f.write("1\n")
            else:
                f.write("0\n")