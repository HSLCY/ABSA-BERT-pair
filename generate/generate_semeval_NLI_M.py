import os

data_dir='../data/semeval2014/'

dir_path = data_dir+'bert-pair/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

with open(dir_path+"test_NLI_M.csv","w",encoding="utf-8") as g:
    with open(data_dir+"Restaurants_Test_Gold.xml","r",encoding="utf-8") as f:
        s=f.readline().strip()
        while s:
            category=[]
            polarity=[]
            if "<sentence id" in s:
                left=s.find("id")
                right=s.find(">")
                id=s[left+4:right-1]
                while not "</sentence>" in s:
                    if "<text>" in s:
                        left=s.find("<text>")
                        right=s.find("</text>")
                        text=s[left+6:right]
                    if "aspectCategory" in s:
                        left=s.find("category=")
                        right=s.find("polarity=")
                        category.append(s[left+10:right-2])
                        left=s.find("polarity=")
                        right=s.find("/>")
                        polarity.append(s[left+10:right-2])
                    s=f.readline().strip()
                if "price" in category:
                    g.write(id+"\t"+polarity[category.index("price")]+"\t"+"price"+"\t"+text+"\n")
                else:
                    g.write(id + "\t" + "none" + "\t" + "price" + "\t" + text + "\n")
                if "anecdotes/miscellaneous" in category:
                    g.write(id+"\t"+polarity[category.index("anecdotes/miscellaneous")]+"\t"+"anecdotes"+"\t"+text+"\n")
                else:
                    g.write(id + "\t" + "none" + "\t" + "anecdotes" + "\t" + text + "\n")
                if "food" in category:
                    g.write(id+"\t"+polarity[category.index("food")]+"\t"+"food"+"\t"+text+"\n")
                else:
                    g.write(id + "\t" + "none" + "\t" + "food" + "\t" + text + "\n")
                if "ambience" in category:
                    g.write(id+"\t"+polarity[category.index("ambience")]+"\t"+"ambience"+"\t"+text+"\n")
                else:
                    g.write(id + "\t" + "none" + "\t" + "ambience" + "\t" + text + "\n")
                if "service" in category:
                    g.write(id+"\t"+polarity[category.index("service")]+"\t"+"service"+"\t"+text+"\n")
                else:
                    g.write(id + "\t" + "none" + "\t" + "service" + "\t" + text + "\n")
            else:
                s = f.readline().strip()


with open(dir_path+"train_NLI_M.csv","w",encoding="utf-8") as g:
    with open(data_dir+"Restaurants_Train.xml","r",encoding="utf-8") as f:
        s=f.readline().strip()
        while s:
            category=[]
            polarity=[]
            if "<sentence id" in s:
                left=s.find("id")
                right=s.find(">")
                id=s[left+4:right-1]
                while not "</sentence>" in s:
                    if "<text>" in s:
                        left=s.find("<text>")
                        right=s.find("</text>")
                        text=s[left+6:right]
                    if "aspectCategory" in s:
                        left=s.find("category=")
                        right=s.find("polarity=")
                        category.append(s[left+10:right-2])
                        left=s.find("polarity=")
                        right=s.find("/>")
                        polarity.append(s[left+10:right-1])
                    s=f.readline().strip()
                if "price" in category:
                    g.write(id+"\t"+polarity[category.index("price")]+"\t"+"price"+"\t"+text+"\n")
                else:
                    g.write(id + "\t" + "none" + "\t" + "price" + "\t" + text + "\n")
                if "anecdotes/miscellaneous" in category:
                    g.write(id+"\t"+polarity[category.index("anecdotes/miscellaneous")]+"\t"+"anecdotes"+"\t"+text+"\n")
                else:
                    g.write(id + "\t" + "none" + "\t" + "anecdotes" + "\t" + text + "\n")
                if "food" in category:
                    g.write(id+"\t"+polarity[category.index("food")]+"\t"+"food"+"\t"+text+"\n")
                else:
                    g.write(id + "\t" + "none" + "\t" + "food" + "\t" + text + "\n")
                if "ambience" in category:
                    g.write(id+"\t"+polarity[category.index("ambience")]+"\t"+"ambience"+"\t"+text+"\n")
                else:
                    g.write(id + "\t" + "none" + "\t" + "ambience" + "\t" + text + "\n")
                if "service" in category:
                    g.write(id+"\t"+polarity[category.index("service")]+"\t"+"service"+"\t"+text+"\n")
                else:
                    g.write(id + "\t" + "none" + "\t" + "service" + "\t" + text + "\n")
            else:
                s = f.readline().strip()


