import os

data_dir='../data/semeval2014/'

aspect_name = ['price', 'anecdotes', 'food', 'ambience', 'service']
dir_path = [data_dir + 'bert-single/' + i + '/' for i in aspect_name]
for path in dir_path:
    if not os.path.exists(path):
        os.makedirs(path)

with open(dir_path[0]+"test.csv", "w", encoding="utf-8") as g_price, \
    open(dir_path[1]+"test.csv", "w", encoding="utf-8") as g_anecdotes,\
    open(dir_path[2]+"test.csv", "w", encoding="utf-8") as g_food,\
    open(dir_path[3]+"test.csv", "w", encoding="utf-8") as g_ambience,\
    open(dir_path[4]+"test.csv", "w", encoding="utf-8") as g_service,\
    open(data_dir+"Restaurants_Test_Gold.xml","r",encoding="utf-8") as f:
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
                g_price.write(id+"\t"+polarity[category.index("price")]+"\t"+"price"+"\t"+text+"\n")
            else:
                g_price.write(id + "\t" + "none" + "\t" + "price" + "\t" + text + "\n")
            if "anecdotes/miscellaneous" in category:
                g_anecdotes.write(id+"\t"+polarity[category.index("anecdotes/miscellaneous")]+"\t"+"anecdotes"+"\t"+text+"\n")
            else:
                g_anecdotes.write(id + "\t" + "none" + "\t" + "anecdotes" + "\t" + text + "\n")
            if "food" in category:
                g_food.write(id+"\t"+polarity[category.index("food")]+"\t"+"food"+"\t"+text+"\n")
            else:
                g_food.write(id + "\t" + "none" + "\t" + "food" + "\t" + text + "\n")
            if "ambience" in category:
                g_ambience.write(id+"\t"+polarity[category.index("ambience")]+"\t"+"ambience"+"\t"+text+"\n")
            else:
                g_ambience.write(id + "\t" + "none" + "\t" + "ambience" + "\t" + text + "\n")
            if "service" in category:
                g_service.write(id+"\t"+polarity[category.index("service")]+"\t"+"service"+"\t"+text+"\n")
            else:
                g_service.write(id + "\t" + "none" + "\t" + "service" + "\t" + text + "\n")
        else:
            s = f.readline().strip()


with open(dir_path[0]+"train.csv", "w", encoding="utf-8") as g_price, \
    open(dir_path[1]+"train.csv", "w", encoding="utf-8") as g_anecdotes,\
    open(dir_path[2]+"train.csv", "w", encoding="utf-8") as g_food,\
    open(dir_path[3]+"train.csv", "w", encoding="utf-8") as g_ambience,\
    open(dir_path[4]+"train.csv", "w", encoding="utf-8") as g_service,\
    open(data_dir+"Restaurants_Train.xml","r",encoding="utf-8") as f:
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
                g_price.write(id+"\t"+polarity[category.index("price")]+"\t"+"price"+"\t"+text+"\n")
            else:
                g_price.write(id + "\t" + "none" + "\t" + "price" + "\t" + text + "\n")
            if "anecdotes/miscellaneous" in category:
                g_anecdotes.write(id+"\t"+polarity[category.index("anecdotes/miscellaneous")]+"\t"+"anecdotes"+"\t"+text+"\n")
            else:
                g_anecdotes.write(id + "\t" + "none" + "\t" + "anecdotes" + "\t" + text + "\n")
            if "food" in category:
                g_food.write(id+"\t"+polarity[category.index("food")]+"\t"+"food"+"\t"+text+"\n")
            else:
                g_food.write(id + "\t" + "none" + "\t" + "food" + "\t" + text + "\n")
            if "ambience" in category:
                g_ambience.write(id+"\t"+polarity[category.index("ambience")]+"\t"+"ambience"+"\t"+text+"\n")
            else:
                g_ambience.write(id + "\t" + "none" + "\t" + "ambience" + "\t" + text + "\n")
            if "service" in category:
                g_service.write(id+"\t"+polarity[category.index("service")]+"\t"+"service"+"\t"+text+"\n")
            else:
                g_service.write(id + "\t" + "none" + "\t" + "service" + "\t" + text + "\n")
        else:
            s = f.readline().strip()

print("Finished!")

