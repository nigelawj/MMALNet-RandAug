
import os
import pandas as pd
images=pd.read_csv('data.csv',usecols=['img_path'])
file_name=[]
path=r'C:\Users\weimin\Desktop\dataz\data\data'
for i in images.index:
    file_name.append(os.path.join(path, images['img_path'][i])) 
        
print(file_name[0])

listOfFiles = []
for (dirpath, dirnames, filenames) in os.walk(path+'\image'):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]

for filename in listOfFiles:
    if filename not in file_name:
        os.remove(filename)

final = []
for (dirpath, dirnames, filenames) in os.walk(path+'\image'):
    final += [os.path.join(dirpath, file) for file in filenames]

print(len(final))