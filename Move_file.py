import os
import random as rd
import shutil

path_dir = "/home/leaf/Documents/melProject/Farm/"
go_to = "/home/leaf/Documents/melProject/Cam1/"

#print(os.listdir(path_dir))
elements = len(os.listdir(path_dir))
names = os.listdir(path_dir)
names.sort()
#print(names)

#print(elements)
num1 = rd.randint(1, elements)
#print("Random number = " , num1)
name_file = names[0]
shutil.move(path_dir + name_file, go_to)
