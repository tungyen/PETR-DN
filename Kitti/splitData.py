import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)

def main():
    random.seed(0)
    split_rate = 0.1

    cwd = os.getcwd()
    root = os.path.join(cwd, "calib")
    root = os.path.join(root, "training", "calib")

    names = os.listdir(root)
    names = [name.split('.')[0] for name in names]
    num = len(names)
    
    evalIndex = random.sample(names, k=int(num*split_rate))
    evals = [name for name in names if name in evalIndex]
    trains = [name for name in names if name not in evalIndex]
    
    with open("eval.txt", "w") as file:
        for eval in evals:
            file.write(eval + "\n")
    file.close()        
    
    with open("train.txt", "w") as file:
        for train in trains:
            file.write(train + "\n")
    file.close()
    
    with open("test.txt", 'w') as file:
        for name in names:
            file.write(name+"\n")
    file.close()
    print("processing done!")

if __name__ == '__main__':
    main()