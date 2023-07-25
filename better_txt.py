import os 

path = './log'
old_doc = os.path.join(path,'progress.txt')
new_doc = os.path.join(path,'progress_new.txt')

new_file = open(new_doc, "rb")

with open(old_doc, "r", encoding="utf-8") as file:  
    number = 0  # 记录行号
    while True:
        number += 1
        line = file.readline()
        if line == "":
            break
        print(number, line, end="\n")
print("\n ", "=" * 20, "over" * 20, "\n")
