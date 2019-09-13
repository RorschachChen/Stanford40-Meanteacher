import random
import os

numofrandom = [25, 20, 50, 10, 75, 5]

listofdir = os.listdir('./labeled')

with open('train2.txt', 'r+') as f:
    lines = f.read().splitlines()
    for m in range(6):
        for k in range(20):
            a = random.sample([n for n in range(100)], numofrandom[m])
            with open('./labeled/' + '/' + str(listofdir[m + 1]) + '/' + ('%d' %k).zfill(2) + '.txt', 'w+')as tmp:
                for j in range(40):
                    for i in a:
                        line = lines[i + 100 * j] + '\n'
                        tmp.write(line)
