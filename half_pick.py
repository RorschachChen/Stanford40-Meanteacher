import random
a = random.sample([n for n in range(100)], 50)

with open('10classes.txt', 'r+') as f:
    lines = f.read().splitlines()
    with open('500labels.txt', 'w+')as tmp:
        for j in range(10):
            for i in a:
                line = lines[i + 100 * j] + '\n'
                tmp.write(line)