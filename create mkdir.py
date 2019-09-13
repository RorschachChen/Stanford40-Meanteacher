import os
train_path = './labeled'
num_list = [200, 400, 800, 1000, 2000, 3000]
for single in num_list:
    new_folder_path = os.path.join(train_path, str(single)+'_balanced_labels')
    os.makedirs(new_folder_path)

