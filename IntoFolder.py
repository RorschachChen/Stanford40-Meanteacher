import os
import shutil

list = ['applauding', 'blowing_bubbles', 'brushing_teeth', 'cleaning_the_floor', 'climbing', 'cooking', 'cutting_trees',
        'cutting_vegetables', 'drinking', 'feeding_a_horse', 'fishing', 'fixing_a_bike', 'fixing_a_car', 'gardening',
        'holding_an_umbrella', 'jumping', 'looking_through_a_microscope', 'looking_through_a_telescope', 'phoning',
        'playing_guitar', 'playing_violin', 'pouring_liquid', 'pushing_a_cart', 'reading', 'riding_a_bike',
        'riding_a_horse', 'rowing_a_boat', 'running', 'shooting_an_arrow', 'smoking', 'taking_photos',
        'texting_message', 'throwing_frisby', 'using_a_computer', 'walking_the_dog', 'washing_dishes', 'watching_TV',
        'waving_hands', 'writing_on_a_board', 'writing_on_a_book']

# train_path = './by-image/test'
# for single in list:
#     new_folder_path = os.path.join(train_path, single)
#     os.makedirs(new_folder_path)

with open('train2.txt', ) as f:
    lines = f.read().splitlines()
    for image_name in lines:
        [one, two] = image_name.split(' ')
        old_file_path = os.path.join('./JPEGImages', one)
        new_file_path = os.path.join('./by-image/train', two)
        shutil.move(old_file_path, new_file_path)

