list = ['applauding', 'blowing_bubbles', 'brushing_teeth', 'cleaning_the_floor', 'climbing', 'cooking', 'cutting_trees',
        'cutting_vegetables', 'drinking', 'feeding_a_horse', 'fishing', 'fixing_a_bike', 'fixing_a_car', 'gardening',
        'holding_an_umbrella', 'jumping', 'looking_through_a_microscope', 'looking_through_a_telescope', 'phoning',
        'playing_guitar', 'playing_violin', 'pouring_liquid', 'pushing_a_cart', 'reading', 'riding_a_bike',
        'riding_a_horse', 'rowing_a_boat', 'running', 'shooting_an_arrow', 'smoking', 'taking_photos',
        'texting_message', 'throwing_frisby', 'using_a_computer', 'walking_the_dog', 'washing_dishes', 'watching_TV',
        'waving_hands', 'writing_on_a_board', 'writing_on_a_book']

# with open('train.txt', 'r+') as f:
#     lines = f.read().splitlines()
#     print(lines)
#     for j in range(40):
#         for i in range(100 * j, 100 * (j + 1)):
#             lines[i] += ' ' + list[j]+'\n'
#             f.writelines(lines[i])

with open('test.txt', 'r+') as f:
    lines = f.read().splitlines()
    for name in lines:
        # [one, two] = name.split('_')
        for i in range(len(name)):
            if name[i] == '_' and name[i + 1] >= '0' and name[i + 1] <= '9':
                temp = name[0:i]
        name+=' '+ temp + '\n'
        f.writelines(name)