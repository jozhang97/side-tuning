# Reads the document that maps building to train/val/test split
# Used in dataloader

import csv
mid_level_train_buildings = ['beechwood']
mid_level_test_buildings = ['aloha', 'ancor', 'corder', 'duarte', 'eagan', 'globe', 'hanson', 'hatfield', 'kemblesville', 'martinville', 'sweatman', 'vails', 'wiconisco']

forbidden_buildings = ['mosquito', 'castroville', 'goodyear']

with open('train_val_test_fullplus.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')

    train_list = []
    val_list = []
    test_list = []

    for row in readCSV:
        name, is_train, is_val, is_test = row

        if is_train == '1':
            train_list.append(name)
        if is_val == '1':
            val_list.append(name)
        if is_test == '1':
            test_list.append(name)

    print(train_list)
    print(val_list)
    print(test_list)

    print(len(train_list))
    print(len(val_list))
    print(len(test_list))


l = [b for b in mid_level_test_buildings if b not in train_list]
print(l)

train_list = [b for b in train_list if b not in mid_level_test_buildings]
print(len(train_list))
print(len(mid_level_test_buildings))

