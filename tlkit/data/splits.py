import csv
import os

forbidden_buildings = ['mosquito', 'tansboro']
# forbidden_buildings = ['mosquito', 'tansboro', 'tomkins', 'darnestown', 'brinnon']
# We do not have the rgb data for tomkins, darnestown, brinnon

SPLIT_TO_NUM_IMAGES = {
    'few100': 100,
    'debug2': 100,
    'debug': 2863,
    'supersmall': 14575,
    'tiny': 262745,
    'fullplus': 3349691,
}

def get_splits(split_path):
    with open(split_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')

        train_list = []
        val_list = []
        test_list = []

        for row in readCSV:
            name, is_train, is_val, is_test = row
            if name in forbidden_buildings:
                continue
            if is_train == '1':
                train_list.append(name)
            if is_val == '1':
                val_list.append(name)
            if is_test == '1':
                test_list.append(name)
    return {
        'train': sorted(train_list),
        'val': sorted(val_list),
        'test': sorted(test_list)
    }


subsets = ['debug', 'tiny', 'medium', 'full', 'fullplus', 'supersmall', 'few5', 'few100', 'few500', 'few1000', 'debug2']
split_files = {s:  os.path.join(os.path.dirname(__file__),
                                'splits_taskonomy',
                                'train_val_test_{}.csv'.format(s.lower()))
               for s in subsets}

taskonomy = {s: get_splits(split_files[s]) for s in subsets}

midlevel = {
    'train': ['beechwood'],
    'test': ['aloha', 'ancor', 'corder', 'duarte', 'eagan', 'globe', 'hanson', 'hatfield', 'kemblesville', 'martinville', 'sweatman', 'vails', 'wiconisco']
}
taskonomy_no_midlevel = {subset: {split: sorted(buildings) for split, buildings in taskonomy[subset].items()}
                         for subset in taskonomy.keys()}
for subset, splits in taskonomy_no_midlevel.items():
    taskonomy_no_midlevel[subset]['train'] = [b for b in splits['train'] if b not in midlevel['test']]
    