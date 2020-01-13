import os
import re
import subprocess
ns = [5, 10, 500, 1000]
classes = ['rgb', 'normal', 'curvature_encoding', 'mask_valid']
BASEDIR = '/mnt/hdd1/taskonomy/small/'
DRY = False
n = 5
get_point_no = lambda x: re.search(r'\d+', x).group()
get_images_from_dir = lambda wd: sorted([x for x in os.listdir(wd) if '.png' in x or '.npy in x'])

for n in ns:
    for cl in classes:
        print(f'---copying {n} from {cl} ---')
        old_dir = os.path.join(BASEDIR, cl, 'collierville')
        new_dir = os.path.join(BASEDIR, cl, f'collierville{n}')
        mkdir_cmd = f'mkdir -p {new_dir}'
        if DRY:
            print(mkdir_cmd)
        else:
            subprocess.Popen(mkdir_cmd, shell=True)
        images = get_images_from_dir(old_dir)
        seen_point_nos = set()
        for img in images:
            if len(seen_point_nos) > n:
                break
            point_no = get_point_no(img)
            if point_no not in seen_point_nos:
                seen_point_nos.add(point_no)
                src = os.path.join(old_dir, img)
                target = os.path.join(new_dir, img)
                cp_cmd = f'cp {src} {new_dir}'
                if DRY:
                    print(cp_cmd)
                else:
                    subprocess.Popen(cp_cmd, shell=True)


    