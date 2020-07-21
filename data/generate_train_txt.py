# coding:utf-8
import os
'''
    为数据集生成对应的txt文件
'''

import os
import cv2
import matplotlib.pyplot as plt
def gen_txt(stimuli, sal, sph, fix, target_dir):

    stimuli_list = []
    stimuli_path_list = os.listdir(stimuli)
    stimuli_path_list.sort() #　将文件名排序
    for filename in stimuli_path_list:
        stimuli_list.append(os.path.join(stimuli, filename))

    sal_list = []
    sal_path_list = os.listdir(sal)
    sal_path_list.sort() #　将文件名排序
    for filename in sal_path_list:
        sal_list.append(os.path.join(sal ,filename))

    sph_list = []
    sph_path_list = os.listdir(sph)
    sph_path_list.sort()  # 将文件名排序
    for filename in sph_path_list:
        sph_list.append(os.path.join(sph, filename))
    print(len(sph_list))

    fix_list = []
    fix_path_list = os.listdir(fix)
    fix_path_list.sort()  # 将文件名排序
    for filename in fix_path_list:
        fix_list.append(os.path.join(fix, filename))
    print(len(fix_list))

    # 将stimuli_list，sal_list，sph_list写入target_list
    target_list = []
    for value in stimuli_list:
        target_list.append([value])

    for index, value in enumerate(sal_list):
        target_list[index].append(value)

    for index, value in enumerate(sph_list):
        target_list[index].append(value)
    print(len(target_list[2]))

    for index, value in enumerate(fix_list):
        target_list[index].append(value)
    print(len(target_list[3]))

    for i in range(len(target_list)):
        print(i)

    with open(target_dir, 'w') as train:
        for i in range(len(target_list)):
            line = target_list[i][0] + ' ' + target_list[i][1] + ' ' + target_list[i][2] + ' ' + target_list[i][3] +'\n'
            train.write(line)

if __name__ == '__main__':
    # train num = 17290
    gen_txt('/home/cxl/Documents/dataset/ICME2018/train_stimuli',
            '/home/cxl/Documents/dataset/ICME2018/train_salmaps',
            '/home/cxl/Documents/dataset/ICME2018/train_sph',
            '/home/cxl/Documents/dataset/ICME2018/train_fixations',
            '/home/cxl/Documents/dataset/ICME2018/train.txt')
    # test num = 3705
    gen_txt('/home/cxl/Documents/dataset/ICME2018/test_stimuli',
            '/home/cxl/Documents/dataset/ICME2018/test_salmaps',
            '/home/cxl/Documents/dataset/ICME2018/test_sph',
            '/home/cxl/Documents/dataset/ICME2018/test_fixations',
            '/home/cxl/Documents/dataset/ICME2018/test.txt')