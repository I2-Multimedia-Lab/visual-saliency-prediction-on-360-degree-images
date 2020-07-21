
# coding:utf-8
import os
'''
    为数据集生成对应的txt文件
'''

test_txt_path = os.path.join("/home/cxl/Documents/papercode/vis_sal/split3D/test", "bin.txt")
test_dir = os.path.join("/home/cxl/Documents/papercode/vis_sal/split3D", "test")

def gen_txt(txt_path, img_dir):
    f = open(txt_path, 'w')

    for root, s_dirs, _ in os.walk(img_dir, topdown=True):
        s_dirs.sort() # 将子文件夹排序
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)
            img_list = os.listdir(i_dir) # 将子文件排序
            img_list.sort()
            for i in range(len(img_list)):
                if not img_list[i].endswith('bin'):
                    continue
                img_path = os.path.join(i_dir, img_list[i])
                line = img_path +'\n'
                f.write(line)

    f.close()

if __name__ == '__main__':
    gen_txt(test_txt_path, test_dir)
