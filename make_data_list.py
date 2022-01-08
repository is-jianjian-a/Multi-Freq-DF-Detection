import os


# 举例：
# /data-x/g15/ffpp-faces/Deepfakes/c0/larger_images/860_905/000.jpg 0
# /data-x/g15/ffpp-faces/Origin/c40/larger_images/720/000.jpg 1
def makelist(method, compressibility, test=False, remake=True, step=8):
    cnt_0 = cnt_1 = 0
    # 划分数据集, test = 1 表示使用全部数据
    if test:
        trn_max = 20  # 720
        val_max = 30  # 860
        tst_max = 40  # 1000
    else:
        trn_max = 720  # 720
        val_max = 860  # 860
        tst_max = 1000  # 1000
    # 数据个数上限
    # 000-720 avg515 # 720-860 avg510 # 860-999 avg476
    # 列出路径
    txt_path = './data_list/' + method + '/' + compressibility
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)
    txt_path_trn = txt_path + '/trn.txt'
    txt_path_val = txt_path + '/val.txt'
    txt_path_tst = txt_path + '/tst.txt'
    # 删除旧的txt文件, remake = True 表示 若txt文件已存在则无需重复创建
    if remake:
        # 清除已建立的data_list
        if os.path.exists(txt_path_trn):
            os.remove(txt_path_trn)
        if os.path.exists(txt_path_val):
            os.remove(txt_path_val)
        if os.path.exists(txt_path_tst):
            os.remove(txt_path_tst)
        # 如果文件存在则跳过
    else:
        if os.path.exists(txt_path_trn) and os.path.exists(txt_path_val) and os.path.exists(txt_path_tst):
            return

    # 全部伪造方法时放入伪造图
    if method == 'All':
        method_list = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'Origin']
        step_origin = int(step / 4)
    else:
        method_list = [method, 'Origin']
        step_origin = step
    for method in method_list:
        src_path = os.path.join("/data-x/g15/ffpp-faces/", method, compressibility, 'larger_images')
        for dirpath, dirnames, filenames in os.walk(src_path):
            if not filenames:
                continue
            if method == 'Origin':
                num_video = int(dirpath[-3:])
            else:
                num_video = int(dirpath[-7:-4])
            if num_video < trn_max:
                f = open(txt_path_trn, 'a')
            elif trn_max <= num_video < val_max:
                f = open(txt_path_val, 'a')
            elif val_max <= num_video < tst_max:
                f = open(txt_path_tst, 'a')
            if method == 'Origin':
                for filename in filenames:
                    if not int(filename[0:-4]) % step_origin:
                        f.write(dirpath + '/' + filename + ' 1\n')
                        cnt_1 += 1
            else:
                for filename in filenames:
                    if not int(filename[0:-4]) % step:
                        f.write(dirpath + '/' + filename + ' 0\n')
                        cnt_0 += 1
            f.close()
    # return cnt_0, cnt_1
    print('positive examples: {}, negative examples: {}'.format(cnt_1, cnt_0))
    return


if __name__ == '__main__':
    for method in ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures', 'All']:
        for compressibility in ['c0', 'c23', 'c40']:
            print('='*20, '\n', 'making {}_{}'.format(method, compressibility))
            makelist(method, compressibility)
