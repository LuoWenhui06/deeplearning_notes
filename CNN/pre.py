import numpy as np

#处理数据
import os
def reduce_date(mnist_path = r"C:\Users\16131\Desktop\陈学姐作业\1_MLP\data\MNIST\raw"):
##映射文件名
    name_dic ={
        'train_images':'train-images-idx3-ubyte',
        'test_images':'t10k-images-idx3-ubyte',
        'train_labels':'train-labels-idx1-ubyte',
        'test_labels':'t10k-labels-idx1-ubyte'}
##检查文件是否存在
    for name,fname in name_dic.idem():
        fpath = os.path.join(mnist_path,fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"没找到文件:{fpath}")
##解析图片
###获取图片路径并用某种能力在读的模式下打开→二进制转三维数组→归一化→四维数组
    path_tra_img = os.path.join(mnist_path,name_dic['train_images'])
    with open(path_tra_img,'rb') as f:
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype='>u4')
        tra_img = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)        
    tra_img = tra_img.astype(np.float32) / 255.0
    tra_img = tra_img.reshape(num, 1, rows, cols)

    path_tes_img = os.path.join(mnist_path,name_dic['test_images'])
    with open(path_tes_img,'rb') as f:
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype='>u4')
        tes_img = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)        
    tes_img = tes_img.astype(np.float32) / 255.0
    tes_img = tes_img.reshape(num, 1, rows, cols)

##解析标签
    path_tra_lab = os.path.join(mnist_path,name_dic['train_labels'])
    with open(path_tra_lab,'rb') as f:
        magic, num = np.frombuffer(f.read(8), dtype='>u4')
        tra_lab = np.frombuffer(f.read(), dtype=np.uint8)        
    tra_lab_oh = np.eye(10,dtype=np.float32)[tra_lab]

    path_tes_lab = os.path.join(mnist_path,name_dic['test_labels'])
    with open(path_tes_lab,'rb') as f:
        magic, num = np.frombuffer(f.read(8), dtype='>u4')
        tes_lab = np.frombuffer(f.read(), dtype=np.uint8)        
    tes_lab_oh = np.eye(10,dtype=np.float32)[tes_lab]

    print(f"训练集: {tra_img.shape}, 标签: {tra_lab_oh.shape}")
    print(f"测试集: {tes_img.shape}, 标签: {tes_lab_oh.shape}")

    return tes_img,tra_img,tra_lab_oh,tes_lab_oh

    