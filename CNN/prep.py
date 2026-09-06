import numpy as np

#处理数据
import os
def reduce_date(path_mnist = r"C:\Users\16131\Desktop\陈学姐作业\1_MLP\data\MNIST\raw"):

##映射文件名
    name_dic ={
        'train_images':'train-images-idx3-ubyte',
        'test_images':'t10k-images-idx3-ubyte',
        'train_labels':'train-labels-idx1-ubyte',
        'test_labels':'t10k-labels-idx1-ubyte'}
    
##检查文件是否存在
    for name,fname in name_dic.idem():
        fpath = os.path.join(path_mnist,fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"没找到文件:{fpath}")
        
##解析图片：获取图片路径并用某种能力在读的模式下打开→二进制转三维数组→归一化→四维数组
    def parse_image(path_image):
        with open(path_image,'rb') as f:
            magic, num, rows, cols = np.frombuffer(f.read(16), dtype='>u4')
            img = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)        
            img = img.astype (np.float32) / 255.0
            img = img.reshape(num, 1, rows, cols)   
        return img 

    tra_img = parse_image(os.path.join(path_mnist,name_dic['train_images'])) 
    tes_img = parse_image(os.path.join(path_mnist,name_dic['test_images']))
##解析标签：获取标签路径并用某种能力在读的模式下打开→二进制转整数索引→转one_hot编码
    def parse_lab(path_label):
        with open(path_label,'rb') as f:
            magic, num = np.frombuffer(f.read(8), dtype='>u4')
            lab = np.frombuffer(f.read(), dtype=np.uint8)        
            lab_oh = np.eye(10,dtype=np.float32)[lab]
        return lab_oh
    
    tra_lab_oh = parse_lab(os.path.join(path_mnist,name_dic['train_labels']))
    tes_lab_oh = parse_lab(os.path.join(path_mnist,name_dic['test_labels']))

    print(f"训练集: {tra_img.shape}, 标签: {tra_lab_oh.shape}")
    print(f"测试集: {tes_img.shape}, 标签: {tes_lab_oh.shape}")

    return tes_img,tra_img,tra_lab_oh,tes_lab_oh

#卷积层
##    