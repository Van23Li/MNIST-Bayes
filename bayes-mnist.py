import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import collections

def load_mnist(path='/home/van/Documents/MNIST-Bayes/data/', kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:             #读取二进制文件，用'rb'模式打开文件
        magic, numImages = struct.unpack('>II', lbpath.read(8)) 
                        ##使用struct.unpack方法读取前两个数据，>代表高位在前，I代表32位整型。lbpath.read(8)表示一次从文件中读取8个字节
                        #这样读到的前两个数据分别是magic number和样本个数
        labels = np.fromfile(lbpath, dtype=np.uint8)
                        #使用np.fromstring读取剩下的数据，lbpath.read()表示读取所有的数据

    with open(images_path, 'rb') as imgpath:
        magic, numImages, rows, cols = struct.unpack('>IIII',  imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), rows * cols)

    return images, labels


def show_images_train(img):
    # images_train, labels_train = load_mnist('/home/van/Documents/MNIST-Bayes/data/', 'train')
    # images_test, labels_test = load_mnist('/home/van/Documents/MNIST-Bayes/data/', 't10k')

    fig, ax = plt.subplots(
        nrows=2,
        ncols=5,
        sharex=True,
        sharey=True, )
 
    ax = ax.flatten()
    for i in range(10):
        # img = images_train[labels_train == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

    return True

#读取数据集
images_train, labels_train = load_mnist('/home/van/Documents/MNIST-Bayes/data/', 'train')
images_test, labels_test = load_mnist('/home/van/Documents/MNIST-Bayes/data/', 't10k')

#图片二值化
images_train = np.where(images_train>0, 1, 0)
images_test = np.where(images_test>0, 1, 0)


#计算先验概率
numImages_train_l = []
Py_l = []
for i in range(10):
    numImages_train_l.append(images_train[labels_train == i].shape[0])
    Py_l.append(numImages_train_l[i] / len(images_train))

#计算类条件概率
Pxy_1_l = []
#Pxy_0_l = []
for i in range(10):
    pxy_1_l = []
    #pxy_0_l = []
    for j in range(784):
        pxy_1_l.append((sum(images_train[labels_train == i][:,j]) + 1) / (numImages_train_l[i] + 2))
        #pxy_0_l.append(1 - pxy_1_l[j])
    Pxy_1_l.append(pxy_1_l)
    #Pxy_0_l.append(pxy_0_l)

# #计算分母
# Px_1_l = []
# Px_0_l = []
# for j in range(784):
#     px_1_l = []
#     px_0_l = []
#     for i in range(10):
#         px_1_l.append(Pxy_1_l[i][j] * Py_l[i])
#         px_0_l.append((1 - Pxy_1_l[i][j]) * Py_l[i])
#     Px_1_l.append(sum(px_1_l))
#     Px_0_l.append(sum(px_0_l))

image = images_test[4,:]
image = np.where(image>0, 1, 0)
#show_images_train(image.reshape(28, 28))
Pre_pro = []
# for i in range(10):
for i in [4]:
    pre_pro = 0
    for j in range(784):
        if image[j] == 1:
          pre_pxy = Pxy_1_l[i][j]
          #pre_px = Px_1_l[j]
        else:
          pre_pxy = 1 - Pxy_1_l[i][j]
          #pre_px = Px_0_l[j]
        pre_py = Py_l[i]
        pre_pro = pre_pro + log(pre_pxy * pre_py)
        #pre_pro = pre_pro * pre_pxy * pre_py * pre_px
    Pre_pro.append(pre_pro)

222