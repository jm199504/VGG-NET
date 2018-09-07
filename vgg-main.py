import tensorflow as tf
import numpy as np
import scipy.io # 图像的IO流
import scipy.misc # 图像的调整处理
import os

IMAGE_W = 800 
IMAGE_H = 600
# CONTENT_IMG是我们向做神经迁移的底图
CONTENT_IMG =  './images/scu.jpg'
# STYLE_IMG是我们想给底图添加的风格
STYLE_IMG = './images/StarryNight.jpg'
# OUTOUT_DIR和OUTPUT_IMG是输出图像
OUTOUT_DIR = './results'
OUTPUT_IMG = 'results.png'
# 采用的是ImageNet VGG模型
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
# 设置图像迁移噪声
INI_NOISE_RATIO = 0.7
# 风格图迁移强度
STYLE_STRENGTH = 500
# 迭代次数
ITERATION = 5000

CONTENT_LAYERS =[('conv4_2',1.)]
STYLE_LAYERS=[('conv1_1',1.),('conv2_1',1.),('conv3_1',1.),('conv4_1',1.),('conv5_1',1.)]

MEAN_VALUES = np.array([123, 117, 104]).reshape((1,1,1,3))

def build_net(ntype, nin, nwb=None):
  if ntype == 'conv':
    return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME')+ nwb[1])
  elif ntype == 'pool':
    return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1],
                  strides=[1, 2, 2, 1], padding='SAME')

def get_weight_bias(vgg_layers, i,):
  weights = vgg_layers[i][0][0][0][0][0]
  weights = tf.constant(weights)
  bias = vgg_layers[i][0][0][0][0][1]
  bias = tf.constant(np.reshape(bias, (bias.size)))
  return weights, bias

def build_vgg19(path):
  net = {}
  vgg_rawnet = scipy.io.loadmat(path)
  vgg_layers = vgg_rawnet['layers'][0]
  net['input'] = tf.Variable(np.zeros((1, IMAGE_H, IMAGE_W, 3)).astype('float32'))
  net['conv1_1'] = build_net('conv',net['input'],get_weight_bias(vgg_layers,0))
  net['conv1_2'] = build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2))
  net['pool1']   = build_net('pool',net['conv1_2'])
  net['conv2_1'] = build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5))
  net['conv2_2'] = build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7))
  net['pool2']   = build_net('pool',net['conv2_2'])
  net['conv3_1'] = build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10))
  net['conv3_2'] = build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12))
  net['conv3_3'] = build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14))
  net['conv3_4'] = build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16))
  net['pool3']   = build_net('pool',net['conv3_4'])
  net['conv4_1'] = build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19))
  net['conv4_2'] = build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21))
  net['conv4_3'] = build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23))
  net['conv4_4'] = build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25))
  net['pool4']   = build_net('pool',net['conv4_4'])
  net['conv5_1'] = build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28))
  net['conv5_2'] = build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30))
  net['conv5_3'] = build_net('conv',net['conv5_2'],get_weight_bias(vgg_layers,32))
  net['conv5_4'] = build_net('conv',net['conv5_3'],get_weight_bias(vgg_layers,34))
  net['pool5']   = build_net('pool',net['conv5_4'])
  return net

def build_content_loss(p, x):
  M = p.shape[1]*p.shape[2]
  N = p.shape[3]
  loss = (1./(2* N**0.5 * M**0.5 )) * tf.reduce_sum(tf.pow((x - p),2))  
  return loss


def gram_matrix(x, area, depth):
  x1 = tf.reshape(x,(area,depth))
  g = tf.matmul(tf.transpose(x1), x1)
  return g

def gram_matrix_val(x, area, depth):
  x1 = x.reshape(area,depth)
  g = np.dot(x1.T, x1)
  return g

def build_style_loss(a, x):
  M = a.shape[1]*a.shape[2]
  N = a.shape[3]
  A = gram_matrix_val(a, M, N )
  G = gram_matrix(x, M, N )
  loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A),2))
  return loss

def read_image(path):
  image = scipy.misc.imread(path)
  image = scipy.misc.imresize(image,(IMAGE_H,IMAGE_W))
  image = image[np.newaxis,:,:,:] 
  image = image - MEAN_VALUES
  return image

def write_image(path, image):
  image = image + MEAN_VALUES
  image = image[0]
  image = np.clip(image, 0, 255).astype('uint8')
  scipy.misc.imsave(path, image)


def main():
  # 加载模型
  net = build_vgg19(VGG_MODEL)
  # 创建对话
  sess = tf.Session()
  # 初始化网络中全部变量
  sess.run(tf.initialize_all_variables())
  # 随机均匀分布生成噪声图
  noise_img = np.random.uniform(-20, 20, (1, IMAGE_H, IMAGE_W, 3)).astype('float32')
  # 读取底图和风格图
  content_img = read_image(CONTENT_IMG)
  style_img = read_image(STYLE_IMG)
  # 将底图输入到输入层
  sess.run([net['input'].assign(content_img)])
  # 底图的内容损失
  cost_content = sum(map(lambda l,: l[1]*build_content_loss(sess.run(net[l[0]]) ,  net[l[0]])
    , CONTENT_LAYERS))
  # 将风格图输入到输入层
  sess.run([net['input'].assign(style_img)])
  # 风格图的风格损失
  cost_style = sum(map(lambda l: l[1]*build_style_loss(sess.run(net[l[0]]) ,  net[l[0]])
    , STYLE_LAYERS))
  # 生成风格迁移图像
  cost_total = cost_content + STYLE_STRENGTH * cost_style
  # 选择Adam优化神经网络模型
  optimizer = tf.train.AdamOptimizer(2.0)
  # 训练优化
  train = optimizer.minimize(cost_total)
  sess.run(tf.initialize_all_variables())
  sess.run(net['input'].assign( INI_NOISE_RATIO* noise_img + (1.-INI_NOISE_RATIO) * content_img))

  if not os.path.exists(OUTOUT_DIR):
      os.mkdir(OUTOUT_DIR)

  for i in range(ITERATION):
    sess.run(train)
    if i%100 ==0:
      result_img = sess.run(net['input'])
      print (sess.run(cost_total))
      write_image(os.path.join(OUTOUT_DIR,'%s.png'%(str(i).zfill(4))),result_img)
  
  write_image(os.path.join(OUTOUT_DIR,OUTPUT_IMG),result_img)
  

if __name__ == '__main__':
  main()
