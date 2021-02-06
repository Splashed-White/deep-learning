import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

# 获取数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print('训练集数据大小：')
print(mnist.train.images.shape)
print(mnist.train.labels.shape)
print('测试集数据大小：')
print(mnist.test.images.shape)
print(mnist.test.labels.shape)
# print('验证集信息：')
# print(mnist.validation.images.shape,mnist.validation.labels.shape)

#打印第0张图片的向量表示
print(mnist.train.images[0,:])
# 打印第0张图片的标签
print(mnist.train.labels[0, :])

# 打印前5张训练图片的label
for i in range(5):
    # 得到独热表示，形如(0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    one_hot_label = mnist.train.labels[i,:]
    # 通过np.argmax我们可以直接获得原始的label
    # 因为只有1位为1，其他都是0
    label = np.argmax(one_hot_label)
    print('mnist_train_%d.jpg label: %d' % (i, label))

# 构建图
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10])) #偏置向量

y = tf.nn.softmax(tf.matmul(x,W) + b) #输出值

y_ = tf.placeholder(tf.float32, [None,10])
# 交叉熵损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 进行训练
tf.global_variables_initializer().run()

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  train_step.run({x: batch_xs, y_: batch_ys})

# 模型评估
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('MNIST手写图片准确率：')
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))