# coding: utf-8

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tf_slim as slim

class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.compat.v1.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.compat.v1.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.compat.v1.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(params=embedding, ids=self.input_x)

        with tf.compat.v1.name_scope("cnn"):
            # CNN layer
            conv = tf.compat.v1.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(input_tensor=conv, axis=[1], name='gmp')

        with tf.compat.v1.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.compat.v1.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = slim.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.compat.v1.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(input=tf.nn.softmax(self.logits), axis=1)  # 预测类别

        with tf.compat.v1.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.stop_gradient(self.input_y))
            self.loss = tf.reduce_mean(input_tensor=cross_entropy)
            # 优化器
            self.optim = tf.compat.v1.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.compat.v1.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(input=self.input_y, axis=1), self.y_pred_cls)
            self.acc = tf.reduce_mean(input_tensor=tf.cast(correct_pred, tf.float32))
