import tensorflow as tf
import numpy as np
import os
import IC_datasets
import data_loader
from tensorflow.contrib.layers.python.layers import batch_norm
from scipy.misc import imsave
import cv2
# import xlwt
from time import time

ICdata_name = "test_data"
#total_batch = int(IC_datasets.DATASET_TO_SIZES[ICdata_name])
test_batch = int(IC_datasets.DATASET_TO_SIZES['test_data']/IC_datasets.batch_size)
weight_init = tf.contrib.layers.variance_scaling_initializer() # kaming init for encoder / decoder
weight_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)

def conv2d(x, channels, kernel=4, stride=(1,1), pad_type="SAME", use_bias=True, name="conv"):
    with tf.variable_scope(name):
        if name.__contains__("discriminator") :
            conv_weight = tf.random_normal_initializer(mean=0.0, stddev=0.02)
            conv_regularizer = weight_regularizer
        else :
            conv_weight = tf.contrib.layers.variance_scaling_initializer()
            conv_regularizer= None
        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=conv_weight,
                             kernel_regularizer=conv_regularizer,
                             strides=stride, padding=pad_type, use_bias=use_bias)
        return x

def deconv(x, channels, kernel=3, stride=(1,1), pad_type="SAME", use_bias=True, name='deconv'):
    with tf.variable_scope(name):
        x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init ,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, padding=pad_type, use_bias=use_bias)
        return x

def linear(x, units, use_bias=True, name="linear"):
    with tf.variable_scope(name):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)
        return x

def max_pool(x, kernel=2, stride=2):
    return tf.nn.max_pool(x, ksize=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], padding="SAME")

def avg_pool(x, kernel=2, stride=2):
    return tf.nn.avg_pool(x, ksize=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], padding="SAME")

def relu(x):
    return tf.nn.relu(x)

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)

def flatten(x):
    return tf.contrib.layers.flatten(x)

def batch_norm_layer(x, is_tain=True):
    return batch_norm(x, decay=0.9, updates_collections=None, is_training=is_tain)

def tanh(x):
    return tf.tanh(x)

def discriminator_loss(real, fake):
    real_loss = tf.losses.mean_squared_error(tf.ones_like(real), real)
    fake_loss = tf.losses.mean_squared_error(tf.zeros_like(fake), fake)
    d_loss = (real_loss + fake_loss) / 2.0
    return d_loss, real_loss, fake_loss

def generator_loss(fake):
    g_loss = tf.losses.mean_squared_error(tf.ones_like(fake), fake)
    #feature_loss = L1_loss(real_feature, fake_feature)
    #loss = g_loss + feature_loss
    return g_loss

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss

def upsample(x, h_sacle, w_scale):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * h_sacle, w * w_scale]
    return tf.image.resize_nearest_neighbor(x, size=new_size)


def generator(z, train_sign=True, reuse=False, scope="generator"):
    with tf.variable_scope(scope, reuse=reuse):
        ch = 1024
        x = lrelu(batch_norm_layer(conv2d(z, ch, 4, (1, 1), use_bias=False, name='g0'), train_sign), 0.02)  # [-1,7,2,1024]
        for i in range(3):
            ch = ch // 2
            if i == 0:
                x = upsample(x, 3, 1)
            else:
                x = upsample(x, 2, 1)
            x = lrelu(batch_norm_layer(conv2d(x, ch, 3, (1, 1), name='g1_%d'%(i)), train_sign), 0.02)
            ch = ch // 2
            x = upsample(x, 1, 2)
            x = lrelu(batch_norm_layer(conv2d(x, ch, 3, (1, 1), name='g2_%d'%(i)), train_sign), 0.02)
        x = conv2d(x, 3, 3, (1, 1), name='g3')
        out = tanh(x)
        return out

def encoder(x, train_sign=True, reuse=False, scope="encoder"):
    with tf.variable_scope(scope, reuse=reuse):
        ch = 16
        x = lrelu(batch_norm_layer(conv2d(x, ch, 3, (1, 1), use_bias=False, name='g0'), train_sign), 0.02)  # [-1,7,2,1024]
        for i in range(3):
            ch = ch * 2
            x = lrelu(batch_norm_layer(conv2d(x, ch, 3, (1, 2), name='g2_%d' % (i)), train_sign), 0.02)
            ch = ch * 2
            if i == 2:
                x = lrelu(batch_norm_layer(conv2d(x, ch, 3, (3, 1), name='g1_%d' % (i)), train_sign), 0.02)
            else:
                x = lrelu(batch_norm_layer(conv2d(x, ch, 3, (2, 1), name='g1_%d' % (i)), train_sign), 0.02)
        out = conv2d(x, 16, 4, (1, 1), name='g3')
        return out

def discriminator(x, train_sign=True, reuse=False, scope="discriminator"):
    with tf.variable_scope(scope, reuse=reuse):
        ch=16
        y = lrelu(batch_norm_layer(conv2d(x, ch, 3, 1, name='d0'), train_sign), 0.02)
        for i in range(3):
            ch = ch * 2
            y = lrelu(batch_norm_layer(conv2d(y, ch, 3, 1, name='d1_%d'%(i)), train_sign), 0.02)
            y = tf.nn.max_pool(y,[1,2,1,1],[1,2,1,1],'SAME' )
            ch = ch * 2
            y = lrelu(batch_norm_layer(conv2d(y, ch, 3, 1, name='d2_%d'%(i)), train_sign), 0.02)
            y = tf.nn.max_pool(y,[1,1,2,1],[1,1,2,1],'SAME' )
            if i == 2:
                out2 = y
        out = conv2d(y, 1024, 3, 1, use_bias=False, name='d3')
        return out, out2
def get_list(foldername, suffix=".jpg"):
    file_list_tmp = os.listdir(foldername)
    file_list = []
    name_list = []
    label_list = []
    for item in file_list_tmp:
        if item.endswith(suffix):
            file_list.append(os.path.join(foldername, item))
            name_list.append(item)
            label = item.split('_')[0]
            label_list.append(label)
    return file_list, name_list, label_list

def get_label(flie_list):
    label_list = []
    for item in flie_list:
        label = item.split('/')[-1].split('_')[0]
        label_list.append(label)
    return label_list

def creat_fmimg(src_list, gen_list, label_list, creat_num, epoch):
    added_img = np.ones((84, 16), dtype='uint8')
    img_num = 0
    for src_filename, gen_filename, label in zip(src_list, gen_list, label_list):
        if int(label) == 1:
            continue
        src_img = cv2.imread(src_filename)
        gen_img = cv2.imread(gen_filename)

        gen_hsv = cv2.cvtColor(gen_img, cv2.COLOR_BGR2HSV)
        (gen_h, gen_s, gen_v) = cv2.split(gen_hsv)
        src_hsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
        (src_h, src_s, src_v) = cv2.split(src_hsv)
        cv2.absdiff(src_h, gen_h, src_h)
        add_img = np.where(src_h < 30, 0, 1)
        added_img = np.add(added_img, add_img)
        img_num = img_num + 1
        if img_num == creat_num:
            break
    fm_name =  'fm_' + str(epoch) + '.jpg'
    cv2.imwrite(os.path.join('./output', fm_name), added_img)
    return added_img

def img_process(epoch, src_dir, gen_dir, fm_num):
    right_num=0
    error_num=0
    omiss_num=0
    src_list, name_list, label_list = get_list(src_dir)
    gen_list, _, _ = get_list(gen_dir)
    fm_img = creat_fmimg(src_list, gen_list, label_list, fm_num, epoch)
    v_img = fm_num/fm_img
    fenmu = np.max(v_img)-np.min(v_img)
    v_img = v_img/fenmu
    #v_img = v_img[56:70,0:16]
    for src_filename, gen_filename, label, name in zip(src_list, gen_list, label_list, name_list):
        src_img = cv2.imread(src_filename)
        gen_img = cv2.imread(gen_filename)

        gen_hsv = cv2.cvtColor(gen_img, cv2.COLOR_BGR2HSV)
        (gen_h, gen_s ,gen_v) = cv2.split(gen_hsv)
        src_hsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
        (src_h, src_s ,src_v) = cv2.split(src_hsv)
        cv2.absdiff(src_h, gen_h, src_h)
        #cv2.imwrite(os.path.join('./test_output/hsv', name), src_h)
        src_h = np.where(src_h < 25, 0, 1)  #30
        src_v = np.where(src_v < 150, 0, 1)  #100
        src_h = cv2.bitwise_and(src_h, src_v)
        #src_h = src_h[56:70,0:16]
        #cv2.imwrite(os.path.join('./test_output/src_h', name), src_h*255)
        scrop = np.sum(np.multiply(v_img, src_h))
        if(scrop >= 5):
            img_label = 1
        else:
            img_label = 0

        if(img_label == int(label)):
            right_num = right_num + 1
        else:
            error_num = error_num + 1

        if int(label) == 1 and img_label == 0:
            omiss_num = omiss_num + 1
    return right_num, error_num, omiss_num

def main():
    img = tf.placeholder(tf.float32, [None, 84, 16, 3])
    train_sign = tf.placeholder(tf.bool)

    z = encoder(img, train_sign)
    gen_img = generator(z, train_sign)

    t_vars = tf.all_variables()
    G_vars = [var for var in t_vars if 'generator' in var.name]
    encoder_vars = [var for var in t_vars if 'encoder' in var.name]

    testdata_input = data_loader.load_data('test_data', do_shuffle=False, one_hot=False)
    # saver to save model
    GE_saver = tf.train.Saver(G_vars + encoder_vars)

    if not os.path.exists('./output'):
        os.makedirs('./output')

    star_time = time()
    with tf.Session() as sess:
        with tf.device('/cpu:0'): #"/gpu:0"
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            GE_saver.restore(sess, os.path.join(IC_datasets.model_savedir, "ic_400"))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            gen_dir_name = './output/gen_image'
            src_dir_name = './output/src_image'
            if not os.path.exists(gen_dir_name):
                os.makedirs(gen_dir_name)
            if not os.path.exists(src_dir_name):
                os.makedirs(src_dir_name)

            image_num = 0
            for testepoch in range(test_batch):
                test_inputs = sess.run(testdata_input)
                generator_img = gen_img.eval(feed_dict={img: test_inputs['image'], train_sign: False}, session=sess)
                for num in range(0, IC_datasets.batch_size):
                    image_num = image_num + 1
                    image_name = str(test_inputs['label'][num]) + "_" + str(image_num) + ".jpg"
                    src_name = str(test_inputs['label'][num]) + "_" + str(image_num) + ".jpg"
                    save_img = generator_img[num].reshape((84, 16, 3))
                    src_img = test_inputs['image'][num].reshape((84, 16, 3))
                    imsave(os.path.join(gen_dir_name, image_name),
                           ((save_img + 1) * 127.5).astype(np.uint8))
                    imsave(os.path.join(src_dir_name, src_name),
                           ((src_img + 1) * 127.5).astype(np.uint8))

            coord.request_stop()
            coord.join(threads)

        num_rignt, num_error, num_omiss = img_process(260, src_dir_name, gen_dir_name, 260)
        print("num_rignt=%d, num_error=%d, num_omiss=%d"%(num_rignt, num_error, num_omiss))
        print("time = %f s"%(time()-star_time))


if __name__ == '__main__':
    main()
