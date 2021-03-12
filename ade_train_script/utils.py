# -*- coding: utf-8 -*-
# Diagnostic helper functions for Tensorflow session

import tensorflow as tf
import numpy as np
import os, time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from arithmetic import adaptivearithmeticcompress as AAC
from config import directories

class Utils(object):
    
    @staticmethod
    def conv_block(x, filters, kernel_size=[3,3], strides=2, padding='same', actv=tf.nn.relu):
        in_kwargs = {'center':True, 'scale': True}
        x = tf.layers.conv2d(x, filters, kernel_size, strides=strides, padding=padding, activation=None)
        x = tf.contrib.layers.instance_norm(x, **in_kwargs)
        x = actv(x)

        return x

    @staticmethod
    def upsample_block(x, filters, kernel_size=[3,3], strides=2, padding='same', actv=tf.nn.relu):
        in_kwargs = {'center':True, 'scale': True}
        x = tf.layers.conv2d_transpose(x, filters, kernel_size, strides=strides, padding=padding, activation=None)
        x = tf.contrib.layers.instance_norm(x, **in_kwargs)
        x = actv(x)

        return x

    @staticmethod
    def residual_block(x, n_filters, kernel_size=3, strides=1, actv=tf.nn.relu):
        init = tf.contrib.layers.xavier_initializer()
        # kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
        strides = [1,1]
        identity_map = x

        p = int((kernel_size-1)/2)
        res = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
        res = tf.layers.conv2d(res, filters=n_filters, kernel_size=kernel_size, strides=strides,
                activation=None, padding='VALID')
        res = actv(tf.contrib.layers.instance_norm(res))

        res = tf.pad(res, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
        res = tf.layers.conv2d(res, filters=n_filters, kernel_size=kernel_size, strides=strides,
                activation=None, padding='VALID')
        res = tf.contrib.layers.instance_norm(res)

        assert res.get_shape().as_list() == identity_map.get_shape().as_list(), 'Mismatched shapes between input/output!'
        out = tf.add(res, identity_map)

        return out

    @staticmethod
    def get_available_gpus():
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        #return local_device_protos
        print('Available GPUs:')
        print([x.name for x in local_device_protos if x.device_type == 'GPU'])

    @staticmethod
    def scope_variables(name):
        with tf.variable_scope(name):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

    @staticmethod
    def run_diagnostics(model,G_loss, D_loss, Dis_loss,Mat_loss,config, directories, sess, saver, train_handle, start_time, epoch, name, G_loss_best, D_loss_best,img_l):
        t0 = time.time()
        improved = ''
        sess.run(tf.local_variables_initializer())
        feed_dict_test = {model.training_phase: False, model.handle: train_handle,model.img1:img_l[0],model.img2:img_l[1],model.img3:img_l[2],model.img4:img_l[3]}

        try:
            #G_loss, D_loss, summary = sess.run([model.G_loss, model.D_loss, model.merge_op], feed_dict=feed_dict_test)
            #model.train_writer.add_summary(summary)\
            G_loss, D_loss,Dis_loss,Mat_loss = sess.run([G_loss, D_loss,Dis_loss,Mat_loss], feed_dict=feed_dict_test)
        except tf.errors.OutOfRangeError:
            G_loss, D_loss = float('nan'), float('nan')

        if G_loss < G_loss_best and D_loss < D_loss_best:
            G_loss_best, D_loss_best = G_loss, D_loss
            improved = '[*]'

            save_path = saver.save(sess,
                            os.path.join(directories.checkpoints_best, '{}_epoch{}.ckpt'.format(name, epoch)),
                            global_step=epoch)
            print('Graph saved to file: {}'.format(save_path))

        if epoch % 5 == 0 and epoch > 5:
            save_path = saver.save(sess, os.path.join(directories.checkpoints, '{}_epoch{}.ckpt'.format(name, epoch)), global_step=epoch)
            print('Graph saved to file: {}'.format(save_path))

        try:
            print('Epoch {} Loss : Generator: {:.3f} | Discriminator: {:.3f} | GAN: {:.3f} | Distortion: {:.3f} | Match: {:.3f} ({:.2f} s) {}'.format(epoch, G_loss, D_loss, G_loss-Dis_loss-Mat_loss, Dis_loss,Mat_loss,time.time() - start_time, improved))
        except TypeError:
            print('Type Error Encountered! Continue...')
        return G_loss_best, D_loss_best

    @staticmethod
    def single_plot(epoch,reconstruction, global_step, sess, model, handle, name, config, img_l,single_compress=False):

        #real = model.example
        gen = reconstruction

        # Generate images from noise, using the generator network.
        g = sess.run(gen, feed_dict={model.training_phase:False, model.handle: handle,model.img1:img_l[0],model.img2:img_l[1],model.img3:img_l[2],model.img4:img_l[3]})
        g = g[0]
        #r = r[0:1]
        r = img_l[0]
        images = list()

        for im, imtype in zip([r,g], ['real', 'gen']):
            im = ((im+1.0))/2  # [-1,1] -> [0,1]
            im = np.squeeze(im)
            im = im[:,:,:3]
            images.append(im)

            # Uncomment to plot real and generated samples separately
            # f = plt.figure()
            # plt.imshow(im)
            # plt.axis('off')
            # f.savefig("{}/gan_compression_{}_epoch{}_step{}_{}.pdf".format(directories.samples, name, epoch,
            #                     global_step, imtype), format='pdf', dpi=720, bbox_inches='tight', pad_inches=0)
            # plt.gcf().clear()
            # plt.close(f)

        comparison = np.hstack(images)
        f = plt.figure()
        plt.imshow(comparison)
        plt.axis('off')
        if single_compress:
            f.savefig(name, format='png', dpi=720, bbox_inches='tight', pad_inches=0)
        else:
            f.savefig("{}/gan_compression_{}_epoch{}_step{}_{}_comparison.png".format(directories.samples, name, epoch,
                global_step, imtype), format='png', dpi=720, bbox_inches='tight', pad_inches=0)
        plt.gcf().clear()
        plt.close(f)
        
        

    @staticmethod
    def weight_decay(weight_decay, var_label='DW'):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'{}'.format(var_label)) > 0:
                costs.append(tf.nn.l2_loss(var))

        return tf.multiply(weight_decay, tf.add_n(costs))





    @staticmethod
    def single_plot_compress(epoch,thres,image,merge_op,res,reconstruction, global_step, sess, model, handle, name, config, single_compress=False):

        real = model.example
        gen = reconstruction

        # Generate images from noise, using the generator network.
        g ,merge_op,w_hat,w_mask = sess.run([gen,merge_op]+res, feed_dict={model.training_phase:False,model.thres:thres,model._x:image,model.handle: handle})
        g = g[0]
        
        w_hat = w_hat.astype(int)
        w_mask = w_mask.astype(int)
        w_hat = w_hat -1
        images = list()

        for im, imtype in zip([g], ['gen']):
            im = ((im+1.0))/2  # [-1,1] -> [0,1]
            im = np.squeeze(im)
            im = im[:,:,:3]
            images.append(im)

            # Uncomment to plot real and generated samples separately
            # f = plt.figure()
            # plt.imshow(im)
            # plt.axis('off')
            # f.savefig("{}/gan_compression_{}_epoch{}_step{}_{}.pdf".format(directories.samples, name, epoch,
            #                     global_step, imtype), format='pdf', dpi=720, bbox_inches='tight', pad_inches=0)
            # plt.gcf().clear()
            # plt.close(f)
        #g = np.squeeze(g)
        #rows,cols,channel = g.shape
        #comparison = np.hstack(images)
        #f = plt.figure()
        #plt.imshow(comparison)
        #plt.axis('off')
        nums = AAC.compress(w_hat,w_mask,5)
        #if single_compress:
         #   f.savefig(name, format='png', dpi=720, bbox_inches='tight', pad_inches=0)
        #else:
         #   f.savefig("{}/gan_compression_{}_epoch{}_step{}_thres{}_{}_bpp{}_comparison.png".format(directories.samples, name, epoch,
          #      global_step, thres,imtype,nums/rows/cols), format='png', dpi=720, bbox_inches='tight', pad_inches=0)
        #plt.gcf().clear()
        #plt.close(f)

       

        #f = plt.figure()
        #f_std = np.squeeze(w_mask)
        #plt.imshow(f_std,cmap = plt.get_cmap('gray'))
        #plt.axis('off')
        #f.savefig("{}/mask_step{}.png".format(directories.samples, global_step))
        #plt.gcf().clear()
        #plt.close(f)

        #feature_map = np.squeeze(feature_map)
        #for i in range(4):

            #f = plt.figure()
            #f_m = np.squeeze(feature_map[:,:,i])
            #plt.imshow(f_m,cmap = plt.get_cmap('gray'))
            #plt.axis('off')
            #f.savefig("{}/feature_step{}_f{}.png".format(directories.samples, global_step,i))
            #plt.gcf().clear()
            #plt.close(f)

        return merge_op,nums


    @staticmethod
    def Dense_upsample(x):
        def Time4_upsample(img):
            assert len(img.get_shape())==4,'Data Dimension should be 4'
            shape_HWC = tf.shape(img)
            N = int(img.shape[0])
            H = shape_HWC[1]
            W = shape_HWC[2]
            C = int(img.shape[3])
            assert C==4, 'Channel should be 4'

            img_1,img_2 = tf.split(img,2,axis=-1)
            img_tmp = tf.concat([double_width(tmp) for tmp in tf.split(img,2,axis=-1)],axis=-1)

            img1 = tf.transpose(img_tmp,[0,1,3,2])
            img1 = tf.reshape(img1,[N,2*H,2*W,1])
            return img1
	
        def double_width(img):
            shape_HWC = tf.shape(img)
            assert len(img.get_shape())==4,'Data Dimension should be 4'
            N = int(img.shape[0])
            H = shape_HWC[1]
            W = shape_HWC[2]
            C = int(img.shape[3])
            assert C==2, 'Channel should be 2'
            img1 = tf.reshape(img,[N,H,2*W,1])
            return img1
        total_channel = int(x.shape[3])
        assert total_channel%4==0, 'Channels shoud be interger times 4'
        up_sampled = tf.concat([Time4_upsample(x_tmp) for x_tmp in tf.split(x,num_or_size_splits=int(total_channel/4),axis=-1)],axis=-1)
        return up_sampled

