#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd
import time, os, sys
import argparse

# User-defined
from network import Network
from utils import Utils
from data import Data
from model import Model
from config import config_test, directories

tf.logging.set_verbosity(tf.logging.ERROR)

def single_compress(config, args):
    start = time.time()
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)
    assert (ckpt.model_checkpoint_path), 'Missing checkpoint file!'

    if config.use_conditional_GAN:
        print('Using conditional GAN')
        paths, semantic_map_paths = np.array([args.image_path]), np.array([args.semantic_map_path])
    else:
        paths = Data.load_dataframe(directories.train)
    ngpus = 1
    gan = Model(config, paths, name='single_compress', dataset=args.dataset, npgus = ngpus, evaluate=True)
    saver = tf.train.Saver()

    if config.use_conditional_GAN:
        feed_dict_init = {gan.path_placeholder: paths,
                          gan.semantic_map_path_placeholder: semantic_map_paths}
    else:
        feed_dict_init = {gan.path_placeholder: paths}

    G_opt = tf.train.AdamOptimizer(learning_rate=config.G_learning_rate, beta1=0.5)
    D_opt = tf.train.AdamOptimizer(learning_rate=config.D_learning_rate, beta1=0.5)
    tower_G_grads = []
    tower_D_grads = []	
    G_loss_total = []
    D_loss_total = []
    Distor_total = []
    Match_total  = []
    reconstruction_total = []	
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(ngpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower%d' % i) as scope:
                    _x = gan.example[i*config.batch_size:(i+1)*config.batch_size]
#print(_x.shape)
#time.sleep()
                    D_loss,G_loss, reconstruction,distortion_penalty,match_loss,Flag = Model.create_model(_x,gan.training_phase,config)
                    reconstruction_total.append(reconstruction)
                    tf.add_to_collection('Glosses',G_loss)
                    tf.add_to_collection('Dlosses',D_loss)
                    tf.get_variable_scope().reuse_variables()
                    theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                    theta_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
#print(theta_D)
                    G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
                    D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
                    G_losses = tf.get_collection('Glosses',scope)
                    G_loss  = tf.add_n(G_losses)
                    D_losses = tf.get_collection('Dlosses',scope)
                    D_loss  = tf.add_n(D_losses)
                    with tf.control_dependencies(G_update_ops):
                        G_grads_all = G_opt.compute_gradients(G_loss,   var_list=theta_G)
                    with tf.control_dependencies(D_update_ops):
                        D_grads_all = D_opt.compute_gradients(D_loss,   var_list=theta_D)
                    G_grads = [(g,v) for (g,v) in G_grads_all if g is not None]
                    D_grads = [(g,v) for (g,v) in D_grads_all if g is not None]
#G_grads = G_opt.compute_gradients(G_loss, G_grads_var  )
#D_grads = D_opt.compute_gradients(D_loss,  D_grads_var )
                    tower_G_grads.append(G_grads)
                    tower_D_grads.append(D_grads)
                    G_loss_total.append(G_loss)
                    D_loss_total.append(D_loss)
                    Distor_total.append(distortion_penalty)
                    Match_total.append(match_loss)
						#print(D_grads)
    G_grads_total = Model.average_gradients(tower_G_grads)
    D_grads_total = Model.average_gradients(tower_D_grads)
    G_loss_mean  = tf.reduce_mean(G_loss_total)
    D_loss_mean  = tf.reduce_mean(D_loss_total)
    Distor_mean  = tf.reduce_mean(Distor_total)
    Match_mean   = tf.reduce_mean(Match_total)
		
    G_opt_op = G_opt.apply_gradients(G_grads_total)
    D_opt_op = D_opt.apply_gradients(D_grads_total)
    G_global_step_op  = tf.assign(gan.G_global_step,tf.add(gan.G_global_step,1))
    D_global_step_op  = tf.assign(gan.D_global_step,tf.add(gan.D_global_step,1))
    theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    theta_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    with tf.control_dependencies([G_global_step_op,G_opt_op]):
        G_ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=gan.G_global_step)
        G_maintain_averages_op = G_ema.apply(theta_G)
    with tf.control_dependencies([D_global_step_op,D_opt_op]):
        D_ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=gan.D_global_step)
        D_maintain_averages_op = D_ema.apply(theta_D)
#with tf.control_dependencies([G_opt_op]):
    G_train_op = tf.group(G_maintain_averages_op)
#with tf.control_dependencies([D_opt_op]):
    D_train_op = tf.group(D_maintain_averages_op)


    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        handle = sess.run(gan.train_iterator.string_handle())

        if args.restore_last and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Most recent {} restored.'.format(ckpt.model_checkpoint_path))
        else:
            if args.restore_path:
                new_saver = tf.train.import_meta_graph('{}.meta'.format(args.restore_path))
                new_saver.restore(sess, args.restore_path)
                print('Previous checkpoint {} restored.'.format(args.restore_path))

        sess.run(gan.train_iterator.initializer, feed_dict=feed_dict_init)
        eval_dict = {gan.training_phase: False, gan.handle: handle}

        if args.output_path is None:
            output = os.path.splitext(os.path.basename(args.image_path))
            save_path = os.path.join(directories.samples, '{}_compressed.pdf'.format(output[0]))
        else:
            save_path = args.output_path
        step = 0
        while True:
            try:
                Utils.single_plot(0, reconstruction_total,step, sess, gan, handle, save_path, config)
                print('Reconstruction saved to', save_path)
                step += 1
            except tf.errors.OutOfRangeError:
                print('End of epoch!')
                break
    return


def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("-rl", "--restore_last", help="restore last saved model", action="store_true")
    parser.add_argument("-r", "--restore_path", help="path to model to be restored", type=str)
    parser.add_argument("-i", "--image_path", help="path to image to compress", type=str)
    parser.add_argument("-sm", "--semantic_map_path", help="path to corresponding semantic map", type=str)
    parser.add_argument("-o", "--output_path", help="path to output image", type=str)
    parser.add_argument("-ds", "--dataset", default="cityscapes", help="choice of training dataset. Currently only supports cityscapes/ADE20k", choices=set(("cityscapes", "ADE20k")), type=str)
    args = parser.parse_args()

    # Launch training
    single_compress(config_test, args)

if __name__ == '__main__':
    main()
