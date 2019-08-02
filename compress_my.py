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
from config import config_train, directories
import csv
#from numpyfile import test_pca
tf.logging.set_verbosity(tf.logging.ERROR)

def train(config, args):
	with tf.device("/cpu:0"):
		start_time = time.time()
		G_loss_best, D_loss_best = float('inf'), float('inf')
		ckpt = tf.train.get_checkpoint_state(directories.checkpoints)

		# Load data
		print('Training on dataset', args.dataset)
		if config.use_conditional_GAN:
			print('Using conditional GAN')
			paths, semantic_map_paths = Data.load_dataframe(directories.train, load_semantic_maps=True)
			test_paths, test_semantic_map_paths = Data.load_dataframe(directories.test, load_semantic_maps=True)
		else:
			paths = Data.load_dataframe(directories.train)
			test_paths = Data.load_dataframe(directories.val)

	    # Build graph	
		ngpus = 1
		gan = Model(config, paths, name=args.name, dataset=args.dataset,ngpus = ngpus)
		

		if config.use_conditional_GAN:
			feed_dict_test_init = {gan.test_path_placeholder: test_paths, 
					       gan.test_semantic_map_path_placeholder: test_semantic_map_paths}
			feed_dict_train_init = {gan.path_placeholder: paths,
						gan.semantic_map_path_placeholder: semantic_map_paths}
		else:
			feed_dict_test_init = {gan.test_path_placeholder: test_paths}
			feed_dict_train_init = {gan.path_placeholder: paths}

		
		G_opt = tf.train.AdamOptimizer(learning_rate=config.G_learning_rate, beta1=0.5)
		D_opt = tf.train.AdamOptimizer(learning_rate=config.D_learning_rate, beta1=0.5)
		tower_G_grads = []
		tower_D_grads = []	
		G_loss_total = []
		D_loss_total = []
		Distor_total = []
		Match_total  = []
		reconstruction_total = []
		vgg_total = []	
		#thres = tf.placeholder(tf.float32, shape=[])
		with tf.variable_scope(tf.get_variable_scope()):
			for i in range(ngpus):
				with tf.device('/gpu:%d' % i):
					with tf.name_scope('tower%d' % i) as scope:
						image = gan.example[i*config.batch_size:(i+1)*config.batch_size]
						
						#print(_x.shape)
						#time.sleep()
						D_loss,G_loss, reconstruction,distortion_penalty,match_loss,Flag,merge_op,*res = Model.create_model(gan._x,gan._pca,gan.training_phase,config,gan.thres)
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
						#vgg_total.append(vgg_loss)
						#print(D_grads)
		G_grads_total = Model.average_gradients(tower_G_grads)
		D_grads_total = Model.average_gradients(tower_D_grads)
		G_loss_mean  = tf.reduce_mean(G_loss_total)
		D_loss_mean  = tf.reduce_mean(D_loss_total)
		Distor_mean  = tf.reduce_mean(Distor_total)
		Match_mean   = tf.reduce_mean(Match_total)
		#vgg_mean     = tf.reduce_mean(vgg_total)
		#print(vgg_mean)
		#G_grads_holder = [(tf.placeholder(tf.float32,shape=g.get_shape()),v) for (g,v) in G_grads]
		#D_grads_holder = [(tf.placeholder(tf.float32,shape=g.get_shape()),v) for (g,v) in D_grads]


#=================================================Update Parameters=====================================================
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
		saver = tf.train.Saver(max_to_keep=15)
		#print(D_grads_total)
		train_writer = tf.summary.FileWriter(
		    os.path.join(directories.tensorboard, '{}_train_{}'.format(args.name, time.strftime('%d-%m_%I:%M'))), graph=tf.get_default_graph())


		ori_image,height_i,width_i = _aspect_preserving_width_resize(image)
#============================================Start Session==============================================================
		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())
			train_handle = sess.run(gan.train_iterator.string_handle())
			test_handle = sess.run(gan.test_iterator.string_handle())

			if args.restore_last and ckpt.model_checkpoint_path:
			    # Continue training saved model
			    saver.restore(sess, ckpt.model_checkpoint_path)
			    print('{} restored.'.format(ckpt.model_checkpoint_path))
			else:
				if args.restore_path:
					new_saver = tf.train.import_meta_graph('{}.meta'.format(args.restore_path))
					new_saver.restore(sess, args.restore_path)
					print('{} restored.'.format(args.restore_path))

			sess.run(gan.test_iterator.initializer, feed_dict=feed_dict_test_init)
			step =0
			
			PSNR_total = 0
			SSIM_total = 0
			PSNR_total_r = 0
			SSIM_total_r = 0
			thres_l = [-0.1]
			AACave = [0]*len(thres_l)
			w_numpy = []
			for epoch in range(config.num_epochs):

				sess.run(gan.train_iterator.initializer, feed_dict=feed_dict_train_init)
			
				# Run diagnostics
				while step<6000:
					try:
						# Update generator
						#for _ in range(8):
						
						# Update discriminator 
						#step,_,F,m = sess.run([gan.D_global_step,D_train_op,Flag,merge_op], feed_dict=feed_dict)
						step += 1
						oriimage,height,width = sess.run([ori_image,height_i,width_i],feed_dict={gan.handle: test_handle})

						#if step<=460: continue
						#if step%2 == 0:
							
						for cnt in range(len(thres_l)):#[-0.1,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.2]:

							AACNUM,w_hat = Utils.single_plot_compress(epoch,thres_l[cnt],oriimage,merge_op, res,reconstruction_total,step, sess, gan, test_handle, args.name, config,height,width)
							#SSIM_total_r += SSIM_r
							#PSNR_total_r += PSNR_r
							#w_newshape = w_hat.reshape(1,-1)
							#w_numpy.append(w_newshape)
							#train_writer.add_summary(m,step)
							AACave[cnt] += AACNUM
							print("Current Image: {} threshlod: {} bpp: {} Average bpp: {}  ".format(step,thres_l[cnt],AACNUM,AACave[cnt]/step))
							#if step%1 == 0:
								#with open("numpyfile/ADEtrain{}.csv".format(step),'w') as csvfile:
									#writer = csv.writer(csvfile)
									#for fp in w_numpy:
										#writer.writerow(fp)
										#file_img.write('\n')
									#file_img.close()
								#feature_save = np.vstack(w_numpy)
								#print(feature_save.shape)
								#np.save("numpyfile/ADEtrain{}.npy".format(step),feature_save)
								#print('File Saved!')
							#PSNR_total += PSNR
							#SSIM_total +=SSIM
							#PSNR_total_r += PSNR_r
							#SSIM_total_r +=SSIM_r
							#print("step: {} | threshold:{} | bpp: {}  PSNR:{}  SSIM: {} | After Reduce: PSNR:{}  SSIM: {}".format(step,thres,AACNUM/512/1024,PSNR,SSIM,PSNR_r,SSIM_r))
						
					


					except tf.errors.OutOfRangeError:
						print('End of epoch!')
						#feature_save = np.array(w_numpy)
						#np.save("numpyfile/train.npy",feature_save)
						#print('File Saved!')
						break


			#save_path = saver.save(sess, os.path.join(directories.checkpoints,
				       #'{}_end.ckpt'.format(args.name)),
				       #global_step=epoch)

		#print("Training Complete. Model saved to file: {} Time elapsed: {:.3f} s".format(save_path, time.time()-start_time))

def _aspect_preserving_width_resize(image, width=512):
                height_i = tf.shape(image)[1]
                width_i = tf.shape(image)[2]
                #ratio = tf.to_float(height_i) / tf.to_float(width_i)
                #new_height = tf.to_int32(tf.to_float(width) * ratio)
                new_height = height_i - tf.floormod(height_i, 16)
                new_width  = width_i - tf.floormod(width_i, 16)
                tf_size = tf.stack([new_height, new_width ],axis=0)
                return tf.image.resize_images(image, tf_size),height_i,width_i



def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("-rl", "--restore_last", help="restore last saved model", action="store_true")
    parser.add_argument("-r", "--restore_path", help="path to model to be restored", type=str)
    parser.add_argument("-opt", "--optimizer", default="adam", help="Selected optimizer", type=str)
    parser.add_argument("-name", "--name", default="gan-train", help="Checkpoint/Tensorboard label")
    parser.add_argument("-ds", "--dataset", default="cityscapes", help="choice of training dataset. Currently only supports cityscapes/ADE20k", choices=set(("cityscapes", "ADE20k")), type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # Launch training
    train(config_train, args)

if __name__ == '__main__':
    main()
