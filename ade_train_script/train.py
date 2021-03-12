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
			test_paths = Data.load_dataframe(directories.test)

	    # Build graph
		ngpus = 4
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
		
		img_list = [gan.img1,gan.img2,gan.img3,gan.img4]

		with tf.variable_scope(tf.get_variable_scope()):
			for i in range(ngpus):
				with tf.device('/gpu:%d' % i):
					with tf.name_scope('tower%d' % i) as scope:
						_x = img_list[i]
						#print(_x.shape)
						#time.sleep()
						D_loss,G_loss, reconstruction,distortion_penalty,match_loss,merge_op,*Res  = Model.create_model(_x,gan.training_phase,config,-0.1)
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
			
			for epoch in range(config.num_epochs):

				sess.run(gan.train_iterator.initializer, feed_dict=feed_dict_train_init)
				img_l = []
				for ii in range(ngpus):
					oriimage = sess.run(gan.example,feed_dict={gan.handle: train_handle})
					img_l.append(oriimage)

				# Run diagnostics
				G_loss_best, D_loss_best = Utils.run_diagnostics(gan,G_loss_mean,D_loss_mean,Distor_mean,Match_mean, config, directories, sess, saver, train_handle,start_time, epoch, args.name, G_loss_best, D_loss_best,img_l)
				while True:
					try:
						# Update generator
						#for _ in range(8):
						img_l = []
						for ii in range(ngpus):
							oriimage = sess.run(gan.example,feed_dict={gan.handle: train_handle})
							img_l.append(oriimage)

						feed_dict = {gan.training_phase:True,gan.handle: train_handle,gan.img1:img_l[0],gan.img2:img_l[1],gan.img3:img_l[2],gan.img4:img_l[3]}
						sess.run(G_train_op, feed_dict=feed_dict)
						
						img_l = []
						for ii in range(ngpus):
							oriimage = sess.run(gan.example,feed_dict={gan.handle: train_handle})
							img_l.append(oriimage)
						feed_dict = {gan.training_phase:True,gan.handle: train_handle,gan.img1:img_l[0],gan.img2:img_l[1],gan.img3:img_l[2],gan.img4:img_l[3]}

						step,_,m = sess.run([gan.D_global_step,D_train_op,merge_op], feed_dict=feed_dict)
				
						if step % (config.diagnostic_steps/ngpus) == 0:
							G_loss_best, D_loss_best = Utils.run_diagnostics(gan,G_loss_mean,D_loss_mean,Distor_mean,Match_mean,config, directories, sess, saver, train_handle,start_time, epoch, args.name, G_loss_best, D_loss_best,img_l)			
							Utils.single_plot(epoch, reconstruction_total,step, sess, gan, train_handle, args.name, config,img_l)
							train_writer.add_summary(m,step)
					


					except tf.errors.OutOfRangeError:
						print('End of epoch!')
						break

					except KeyboardInterrupt:
						save_path = saver.save(sess, os.path.join(directories.checkpoints,
							'{}_last.ckpt'.format(args.name)), global_step=epoch)
						print('Interrupted, model saved to: ', save_path)
						sys.exit()

			save_path = saver.save(sess, os.path.join(directories.checkpoints,
				       '{}_end.ckpt'.format(args.name)),
				       global_step=epoch)

		print("Training Complete. Model saved to file: {} Time elapsed: {:.3f} s".format(save_path, time.time()-start_time))

def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("-rl", "--restore_last", help="restore last saved model", action="store_true")
    parser.add_argument("-r", "--restore_path", help="path to model to be restored", type=str)
    parser.add_argument("-opt", "--optimizer", default="adam", help="Selected optimizer", type=str)
    parser.add_argument("-name", "--name", default="gan-train", help="Checkpoint/Tensorboard label")
    parser.add_argument("-ds", "--dataset", default="cityscapes", help="choice of training dataset. Currently only supports cityscapes/ADE20k", choices=set(("cityscapes", "ADE20k")), type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # Launch training
    train(config_train, args)

if __name__ == '__main__':
    main()
