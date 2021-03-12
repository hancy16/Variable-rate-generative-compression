#!/usr/bin/python3
    
import tensorflow as tf
import numpy as np
import glob, time, os
import time
from network import Network
from data import Data
from config import directories
from utils import Utils
import random
from tensorflowvgg import vgg16
class Model():
	@staticmethod
	def average_gradients(tower_grads):
		average_grads = []
		for grad_and_vars in zip(*tower_grads):
			grads = []
			for g,_ in grad_and_vars:
				expend_g = tf.expand_dims(g,axis=0)
				grads.append(expend_g)
			grad  = tf.concat(grads,0)			
			grad  = tf.reduce_mean(grad,0)
			v = grad_and_vars[0][1]
			grad_and_var = (grad,v)
			average_grads.append(grad_and_var)
		return average_grads

	def __init__(self, config, paths, dataset, ngpus, name='gan_compression', evaluate=False):
# Build the computational graph

		
		print('Using %d GPUs...' % ngpus )
		print('Building computational graph ...')
		self.G_global_step = tf.Variable(0, trainable=False)
		self.D_global_step = tf.Variable(0, trainable=False)
		self.handle = tf.placeholder(tf.string, shape=[])
		self.training_phase = tf.placeholder(tf.bool)
		self.thres = tf.placeholder(tf.float32, shape=[])
		self._x = tf.placeholder(tf.float32, shape=[1,None,None,3])
		# >>> Data handling
		self.path_placeholder = tf.placeholder(paths.dtype, paths.shape)
		self.test_path_placeholder = tf.placeholder(paths.dtype)            

		self.semantic_map_path_placeholder = tf.placeholder(paths.dtype, paths.shape)
		self.test_semantic_map_path_placeholder = tf.placeholder(paths.dtype)  

		self.img1 = tf.placeholder(tf.float32, shape=[1,None,None,3])
		self.img2 = tf.placeholder(tf.float32, shape=[1,None,None,3])
		self.img3 = tf.placeholder(tf.float32, shape=[1,None,None,3])
		self.img4 = tf.placeholder(tf.float32, shape=[1,None,None,3])

		train_dataset = Data.load_dataset(self.path_placeholder,
					          config.batch_size,
					          augment=False,
					          training_dataset=dataset,
					          use_conditional_GAN=config.use_conditional_GAN,
					          semantic_map_paths=self.semantic_map_path_placeholder)

		test_dataset = Data.load_dataset(self.test_path_placeholder,
					         config.batch_size,
					         augment=False,
					         training_dataset=dataset,
					         use_conditional_GAN=config.use_conditional_GAN,
					         semantic_map_paths=self.test_semantic_map_path_placeholder,
					         test=True)

		self.iterator = tf.data.Iterator.from_string_handle(self.handle,
					                                    train_dataset.output_types,
					                                    train_dataset.output_shapes)

		self.train_iterator = train_dataset.make_initializable_iterator()
		self.test_iterator = test_dataset.make_initializable_iterator()

		if config.use_conditional_GAN:
		    self.example, self.semantic_map = self.iterator.get_next()
		else:
		    self.example = self.iterator.get_next()

		
		

        # Global generator: Encode -> quantize -> reconstruct
        # =======================================================================================================>>>
	@staticmethod
	def create_model(sample,training_phase,config,threshold, evaluate=False):
		with tf.variable_scope('generator'):
			#_x = tf.add(sample,1*tf.random_normal(shape=[1,256,512,3]))
			_x_ori = sample
			feature_map = Network.encoder(_x_ori, config, training_phase, config.channel_bottleneck)
			w_hat = Network.quantizer(feature_map, config)
			if config.sample_noise is True:
				print('Sampling noise...')
				noise_prior = tf.contrib.distributions.MultivariateNormalDiag(loc=tf.zeros([config.noise_dim]), scale_diag=tf.ones([config.noise_dim]))
				v = noise_prior.sample(tf.shape(_x_ori)[0])
				Gv = Network.dcgan_generator(v, config, training_phase, C=config.channel_bottleneck, upsample_dim=config.upsample_dim)
				z = tf.concat([w_hat, Gv], axis=-1)
			else:
				z = w_hat

			reconstruction_ori = Network.decoder(z, config, training_phase, C=config.channel_bottleneck)
			result_reconstruction = reconstruction_ori
			print('Real image shape:', _x_ori.get_shape().as_list())
			print('Reconstruction shape:', reconstruction_ori.get_shape().as_list())

			if evaluate:
				return
			
			hh = tf.reduce_mean(tf.reduce_mean(w_hat,axis = 1),axis=1)

			f_mean = tf.stack([tf.reduce_mean(feature_map,axis = -1) for _ in range(config.channel_bottleneck)],axis=-1)
			f_std  = tf.expand_dims(tf.reduce_mean(tf.square(feature_map-f_mean),axis=-1),axis=-1)
			#ups = tf.pad(f_std, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
			w_mask = tf.cast(tf.greater(f_std,threshold),dtype=tf.float32)
			
			w_loss = 1 * tf.reduce_mean(f_std)
			w_maskn = tf.concat([w_mask for _ in range(config.channel_bottleneck)],axis=-1)
			#
			#lambd_rate = 1/tf.reduce_mean(f_std)
			#lambd_noise_prior    = tf.distributions.Exponential(lambd_rate)
			#lambd_noise = lambd_noise_prior.sample([32*64])
			#lambd_noise_reg = tf.minimum(lambd_noise,threshold)
			#noise_prior = tf.contrib.distributions.MultivariateNormalDiag(loc=tf.zeros([32*64]), scale_diag=lambd_noise_reg)
			#noise = tf.stack([tf.expand_dims(tf.reshape(noise_prior.sample([1]),[32,64]),axis=-1) for _ in range(16)],axis=-1)
			#print(tf.shape(noise))
			#lambd_noise = tf.reshape(lambd_noise,[1,32,64])
			#lambd_noise = tf.stack([lambd_noise for _ in range(16)],axis=-1)

			#noise = lambd_noise*tf.random_uniform(tf.shape(feature_map),minval=-0.5,maxval=0.5)

			#noise = tf.layers.average_pooling2d(noise,pool_size=3,strides=1,padding='same')
			f_reduce = tf.multiply(feature_map,w_maskn) + tf.multiply(f_mean,1-w_maskn)
			#print(f_reduce)
			w_reduce = Network.quantizer(f_reduce, config)
			
			w_mask1 = tf.concat([tf.ones_like(f_std) for _ in range(config.channel_bottleneck-8)]+[tf.zeros_like(f_std) for _ in range(8)],axis=-1)
			f1,f2 = tf.split(feature_map,num_or_size_splits=2,axis=-1)
			f_reduce1 = tf.concat([f1 for _ in range(2)], axis=-1)			
			#w_1 = tf.stack([tf.reduce_sum(tf.multiply(tf.concat([tf.ones_like(w_std)]+[tf.zeros_like(w_std)  for _ in range(config.channel_bottleneck-1)],axis=-1),w_hat),axis=-1) for _ in range(config.channel_bottleneck)],axis=-1)
			#f_reduce1= tf.multiply(feature_map,w_mask1) + tf.multiply(f_mean,1-w_mask1)
			w_reduce1 = Network.quantizer(f_reduce1, config)

			reconstruction_reduce = Network.decoder(w_reduce, config, False, C=config.channel_bottleneck)
			#reconstruction_reduce1 = Network.decoder(w_reduce1, config, False, C=config.channel_bottleneck)
			
#==============================================Concat==================================================================
			#original_image = _x_ori
			Flag = tf.less(tf.reduce_mean(tf.random_uniform((1,))-tf.constant([0.5])),0)
			#reconstruction = tf.cond(Flag,lambda: tf.concat([reconstruction_ori, original_image], axis=-1),lambda:tf.concat([reconstruction_ori,tf.zeros_like(reconstruction_ori)],axis=-1))
			#_x = tf.cond(Flag,lambda:tf.concat([_x_ori,original_image], axis=-1),lambda:tf.concat([_x_ori,tf.zeros_like(_x_ori)],axis=-1))
			_x = _x_ori
			reconstruction = reconstruction_ori
		if config.multiscale:
			D_x, D_x2, D_x4, *Dk_x = Network.multiscale_discriminator(_x, config, training_phase, use_sigmoid=config.use_vanilla_GAN, mode='real')
			D_Gz, D_Gz2, D_Gz4, *Dk_Gz = Network.multiscale_discriminator(reconstruction, config, training_phase,use_sigmoid=config.use_vanilla_GAN, mode='reconstructed', reuse=True)
		else:
			D_x = Network.discriminator(_x, config, training_phase, use_sigmoid=config.use_vanilla_GAN)
			D_Gz = Network.discriminator(reconstruction, config, training_phase, use_sigmoid=config.use_vanilla_GAN, reuse=True)
				



		# Loss terms 
			# =======================================================================================================>>>
		if config.use_vanilla_GAN is True:
		# Minimize JS divergence
			D_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_x,
			labels=tf.ones_like(D_x)))
			D_loss_gen = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_Gz,
			labels=tf.zeros_like(D_Gz)))
			D_loss = D_loss_real + D_loss_gen
			# G_loss = max log D(G(z))
			G_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_Gz,
			labels=tf.ones_like(D_Gz)))
		else:
		# Minimize $\chi^2$ divergence
			D_loss = tf.reduce_mean(tf.square(D_x - 1.)) + tf.reduce_mean(tf.square(D_Gz))
			G_loss = tf.reduce_mean(tf.square(D_Gz - 1.))

		if config.multiscale:
			D_loss += tf.reduce_mean(tf.square(D_x2 - 1.)) + tf.reduce_mean(tf.square(D_x4 - 1.))
			D_loss += tf.reduce_mean(tf.square(D_Gz2)) + tf.reduce_mean(tf.square(D_Gz4))
			G_loss += tf.reduce_mean(tf.square(D_Gz2 - 1.)) + tf.reduce_mean(tf.square(D_Gz4 - 1.))

		distortion_penalty = config.lambda_X * tf.losses.mean_squared_error(_x, reconstruction)
		G_loss += distortion_penalty
		#match_loss = tf.Variable(0, trainable=False)
		if config.use_feature_matching_loss:  # feature extractor for generator
			D_x_layers, D_Gz_layers = [j for i in Dk_x for j in i], [j for i in Dk_Gz for j in i]
			feature_matching_loss = tf.reduce_sum([tf.reduce_mean(tf.abs(Dkx-Dkz)) for Dkx, Dkz in zip(D_x_layers, D_Gz_layers)])
			match_loss =  config.feature_matching_weight * feature_matching_loss / 3.0
			G_loss += match_loss

		if config.use_VGG_loss:
			VGGweights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0] 
			vgg = vgg16.Vgg16()
			_,*Vgg16_x = vgg.build(_x)
			_,*Vgg16_Gz = vgg.build(reconstruction)
			VGG_loss =  tf.reduce_sum([weight*tf.reduce_mean(tf.abs(vggx-vggz)) for vggx, vggz,weight in zip(Vgg16_x, Vgg16_Gz,VGGweights)])
			G_loss += config.VGG_loss_weight *VGG_loss
		G_loss += w_loss
		
		PSNR = tf.image.psnr(reconstruction,sample,max_val=1.0)
		SSIM = tf.image.ssim_multiscale(reconstruction,sample,max_val=1.0)
		PSNR_r = tf.image.psnr(reconstruction_reduce,sample,max_val=1.0)
		SSIM_r = tf.image.ssim_multiscale(reconstruction_reduce,sample,max_val=1.0)
		#reconstruction_1 = Network.decoder(w_hat, config, False, C=config.channel_bottleneck)

		#tf.summary.image('Feature std ',tf.concat([f_std for _ in range(4)],axis=-1))
		#tf.summary.histogram('Quantization std', f_std)
		#tf.summary.histogram('Quantization mean', f_mean)
		#tf.summary.histogram('hh', hh)
		#tf.summary.image('Greater Than 0.6',w_mask)


		#for i,cnt in zip(tf.split(feature_map ,config.channel_bottleneck,axis=-1),range(config.channel_bottleneck)):	
		#	tf.summary.image('feature Map%d '% cnt,i)
		#	tf.summary.histogram('feature %d'% cnt, i)

		#for i,cnt in zip(tf.split(w_reduce1,config.channel_bottleneck,axis=-1),range(config.channel_bottleneck)):	
		#	tf.summary.image('Reduce Map%d '% cnt,i)
		#	tf.summary.histogram('Reduce Map%d'% cnt, i)


		tf.summary.image('Original Image',sample)
		tf.summary.image('Reconstruction',reconstruction)
		#tf.summary.image('Reconstruction_reduce',reconstruction_reduce)
		#tf.summary.image('Reconstruction_reduce1',reconstruction_reduce1)
		tf.summary.scalar('G_loss', G_loss)
		tf.summary.scalar('D_loss', D_loss)
		tf.summary.scalar('w_loss', w_loss)
		#tf.summary.scalar('PSNR_r', tf.squeeze(PSNR_r))
		#tf.summary.scalar('SSIM_r', tf.squeeze(SSIM_r))
		merge_op = tf.summary.merge_all()


		return D_loss,G_loss,reconstruction_ori ,distortion_penalty,match_loss,merge_op,w_hat,w_mask
		
# Optimization
			# =======================================================================================================>>>
	'''
						G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
						D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator/discriminator')
						#print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'generator'))

						#print(tf.trainable_variables('generator'))
					
						theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
						theta_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator/discriminator')
					
						print(D_loss)
						# Execute the update_ops before performing the train_step
						with tf.control_dependencies(G_update_ops):
						    G_grads = G_opt.compute_gradients(G_loss,   var_list=theta_G)
						with tf.control_dependencies(D_update_ops):
						    D_grads = D_opt.compute_gradients(D_loss,   var_list=theta_D)
						print(G_grads)					
						tower_G_grads.append(G_grads)
						tower_D_grads.append(D_grads)

		
		# print('Generator parameters:', theta_G)
		# print('Discriminator parameters:', theta_D)

		#print(tower_G_grads)
		self.G_grads = Model.average_gradients(tower_G_grads)
		self.D_grads = Model.average_gradients(tower_D_grads)
		self.G_opt_op = G_opt.apply_gradients(self.G_grads)
		self.D_opt_op = D_opt.apply_gradients(self.D_grads)
		
		self.G_global_step += 1
		self.D_global_step += 1
		G_ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=self.G_global_step)
		G_maintain_averages_op = G_ema.apply(theta_G)
		D_ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=self.D_global_step)
		D_maintain_averages_op = D_ema.apply(theta_D)

		with tf.control_dependencies(G_update_ops+[self.G_opt_op]):
		    self.G_train_op = tf.group(G_maintain_averages_op)
		with tf.control_dependencies(D_update_ops+[self.D_opt_op]):
		    self.D_train_op = tf.group(D_maintain_averages_op)

		# >>> Monitoring
		# tf.summary.scalar('learning_rate', learning_rate)
		tf.summary.scalar('generator_loss', self.G_loss)
		tf.summary.scalar('discriminator_loss', self.D_loss)
		tf.summary.scalar('distortion_penalty', distortion_penalty)
		if config.use_feature_matching_loss:
		    tf.summary.scalar('feature_matching_loss', feature_matching_loss)
		tf.summary.scalar('G_global_step', self.G_global_step)
		tf.summary.scalar('D_global_step', self.D_global_step)
		tf.summary.image('real_images', self.example[:,:,:,:3], max_outputs=4)
		tf.summary.image('compressed_images', self.reconstruction[:,:,:,:3], max_outputs=4)
		if config.use_conditional_GAN:
		    tf.summary.image('semantic_map', self.semantic_map, max_outputs=4)
		self.merge_op = tf.summary.merge_all()

		self.train_writer = tf.summary.FileWriter(
		    os.path.join(directories.tensorboard, '{}_train_{}'.format(name, time.strftime('%d-%m_%I:%M'))), graph=tf.get_default_graph())
		self.test_writer = tf.summary.FileWriter(
		    os.path.join(directories.tensorboard, '{}_test_{}'.format(name, time.strftime('%d-%m_%I:%M'))))
'''



