#!/usr/bin/var

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



img = tf.read_file('/home/jason/project/generative-compression_hancy16/data/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png')
img = tf.image.decode_png(img,channels=1)
img = tf.expand_dims(img,axis=0)
img_s = tf.concat([img for _ in range(2)]+[tf.zeros_like(img) for _ in range(2)],axis=-1)
def dense(img):
	assert len(img.get_shape())==4,'Data Dimension should be 4'
	shape_HWC = tf.shape(img)
	N = int(img.shape[0])
	H = shape_HWC[1]
	W = shape_HWC[2]
	C = int(img.shape[3])
	assert C==4, 'Channel should be 4'

	img_1,img_2 = tf.split(img,2,axis=-1)
	

	img_tmp = tf.concat([dense1(hh) for hh in tf.split(img,2,axis=-1)],axis=-1)

	img1 = tf.transpose(img_tmp,[0,1,3,2])
	img1 = tf.reshape(img1,[N,2*H,2*W,C/4])
	return img1
	



def dense1(img):
	shape_HWC = tf.shape(img)
	assert len(img.get_shape())==4,'Data Dimension should be 4'
	N = int(img.shape[0])
	H = shape_HWC[1]
	W = shape_HWC[2]
	C = int(img.shape[3])
	assert C==2, 'Channel should be 2'
	img1 = tf.reshape(img,[N,H,2*W,C/2])
	return img1

with tf.Session() as sess:
	print(img_s.shape)	
	a1 =dense(img_s)
	a1 = tf.squeeze(a1)
	img =sess.run(a1)
	print (img.shape)
	plt.imshow(img,cmap=plt.cm.gray)
	plt.axis('off')
	plt.savefig('test.png',dpi=720)
