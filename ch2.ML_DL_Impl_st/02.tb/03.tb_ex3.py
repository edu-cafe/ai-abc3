#%%
# tensorboard  ex3
import tensorflow as tf
tf.reset_default_graph()

a = tf.constant(3, name='a')
b = tf.constant(4, name='b')
x = a*b

#%%
with tf.Session() as sess:
	writer = tf.summary.FileWriter('./tb3', sess.graph)  
	#sess.run(x)
	writer.close()
	sess.close()
    