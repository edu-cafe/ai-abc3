#%%
# tensorboard  ex4
import tensorflow as tf
tf.reset_default_graph() 

a = tf.constant(3.0, name='a')
b = tf.constant(4.0, name='b')
c = a*b

c_summary = tf.summary.scalar('point', c)

#%%
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./ex41', sess.graph)  
    rst = sess.run(c_summary)	
    writer.add_summary(rst)
    writer.close()
    sess.close()

    