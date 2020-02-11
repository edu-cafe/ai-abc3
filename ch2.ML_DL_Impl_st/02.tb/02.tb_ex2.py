#%%
# tensorboard ex2
import tensorflow as tf
tf.reset_default_graph()

sess = tf.InteractiveSession()

a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b)
y = tf.multiply(a, b)

#%%
writer = tf.summary.FileWriter('./tb2', sess.graph)
#print(sess.run(x))
print(sess.run(y))
writer.close()
sess.close()
