#%%
# tensorboard ex1
import tensorflow as tf
tf.reset_default_graph()

sess = tf.InteractiveSession()

a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
y = tf.multiply(a, b)

#%%
#writer = tf.summary.FileWriter('c:/share/tb/ex1', sess.graph)  #ok
writer = tf.summary.FileWriter('./ex1', sess.graph)  #ok
#sess.run(x)
writer.close()
sess.close()
