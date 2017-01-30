
import tensorflow as tf
import random
import math
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.nan)
random.seed()


handles = []
# get regular sine wave points

x_reg = np.linspace(-0.1, 1.1, 1000)
y_reg = np.sin(x_reg*2*np.pi)

# plot sample points

plt.plot(x_reg, y_reg)


def model_variable(shape, name):
    variable = tf.get_variable(name=name,
                               dtype=tf.float32,
                               shape=shape,
                               initializer=tf.random_uniform_initializer(minval=0, maxval=1)
    )
    tf.add_to_collection('model_variables', variable)
    return variable
    
class Model():
    def __init__(self, sess, data, nEpochs, learning_rate, m):
        self.sess = sess
        self.data = data
        self.nEpochs = nEpochs
        self.learning_rate = learning_rate
        self.m = m
        self.build_model()

    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape=[])
        self.y = tf.placeholder(tf.float32, shape=[])

        w_vec = model_variable([self.m], 'w')
        b = model_variable([], 'b')
        mu_vec = model_variable([self.m], 'mu')
        sigma_vec = model_variable([self.m], 'sigma')
        input_vec = tf.scalar_mul(self.x, tf.ones(self.m))
        phi_vec = tf.exp(tf.scalar_mul(-1, tf.div(tf.pow(tf.subtract(input_vec, mu_vec), 2), tf.pow(sigma_vec, 2))))
        self.yhat = tf.reduce_sum(tf.multiply(w_vec, phi_vec)) + b
        self.loss = (1/2)*(tf.pow(self.yhat - self.y, 2))

    def train_init(self):
        model_variables = tf.get_collection('model_variables')
        self.optim = (
            tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            .minimize(self.loss, var_list=model_variables)
            )
        self.sess.run(tf.initialize_all_variables())

    def train_iter(self, x, y):
        loss, _ = sess.run([self.loss, self.optim],
            feed_dict = {self.x: x, self.y: y})
        print('loss: {}'.format(loss))

    def train(self):
        for _ in range(self.nEpochs):
            for x, y in self.data():
                self.train_iter(x, y)

    def infer(self, x):
        return self.sess.run(self.yhat, feed_dict={self.x : x})

def data():
    num_samp = 50
    sigma = 0.1
    for _ in range(num_samp):
        x = random.random()
        y = math.sin(x*2*math.pi) + random.normalvariate(0, sigma)
        yield x, y

m = 5
sess = tf.Session()
model = Model(sess, data, nEpochs=100, learning_rate=1e-2, m=m)
model.train_init()
model.train()

print(sess.run(tf.get_collection('model_variables')))


y = []
for a in x_reg:
    y.append(model.infer(a))
y = np.array(y)

examples, targets = zip(*list(data()))

plt.plot(x_reg, y, '-', np.array(examples), np.array(targets), 'go')
plt.show()

with tf.variable_scope("", reuse = True):
    w_opt = sess.run(tf.get_variable('w'))
    mu_opt = sess.run(tf.get_variable('mu'))
    sigma_opt = sess.run(tf.get_variable('sigma'))
    b_opt = sess.run(tf.get_variable('b'))


#plot basis functions

handles = []
plt.figure()
for i in range(m):
    local_mu = np.array([mu_opt[i] for j in range(len(x_reg))])
    local_sigma = np.power([sigma_opt[i] for j in range(len(x_reg))], 2)
    local_y = np.exp(np.multiply(np.power(x_reg - local_mu, 2) / np.power(local_sigma, 2), -1));
    local_handle, = plt.plot(x_reg, local_y, label="Line" + str(i))
    handles.append(local_handle)
plt.legend(handles=handles)
plt.show()
