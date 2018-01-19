import pandas as pd
import tensorflow as tf

def multilayer_perceptron(data): #MLP with two hidden layers
    layer_1 = tf.add(tf.matmul(data, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

###### Loading the data set and some pre-processing
train = pd.read_csv("ghouls_train.csv")     #Loading Data
train["type"] = train["type"].map({"Ghoul": 0, "Goblin": 1, "Ghost": 2})
train["color"] = train["color"].map({"white": 0, "black": 1, "clear": 2, "blue": 3, "green": 4, "blood": 5})
train_y = train['type']     #Training data Labels
train.drop('id', axis=1, inplace='true')    #Removing id column
train.drop('type', axis=1, inplace='true')  #Removing labels from training data
train_y = pd.get_dummies(train_y)           #Encoding labels with OneHot
test = train[-50:]                          #Take last 50 rows as your test data (better to use train_test_split function for this purpose)
train.drop(train.index[-50:], axis=0, inplace='true')
test_y = train_y[-50:]
train_y.drop(train_y.index[-50:], axis=0, inplace='true')

###### Seting user defined variables
learning_rate = 0.05
training_epochs = 30
batch_size = 20
display_step = 1
n_hidden_1 = 5 # 1st layer number of neurons
n_hidden_2 = 5 # 2nd layer number of neurons
n_input = 5 # MNIST data input (img shape: 28*28)
n_classes = 3 # MNIST total classes (0-9 digits)

##### Initializing biases and weights
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

out_layer = multilayer_perceptron(X)
###### Define loss function and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)  # There are many other optimizer options e.g. "AdadeltaOptimizer", "AdagradDAOptimizer", "AdagradOptimizer"
train_op = optimizer.minimize(loss_op)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(train)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = train[i*batch_size:(i+1)*batch_size], train_y[i*batch_size:(i + 1)*batch_size]
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
            avg_cost += c / total_batch
    print("Optimization Finished!")

    pred = tf.nn.softmax(out_layer)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: test, Y: test_y}))