import os
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()


MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = MODEL_DIR + '/models/network_model.ckpt'
SUMMARY_DIR = MODEL_DIR + '/logs'


class NeuralNet(object):
    def __init__(self, item_size, node_size, embedding_size):
        self.item_size = item_size
        self.embedding_size = embedding_size
        self.item_embeddings = tf.get_variable("item_embeddings",
                                               [self.item_size, self.embedding_size],
                                               use_resource=True)
        self.node_embeddings = tf.get_variable("node_embeddings",
                                               [node_size, self.embedding_size],
                                               use_resource=True)
        self.saver = None

    def _PRelu(self, x):
        m, n = tf.shape(x)
        value_init = 0.25 * tf.ones((1, n))
        a = tf.Variable(initial_value=value_init, use_resource=True)
        y = tf.maximum(x, 0) + a * tf.minimum(x, 0)
        return y

    def _activation_unit(self, item, node):
        item, node = tf.reshape(item, [1, -1]), tf.reshape(node, [1, -1])
        hybrid = item * node
        feature = tf.concat([item, hybrid, node], axis=1)
        layer1 = tf.layers.dense(feature, 36)
        layer1_prelu = self._PRelu(layer1)
        weight = tf.layers.dense(layer1_prelu, 1)
        return weight

    def _attention_feature(self, item, node, is_leafs, features):
        item_clip = item[item != -2]
        item_embedding = tf.nn.embedding_lookup(self.item_embeddings, item_clip)
        if is_leafs[0] == 0:
            node_embedding = tf.nn.embedding_lookup(self.node_embeddings, node)
        else:
            node_embedding = tf.nn.embedding_lookup(self.item_embeddings, node)
        item_num, _ = tf.shape(item_embedding)
        item_feature = None
        for i in range(item_num):
            item_weight = self._activation_unit(item_embedding[i], node_embedding[0])[0][0]
            if item_feature is None:
                item_feature = item_weight * item_embedding[i]
            else:
                item_feature = tf.add(item_feature, item_weight * item_embedding[i])
        item_feature = tf.concat([tf.reshape(item_feature, [1, -1]), node_embedding], axis=1)
        if features is None:
            features = item_feature
        else:
            features = tf.concat([features, item_feature], axis=0)
        return features

    def _attention_block(self, items, nodes, is_leafs):
        batch, _ = tf.shape(items)
        features = None
        for i in range(batch):
            features = self._attention_feature(items[i], nodes[i], is_leafs[i], features)
        return features

    def _network_structure(self, items, nodes, is_leafs, is_training):
        batch_features = self._attention_block(items, nodes, is_leafs)
        layer1 = tf.layers.dense(batch_features, 128)
        layer1_prelu = self._PRelu(layer1)
        layer1_bn = tf.layers.batch_normalization(layer1_prelu, training=is_training)
        layer2 = tf.layers.dense(layer1_bn, 64)
        layer2_prelu = self._PRelu(layer2)
        layer2_bn = tf.layers.batch_normalization(layer2_prelu, training=is_training)
        layer3 = tf.layers.dense(layer2_bn, 24)
        layer3_prelu = self._PRelu(layer3)
        layer3_bn = tf.layers.batch_normalization(layer3_prelu, training=is_training)
        logits = tf.layers.dense(layer3_bn, 2)
        return logits

    def _check_accuracy(self, validate_data, is_training):
        num_correct, num_samples = 0, 0
        for items_val, nodes_val, is_leafs_val, labels_val in validate_data:
            scores = self._network_structure(items_val, nodes_val, is_leafs_val, is_training)
            scores = scores.numpy()
            label_predict = scores.argmax(axis=1)
            label_true = labels_val.argmax(axis=1)
            label_predict = label_predict[label_predict == label_true]
            label_predict = label_predict[label_predict == 0]
            label_true = label_true[label_true == 0]
            num_samples += label_true.shape[0]
            num_correct += label_predict.shape[0]
        accuracy = float(num_correct) / num_samples
        print("total positive samples: {}, "
              "correct samples: {}, accuracy: {}".format(num_samples, num_correct, accuracy))

    def train(self, use_gpu=False, train_data=None, validate_data=None,
              lr=0.001, b1=0.9, b2=0.999, eps=1e-08, num_epoch=10, check_epoch=200, save_epoch=1000):
        device = '/device:GPU:0' if use_gpu else '/cpu:0'
        with tf.device(device):
            container = tf.contrib.eager.EagerVariableStore()
            check_point = tf.contrib.eager.Checkpointable()
            iter_epoch = 0
            for epoch in range(num_epoch):
                print("Start epoch %d" % epoch)
                for items_tr, nodes_tr, is_leafs_tr, labels_tr in train_data:
                    with tf.GradientTape() as tape:
                        with container.as_default():
                            scores = self._network_structure(items_tr, nodes_tr, is_leafs_tr, 1)
                        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_tr, logits=scores)
                        loss = tf.reduce_sum(loss)
                        print("Epoch {}, Iteration {}, loss {}".format(epoch, iter_epoch, loss))
                    gradients = tape.gradient(loss, container.trainable_variables())
                    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=b1, beta2=b2, epsilon=eps)
                    optimizer.apply_gradients(zip(gradients, container.trainable_variables()))
                    if iter_epoch % check_epoch == 0:
                        self._check_accuracy(validate_data, 0)
                    if iter_epoch % save_epoch == 0:
                        for k, v in container._store._vars.items():
                            setattr(check_point, k, v)
                        self.saver = tf.train.Checkpoint(checkpointable=check_point)
                        self.saver.save(MODEL_NAME)
                    iter_epoch += 1
                    break
                break
        print("It's completed to train the network.")

    def get_embeddings(self, item_list, use_gpu=True):
        """
        TODO: validate and optimize
        """
        model_path = tf.train.latest_checkpoint(MODEL_DIR + '/models/')
        self.saver.restore(model_path)
        device = '/device:GPU:0' if use_gpu else '/cpu:0'
        with tf.device(device):
            item_embeddings = tf.nn.embedding_lookup(self.item_embeddings, np.array(item_list))
            res = item_embeddings.numpy()
        return res.tolist()

    def predict(self, data, use_gpu=True):
        """
        TODO: validate and optimize
        """
        model_path = tf.train.latest_checkpoint(MODEL_DIR+'/models/')
        self.saver.restore(model_path)
        device = '/device:GPU:0' if use_gpu else '/cpu:0'
        with tf.device(device):
            items, nodes, is_leafs = data
            scores = self._network_structure(items, nodes, is_leafs, 0)
            scores = scores.numpy()
        return scores[:, 0]

