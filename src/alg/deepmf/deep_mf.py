import tensorflow as tf
import numpy as np

from src.alg.recsys import RecSys
from src.alg.deepmf.sampler import DeepMFSampler
from collections import namedtuple

class DeepMF(RecSys):

    def __init__(self, name='deep_mf', lr=0.001, epochs=1, all_dataset=True,
                 batch_size=128, num_negative_samples=1, layers=[128,64,32,16], latent_factors=128,
                 save_path=None, load_from_file=False):

        super().__init__()
        self.name = name
        self.num_negative_samples = num_negative_samples

        self.num_users = None
        self.num_items = None

        self.batch_size = batch_size
        self.all_dataset = all_dataset

        # List containing the size of each layer towards the output
        # The last size is the one of the final latent representation
        assert layers, "You did not specify any layers"
        self.layers = layers

        self.latent_factors = latent_factors

        self.lr = lr
        self.epochs = epochs
        self.sampler = None

        # Check if a path is present to either save or load a model and initialize a Saver
        self.save_path = save_path
        if self.save_path:
            to_restore = tf.trainable_variables(name)
            self.saver = tf.train.Saver(var_list=to_restore)

        # Check if the model needs to be loaded from file and if the file path is present
        self.load_from_file = load_from_file
        if self.load_from_file:
            assert self.save_path != None, "You did not specify a path to load the model weights from."


        self.model = None


    def restore_from_file(self):
        # TODO
        pass


    def train(self, dataset):

        # if self.load_from_file:
        #     try:
        #         print("Loading model weights from file...")
        #         pass
        #         return
        #     except ValueError:
        #         print("Error when loading model from file...")
        #         exit()

        print("Starting training...")
        if self.all_dataset:
            urm = self.cache.fetch("interactions")
        else:
            urm = self.cache.fetch("train_set")

        self.num_users = urm.shape[0]
        self.num_items = urm.shape[1]

        self.sampler = DeepMFSampler(urm=urm)
        self.model = self.build_graph()

        # TODO if model from file, load it and skip training

        loss = 0
        processed = 0
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        for _ in range(self.epochs):

            tf_dataset = self.sampler.build_dataset()
            tf_dataset = (tf_dataset
                            .batch(self.batch_size)
                            .shuffle(buffer_size=urm.nnz * (1 + self.num_negative_samples)))

            iterator = tf_dataset.make_one_shot_iterator()
            next_element = iterator.get_next()
            while True:
                try:
                    next_samples = self.sess.run(next_element)
                    new_loss, _= self.sess.run([self.model.loss, self.model.optimizer], feed_dict={
                        self.model.user_input: next_samples[0],
                        self.model.item_input: next_samples[1],
                        self.model.targets: next_samples[2]
                    })

                    if loss == 0:
                        loss = new_loss
                    loss = 0.99 * loss + (0.01) * new_loss
                    processed += 1

                    # TODO remove
                    if (processed % 1000 == 0):
                        break
                    if (processed % 1000 == 0):
                        print("Processed: " + str(processed) + " batches")
                        print("Loss is: " + str(loss))
                except tf.errors.OutOfRangeError:
                    break

    def rate(self, dataset):

        self.train(dataset)

        ratings = np.empty(dataset.shape, dtype=np.float32)
        items = np.arange(dataset.shape[1], dtype=np.float32).reshape(dataset.shape[1], 1)

        for i in range(dataset.shape[0]):
            print(i)

            users = np.full((dataset.shape[1], 1), i, dtype=np.float32)
            ratings[i] = self.sess.run(self.model.prediction, feed_dict={
                self.model.item_input: items,
                self.model.user_input: users
            }).T

        self.sess.close()
        return ratings


    def build_graph(self):
        with tf.variable_scope(self.name):

            user_input = tf.placeholder(tf.float32, shape=[None, 1])
            item_input = tf.placeholder(tf.float32, shape=[None, 1])
            targets = tf.placeholder(tf.float32, shape=[None, 1])

            user_embedding = tf.keras.layers.Embedding(
                input_dim=self.num_users,
                output_dim=self.latent_factors,
                input_length=1
            )

            # Item
            item_embedding = tf.keras.layers.Embedding(
                input_dim=self.num_items,
                output_dim=self.latent_factors,
                input_length=1
            )

            concat_embedding = tf.keras.layers.concatenate([user_embedding(user_input), item_embedding(item_input)])
            flattened = tf.keras.layers.Flatten()(concat_embedding)

            fc = tf.layers.dense(inputs=flattened, units=self.layers[0], activation=tf.nn.relu)
            for layer_size in self.layers[1:]:
                fc = tf.layers.dense(inputs=fc, units=layer_size, activation=tf.nn.relu)

            logits = tf.layers.dense(inputs=fc, units=1, activation=None)
            prediction = tf.layers.dense(inputs=logits, units=1, activation=tf.nn.sigmoid)

            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

            DeepMfGraph = namedtuple(self.name, ['user_input', 'item_input', 'targets', 'optimizer', 'loss', 'prediction'])
            return DeepMfGraph(user_input, item_input, targets, optimizer, loss, prediction)





