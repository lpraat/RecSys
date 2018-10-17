import tensorflow as tf

from src.const import NUM_TRACKS


def get_similarity_matrix(session, input_set):
    """ Generate tracks similarity matrix """

    # Tensorflow input
    input = tf.placeholder(tf.float32, shape = (NUM_TRACKS, NUM_TRACKS))

    # Normalize each row
    normalized = tf.nn.l2_normalize(input, dim = 1)

    # Cross multiply each row
    # and calc distance
    dist = 1 - tf.matmul(normalized, normalized, adjoint_b = True)

    return session.run()


def dok_matrix_to_sparse_tensor(dok):
    return tf.SparseTensor(dok.keys(), dok.values(), dok.get_shape())