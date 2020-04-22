import tensorflow as tf


def _get_label_mask(labels):
    
    a_labels, p_labels = tf.split(labels, 2, axis=0)
    labels_equal = tf.equal(tf.expand_dims(a_labels, 0), tf.expand_dims(p_labels, 1))

    return labels_equal



def triplet_loss(labels, embeddings, margin = 0.1):
    
    # normalize input
    norm_batch = tf.math.l2_normalize(embeddings, axis = 1)
    
    # create adjaceny matrix of cosine similarity
    anchor, pos = tf.split(norm_batch, 2, axis = 0)
    dot_product = 1 - tf.matmul(anchor, tf.transpose(pos))

    # get positive distances
    pos_dists = tf.linalg.diag_part(dot_product)

    # get negatives mask
    labels = tf.squeeze(labels)
    label_mask = 2 * tf.cast(_get_label_mask(labels), tf.float32)

    masked_dot_product = tf.add(dot_product, label_mask)
    # get negative distances
    neg_dists = tf.reduce_min(masked_dot_product, axis=1)

    t_loss = tf.maximum(pos_dists - neg_dists + margin, 0.0)
    
    return t_loss








