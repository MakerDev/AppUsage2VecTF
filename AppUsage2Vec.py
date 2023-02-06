import tensorflow as tf
keras = tf.keras
import keras.layers as layers

class AppUsage2Vec(keras.models.Model):
    def __init__(self, n_users, n_apps, dim, seq_length, n_layers, alpha, k, device):
        super(AppUsage2Vec, self).__init__()

        self.user_emb   = layers.Embedding(n_users, dim)
        self.app_emb    = layers.Embedding(n_apps, dim)
        self.seq_length = seq_length
        self.alpha      = alpha
        self.k          = k
        self.dim        = dim
        self.device     = device
        self.n_layers   = n_layers
        
        self.attn       = tf.keras.layers.Dense(seq_length, input_shape=(seq_length * (dim+1),))
        
        self.user_dnn   = [tf.keras.layers.Dense(dim, input_shape=(dim,)) for i in range(n_layers)]
        self.app_dnn    = [tf.keras.layers.Dense(dim, input_shape=(dim,)) for i in range(n_layers)]
        
        self.classifier = tf.keras.layers.Dense(n_apps, input_shape=(dim+31,))

    @tf.function
    def call(self, users, time_vecs, app_seqs, time_seqs, targets, mode='test'):
        app_seqs_emb = self.app_emb(app_seqs)
        time_seqs = tf.expand_dims(time_seqs, axis=2)
        app_seqs_time = tf.concat([app_seqs_emb, time_seqs], axis=2)
        
        app_seqs_flat = tf.reshape(app_seqs_time, (tf.shape(app_seqs_time)[0], -1))
        
        H_v = tf.tanh(self.attn(app_seqs_flat))
        weights = tf.nn.l2_normalize(H_v, axis=1)
        
        seq_vector = tf.squeeze(tf.matmul(tf.transpose(app_seqs_emb, [0, 2, 1]), tf.expand_dims(weights, axis=2)), axis=2)
        
        user_vector = tf.squeeze(self.user_emb(users), axis=1)
        for i in range(self.n_layers):
            user_vector = self.user_dnn[i](user_vector)
            user_vector = tf.tanh(user_vector)
            seq_vector = self.app_dnn[i](seq_vector)
            seq_vector = tf.tanh(seq_vector)
        
        combination = tf.multiply(user_vector, seq_vector)
        
        combination = tf.concat([combination, time_vecs], axis=1)
        
        scores = self.classifier(combination)
        
        if mode == 'test':
            return scores

        # Comment the following lines when converting into tflite model.
        # nn.spare_softmax_cross_entropy_with_logits is not supported op thus making an error while it is not needed
        # for 'inference'. Loss is only required for training.
        preds = tf.nn.top_k(scores, k=self.k).indices
        indicator = tf.reduce_sum(tf.cast(tf.equal(preds, tf.cast(targets, tf.int32)), tf.float32), axis=1)
        alpha = tf.constant(self.alpha, dtype=tf.float32)
        coefficient = tf.pow(alpha * tf.ones_like(indicator), indicator)
        
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=tf.reshape(targets, [-1])))
        loss = tf.reduce_mean(tf.multiply(coefficient, loss))
        return loss

if __name__ == '__main__':
    model = AppUsage2Vec(1000, 100, 64, 4, 2, 0.1, 5, '/GPU:0')
    
