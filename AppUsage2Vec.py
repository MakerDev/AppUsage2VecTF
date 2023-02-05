import tensorflow as tf
keras = tf.keras
import keras.layers as layers
from keras.losses import SparseCategoricalCrossentropy
from keras.utils import losses_utils

class AppUsage2Vec(keras.models.Model):
    def __init__(self, n_users, n_apps, dim, seq_length, n_layers, alpha, k, device):
        super(AppUsage2Vec, self).__init__()

        self.user_emb   = layers.Embedding(n_users, dim)
        self.app_emb    = layers.Embedding(n_apps, dim)
        self.seq_length = seq_length
        self.alpha      = alpha
        self.k          = k
        self.device     = device
        self.n_layers   = n_layers
        
        self.attn       = tf.keras.layers.Dense(seq_length, input_shape=(seq_length * (dim+1),))
        
        self.user_dnn   = [tf.keras.layers.Dense(dim, input_shape=(dim,)) for i in range(n_layers)]
        self.app_dnn    = [tf.keras.layers.Dense(dim, input_shape=(dim,)) for i in range(n_layers)]
        
        self.classifier = tf.keras.layers.Dense(n_apps, input_shape=(dim+31,))

    def call(self, users, time_vecs, app_seqs, time_seqs, targets, mode):
        app_seqs_emb  = self.app_emb(app_seqs)                        # [batch_size, seq_length, dim]
        time_seqs     = tf.expand_dims(time_seqs, axis=2)             # [batch_size, seq_length, 1]
        app_seqs_time = tf.concat([app_seqs_emb, time_seqs], axis=2)  # [batch_size, seq_length, dim+1]
        
        app_seqs_flat = tf.reshape(app_seqs_time, [app_seqs_time.shape[0], -1])  # [batch_size, seq_length * (dim+1)]
        
        # get sequence vector / Eq.(6)
        H_v     = tf.math.tanh(self.attn(app_seqs_flat))     # [batch_size, seq_length]
        weights = tf.math.softmax(H_v, axis=1)  # [batch_size, seq_length]
        
        # [batch_size, dim, seq_length] * [batch_size, seq_length, 1] = [batch_size, dim]
        seq_vector = tf.matmul(tf.transpose(app_seqs_emb, perm=(0, 2, 1)), tf.expand_dims(weights, axis=2))
        # -> Same with 'seq_vector = tf.matmul(app_seqs_emb, weights, transpose_a=True)' ??
        seq_vector = tf.squeeze(seq_vector, axis=2)

        # dual dnn / Eq.(7)(8)
        user_vector = tf.squeeze(self.user_emb(users), axis=1) # [batch_size, dim]
        for i in range(self.n_layers):
            user_vector = self.user_dnn[i](user_vector)
            user_vector = tf.math.tanh(user_vector)
            seq_vector  = self.app_dnn[i](seq_vector)
            seq_vector  = tf.math.tanh(seq_vector)
        
        # hadamard product / Eq.(10)
        combination = tf.matmul(user_vector, seq_vector, transpose_b=True)  # [batch_size, dim]
        # concat hidden vector and time vector / Eq.(13)
        combination = tf.concat([combination, time_vecs], axis=1) # [batch_size, dim+31]
                
        # softmax / Eq.(4)
        scores = self.classifier(combination)  # [batch_size, n_apps]
        
        if mode == 'predict':
            return scores  # [batch_size, n_apps]
        else:
            preds       = tf.math.top_k(scores,k=self.k).indices  # [batch_size, k]
            comparison  = tf.math.equal(tf.cast(preds, tf.int32), tf.cast(targets, tf.int32))
            indicator   = tf.reduce_sum(tf.cast(comparison, tf.float32), axis=1)  # [batch_size]
            coefficient = tf.math.pow(tf.convert_to_tensor([self.alpha] * indicator.shape[0]), indicator) # [batch_size]
            
            # loss_object = SparseCategoricalCrossentropy(reduction=losses_utils.ReductionV2.NONE)
            # loss        = loss_object(scores, tf.reshape(targets, [-1]))
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=tf.reshape(targets, [-1])) # [batch_size]
            loss = tf.math.reduce_mean(coefficient * loss) 

            return loss

    def call_chatgpt(self, users, time_vecs, app_seqs, time_seqs, targets, mode):
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
        
        if mode == 'predict':
            return scores
        else:
            preds = tf.nn.top_k(scores, k=self.k).indices
            indicator = tf.reduce_sum(tf.cast(tf.equal(preds, targets), tf.float32), axis=1)
            alpha = tf.constant(self.alpha, dtype=tf.float32)
            coefficient = tf.pow(alpha * tf.ones_like(indicator), indicator)
            
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=tf.reshape(targets, [-1])))
            loss = tf.reduce_mean(tf.multiply(coefficient, loss))
            return loss

if __name__ == '__main__':
    model = AppUsage2Vec(1000, 100, 64, 4, 2, 0.1, 5, 'cuda:gpu')
    pass
