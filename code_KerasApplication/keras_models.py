import tensorflow as tf

class VGG16:
    def __init__(self, IMG_X_SIZE, IMG_Y_SIZE, N_CLASSES):
        self.IMG_X_SIZE = IMG_X_SIZE
        self.IMG_Y_SIZE = IMG_Y_SIZE
        self.N_CLASSES = N_CLASSES

    def __call__(self, x, trainable):
        return self.build_network(x, trainable)

    def build_network(self, x, trainable):
        base_model = tf.keras.applications.VGG16(input_shape=(self.IMG_Y_SIZE, self.IMG_X_SIZE, 3),
                                                 include_top=False,
                                                 weights='imagenet')

        ### 畳み込み層の一部のみを利用する場合に使う箇所
        base_input_shape = base_model.layers[0].input_shape[0] # 
        print(base_input_shape) #
        base_model = tf.keras.Sequential(base_model.layers[:7]) #2block last pooling 7, 3block last 11 #
        print(type(base_model)) #
        print(base_model.layers) #
        base_model.summary() #
        input_layer = tf.keras.Input(shape=base_input_shape[1:], batch_size=base_input_shape[0]) #
        prev_layer = input_layer #
        for layer in base_model.layers: #
            prev_layer = layer(prev_layer) #
        base_model = tf.keras.models.Model([input_layer], [prev_layer]) #
        ### 箇所終わり

        feature_batch = base_model(x) 
        print(feature_batch.shape)
        # setting base_model parameter freeze(pre-train) or not(fine-tuning)
        base_model.trainable = trainable
        base_model.summary()
        #global average pooling
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)
        print(feature_batch_average.shape)
        #full connection layer
        fc_layer = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        feature_batch_fc = fc_layer(feature_batch_average)
        print(feature_batch_fc.shape)
        #convert to classifier vector
        prediction_layer = tf.keras.layers.Dense(self.N_CLASSES, activation=tf.nn.softmax)
        prediction_batch = prediction_layer(feature_batch_fc)
        print(prediction_batch.shape)
        #stack above layers
        model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            fc_layer,
            prediction_layer
        ])
        model.summary()
        return model

