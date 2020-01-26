import os
import numpy as np
import tensorflow as tf
import datasets
import keras_models

base_dir = os.path.expanduser("~/Caltech101")
log_dir = os.path.join(base_dir,"log")
model_name = "Caltech_transfer"
os.makedirs(os.path.join(log_dir,model_name), exist_ok=True)

data_dir = os.path.join(*[base_dir,"data","101_ObjectCategories"])

datasize = {"train":28, "valid":2}
IMG_X_SIZE, IMG_Y_SIZE = 256, 256
N_CLASSES = 102

n_epoch = 50

trainable = [True,False][1]
batch_size, val_batch = 64,32

if __name__ == "__main__":
    with tf.device('/device:CPU:0'):
        data = datasets.caltechdata(data_dir=data_dir,
                                    datasize=datasize,
                                    IMG_X_SIZE=IMG_X_SIZE, 
                                    IMG_Y_SIZE=IMG_Y_SIZE,
                                    N_CLASSES=N_CLASSES)
        train_datas, validation_datas, test_datas, num_train, num_val, num_test = data()
        print(train_datas, validation_datas, test_datas)
    with tf.device('/device:GPU:1'):
        test_datas = test_datas.batch(val_batch)

        def change_range(image,label):
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            return (image/255.0-mean)/std, label
        test_datas = test_datas.map(change_range)

    for image_batch, label_batch in test_datas.take(1):
        pass

    print(image_batch.shape)

    network = keras_models.VGG16(IMG_X_SIZE=IMG_X_SIZE,
                                 IMG_Y_SIZE=IMG_Y_SIZE,
                                 N_CLASSES=N_CLASSES)
    model = network(x=image_batch,trainable=trainable)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    print(len(model.trainable_variables))

    checkpoint_path = os.path.join(*[log_dir,model_name,"cp-{epoch:04d}.ckpt"])
    checkpoint_dir = os.path.dirname(checkpoint_path)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt:
        last_model = ckpt.model_checkpoint_path
        model.load_weights(last_model)
        print("restore parmeter")
                        
    test_step = num_test // val_batch
    test_loss, test_accuracy = model.evaluate(test_datas, steps = test_step)

    print("test loss: {:.2f}".format(test_loss))
    print("test accuracy: {:.2f}".format(test_accuracy))
