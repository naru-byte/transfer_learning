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
        train_datas = train_datas.shuffle(buffer_size=1000).batch(batch_size)
        validation_datas = validation_datas.batch(val_batch)
        test_datas = test_datas.batch(val_batch)

        def change_range(image,label):
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            return (image/255.0-mean)/std, label
        train_datas = train_datas.map(change_range)
        validation_datas = validation_datas.map(change_range)
        test_datas = test_datas.map(change_range)

    for image_batch, label_batch in train_datas.take(1):
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

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq='epoch')

    validation_steps = num_val // val_batch
    loss0,accuracy0 = model.evaluate(validation_datas, steps = validation_steps)
    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    history = model.fit(train_datas,
                        epochs=n_epoch,
                        validation_data=validation_datas,
                        callbacks=[cp_callback])
                        
    test_step = num_test // val_batch
    test_loss, test_accuracy = model.evaluate(test_datas, steps = test_step)
