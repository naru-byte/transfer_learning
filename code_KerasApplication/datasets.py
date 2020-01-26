import os
import tensorflow as tf
from utils import crop, flip

def preprocess_image(image, IMG_X_SIZE, IMG_Y_SIZE):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_Y_SIZE, IMG_X_SIZE])
    return image

def load_and_preprocess_image(path, IMG_X_SIZE, IMG_Y_SIZE):
    image = None
    try:
        image = tf.io.read_file(path)
        image = preprocess_image(image, IMG_X_SIZE, IMG_Y_SIZE)
    except Exception as e:
        print(path)
        print(e)
        print("error occured")
    return image

class caltechdata:
    def __init__(self, data_dir, datasize, IMG_X_SIZE, IMG_Y_SIZE, N_CLASSES):
        self.data_dir = data_dir
        self.datasize = datasize
        self.IMG_X_SIZE = IMG_X_SIZE
        self.IMG_Y_SIZE = IMG_Y_SIZE
        self.N_CLASSES = N_CLASSES

    def __call__(self):
        return self.load_datas()

    def train_load_and_preprocess_from_path_label_c(self, path, label, crop_bool, flip_bool):
        image = load_and_preprocess_image(path, self.IMG_X_SIZE, self.IMG_Y_SIZE)
        if flip_bool:
            image = flip(image)
        if crop_bool:
            image = crop(image, self.IMG_Y_SIZE, self.IMG_X_SIZE)
        return image, tf.one_hot(indices=[label], depth=self.N_CLASSES)[0]

    def validation_load_and_preprocess_from_path_label_c(self, path, label):
        image = load_and_preprocess_image(path, self.IMG_X_SIZE, self.IMG_Y_SIZE)
        return image, tf.one_hot(indices=[label], depth=self.N_CLASSES)[0]

    def load_datas(self):
        train_size, valid_size = self.datasize["train"], self.datasize["valid"]

        train_x, train_y, val_x, val_y, test_x, test_y = [], [], [], [], [], []
        crop_bool, flip_bool = [], []
        folders = sorted(os.listdir(self.data_dir))
        # print(folders)
        for cnt, folder in enumerate(folders):
            images = sorted(os.listdir(os.path.join(self.data_dir,folder)))
            for image in images[:train_size]:
                for i in [True,False]:
                    for j in [True,False]:
                        train_x.append(os.path.join(*[self.data_dir,folder,image]))
                        train_y.append(cnt)
                        crop_bool.append(i)
                        flip_bool.append(j)

            for image in images[train_size:train_size+valid_size]:
                val_x.append(os.path.join(*[self.data_dir,folder,image]))
                val_y.append(cnt)

            for image in images[train_size+valid_size:]:
                test_x.append(os.path.join(*[self.data_dir,folder,image]))
                test_y.append(cnt)

        train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y, crop_bool, flip_bool))
        train_ds = train_ds.map(self.train_load_and_preprocess_from_path_label_c)
        validation_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
        validation_ds = validation_ds.map(self.validation_load_and_preprocess_from_path_label_c)
        test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
        test_ds = test_ds.map(self.validation_load_and_preprocess_from_path_label_c)
        return train_ds, validation_ds, test_ds, len(train_x), len(val_x), len(test_x)
