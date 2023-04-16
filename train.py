from datetime import datetime
import os
import shutil
import numpy as np
import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Input
from keras.layers import Dropout
from keras.regularizers import l1_l2
from keras.callbacks import Callback

import tensorflow as tf
import cv2

epochs = 50000
patience = 50

# Directory to save the model to
output_dir = 'F:\\Software\\Hypetrigger\\data\\tensorflow-models\\ow2-elim'

# Dataset directory, contains one folder per class you want to train
dataset_dir = r'F:\Projects\Rust\testing\hypetrigger\output\Dataset'

# If debug is set to true it will do some extra things to help you debug your model (such as saving misclassified images to the output directory)
debug = False


backup = False
# if output_dir exists, set backup to true
if os.path.exists(output_dir):
    backup = True

def resize_image(image):
    resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    scaled_image = resized_image / 255.0

    return scaled_image

def get_base_model():
    return MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    # return ResNet50V2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    # return ResNet101V2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    # return Resnet152V2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

class SaveWeightsEveryNEpochs(Callback):
    def __init__(self, save_every_n_epochs, output_dir):
        self.save_every_n_epochs = save_every_n_epochs
        self.output_dir = output_dir
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_every_n_epochs == 0:
            weights_path = os.path.join(self.output_dir, f'weights_epoch_{epoch+1}.hdf5')
            self.model.save_weights(weights_path)
            print(f"Saved weights to {weights_path}")
    
    # Save the weights after the last epoch
    def on_train_end(self, logs=None):
        if backup:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_dir = os.path.join(os.path.dirname(output_dir), f"backup_{timestamp}")

            # Copy the old model to the backup directory (only the saved_model.pb and variables folder)
            shutil.copytree(output_dir, os.path.join(backup_dir, os.path.basename(output_dir)))

        weights_path = os.path.join(self.output_dir, f'weights_epoch_final.hdf5')
        self.model.save_weights(weights_path)
        print(f"Saved weights to {weights_path}")

def get_latest_checkpoint(checkpoint_dir):
    # Get the latest checkpoint
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.hdf5'))
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint
    
def create_data_generators(combined_dir):
    data_gen = ImageDataGenerator(
        # preprocessing_function=resize_image,
        preprocessing_function=resize_image,
        fill_mode='nearest',
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True,
        validation_split=0.3
    )

    train_gen = data_gen.flow_from_directory(
        combined_dir,
        target_size=(224, 224),
        batch_size=24,
        class_mode='categorical',
        subset='training',
        save_to_dir=f'{output_dir}/train' if debug else None,
        save_prefix='Train' if debug else None,
        save_format='png' if debug else None
    )

    val_gen = data_gen.flow_from_directory(
        combined_dir,
        target_size=(224, 224),
        batch_size=24,
        class_mode='categorical',
        subset='validation'
    )

    return (train_gen, val_gen)

def save_misclassified_images(model, val_gen, output_dir):
    misclassified_dir = os.path.join(output_dir, 'misclassified_images')
    if not os.path.exists(misclassified_dir):
        os.makedirs(misclassified_dir)

    for idx in range(val_gen.samples // val_gen.batch_size + 1):
        x_batch, y_batch = val_gen.next()
        y_pred = model.predict(x_batch)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_batch, axis=1)

        for i, (y_pred_class, y_true_class) in enumerate(zip(y_pred_classes, y_true_classes)):
            if y_pred_class != y_true_class:
                img = (x_batch[i] * 255).astype(np.uint8)
                img_filename = f"img_{idx}_{i}_true_{y_true_class}_pred_{y_pred_class}.png"
                img_filepath = os.path.join(misclassified_dir, img_filename)
                cv2.imwrite(img_filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def create_model(base_model, model_path=output_dir):
    input_layer = Input(shape=base_model.input_shape[1:], name='Image')

    # Replace the base model's input layer with the custom-named input layer
    x = base_model(input_layer)

    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
    x = Dropout(0.5)(x)  # Add dropout layer

    predictions = Dense(2, activation='softmax', name='Confidences')(x)

    model = Model(inputs=input_layer, outputs=predictions)

    # Freeze all layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

     # Load the latest checkpoint, if available
    latest_checkpoint = get_latest_checkpoint(model_path)
    if latest_checkpoint:
        print(f"Loading model weights from: {latest_checkpoint}")
        model.load_weights(latest_checkpoint)

    return model

def train_model(model, train_gen, val_gen):
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    
    # Add callback to save after N epochs
    save_weights_callback = SaveWeightsEveryNEpochs(15, output_dir)

    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // train_gen.batch_size,
        validation_data=val_gen,
        validation_steps=val_gen.samples // val_gen.batch_size,
        epochs=epochs,
        callbacks=[early_stopping, save_weights_callback],
        workers=16,
        # use_multiprocessing=True
    )

    return history

def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if debug:
        if not os.path.exists(os.path.join(output_dir, 'train')):
            os.makedirs(os.path.join(output_dir, 'train'))

    # Create the base model from the pre-trained model MobileNet V2
    base_model = get_base_model()

    # Create the data generators
    train_gen, val_gen = create_data_generators(dataset_dir)

    # Create the model
    model = create_model(base_model)

    # Train the model
    history = train_model(model, train_gen, val_gen)

    if debug:
        save_misclassified_images(model, val_gen, output_dir)

    # Print out final accuracy for the training and validation sets
    print('Training accuracy: ', history.history['accuracy'][-1])
    print('Training loss: ', history.history['loss'][-1])
    print('Validation accuracy: ', history.history['val_accuracy'][-1])
    print('Validation loss: ', history.history['val_loss'][-1])

    # Save the model
    tf.saved_model.save(model, output_dir)

    if debug:
        # Print model type
        print(type(model))

        # Print model summary
        model.summary()

if __name__ == '__main__':
    main()