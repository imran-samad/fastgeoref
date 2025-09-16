# modelbuild.py
# This script allows users to build a cnn for object/animal detection

print(">>> modelbuild module imported")

import numpy as np
import os
import cv2
import random
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam


# ---------------------------
# Data augmentation functions
# ---------------------------
def apply_random_transformations(image):
    transformations = [blur_image, rotate_image, color_jittering]
    transformation = random.choice(transformations)
    return transformation(image)

def blur_image(image):
    kernel_size = random.choice([(3, 3), (5, 5), (7, 7)])
    return cv2.GaussianBlur(image, kernel_size, 0)

def rotate_image(image):
    angle = random.choice([90, 180, 270])
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    else:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def color_jittering(image):
    brightness = random.uniform(0.8, 1.2)
    contrast = random.uniform(0.8, 1.2)
    saturation = random.uniform(0.8, 1.2)
    hue = random.uniform(-0.1, 0.1)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv_image[:, :, 1] *= saturation
    hsv_image[:, :, 2] *= brightness
    hsv_image[:, :, 0] += hue * 360

    hsv_image[:, :, 1:3] = np.clip(hsv_image[:, :, 1:3], 0, 255)
    jittered = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

    jittered = np.clip(contrast * (jittered.astype(np.float64) - 128) + 128, 0, 255)
    return jittered.astype(np.uint8)


# ---------------------------
# CNN definition
# ---------------------------
def def_cnn(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='linear', padding='same')(inputs)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(96, (3, 3), activation='linear', padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), activation='linear', padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(256, (3, 3), activation='linear', padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    flattened = Flatten()(x)
    dense_output = Dense(96, activation='linear')(flattened)
    dense_output = LeakyReLU(alpha=0.1)(dense_output)
    dense_output = Dropout(0.3)(dense_output)

    dense_output = Dense(128, activation='linear')(dense_output)
    dense_output = LeakyReLU(alpha=0.1)(dense_output)
    dense_output = Dropout(0.3)(dense_output)

    dense_output = Dense(num_classes, activation='softmax')(dense_output)

    return Model(inputs, dense_output)


# ---------------------------
# Training workflow
# ---------------------------
def run_training(rt_path, save_model_path, img_size, batch_size, epochs):
    cls0 = os.path.join(rt_path, 'animal')
    cls1 = os.path.join(rt_path, 'background')

    lst0 = [f for f in os.listdir(cls0) if not f.startswith('.')]
    lst1 = [f for f in os.listdir(cls1) if not f.startswith('.')]

    # Data augmentation for balancing
    if len(lst0) != len(lst1):
        output_directory = os.path.join(rt_path, 'balanced_dataset')
        os.makedirs(output_directory, exist_ok=True)
        diff = abs(len(lst0) - len(lst1))
        minority_class = lst0 if len(lst0) < len(lst1) else lst1
        for ag in range(diff):
            im_pos = random.randrange(len(minority_class))
            im = cv2.imread(os.path.join(cls1 if len(lst0) > len(lst1) else cls0, minority_class[im_pos]))
            transformed_im = apply_random_transformations(im)
            image_name = f'aug_{ag}_{minority_class[im_pos]}'
            cv2.imwrite(os.path.join(output_directory, image_name), transformed_im)
        messagebox.showinfo("Info", "Balanced dataset created. Please copy all files from the 'balanced dataset' folder to the animal or background folder (whichever is appropriate) manually and rerun.")
        return

    # Prepare dataset
    lst = lst0 + lst1
    trainData = np.ndarray((len(lst), img_size, img_size, 3), dtype='uint8')
    targetData = np.hstack((np.zeros(len(lst0)), np.ones(len(lst1))))

    for i in range(len(lst)):
        if i < len(lst0):
            im = cv2.imread(os.path.join(cls0, lst[i]))
        else:
            im = cv2.imread(os.path.join(cls1, lst[i - len(lst0)]))
        if im is not None:
            trainData[i] = cv2.resize(im, (img_size, img_size))

    targetH = to_categorical(targetData)
    train_X = trainData.astype('float32') / 255.0
    train_X, valid_X, train_label, valid_label = train_test_split(
        train_X, targetH, test_size=0.4, random_state=13
    )

    num_classes = 2
    model = def_cnn((img_size, img_size, 3), num_classes)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(valid_X, valid_label))

    model.save(save_model_path)
    messagebox.showinfo("Success", f"Model saved to: {save_model_path}")


# ---------------------------
# GUI
# ---------------------------
def gui_entry():
    """Tkinter form for CNN model training inputs"""
    root = tk.Tk()
    root.title("CNN Model Builder")

    inputs = {}

    # --- Path selection functions ---
    def select_training_data():
        path = filedialog.askdirectory(title="Select Training Data Folder")
        if path:
            training_entry.delete(0, tk.END)
            training_entry.insert(0, path)

    def select_model_path():
        path = filedialog.asksaveasfilename(
            defaultextension=".h5py",
            filetypes=[("HDF5 Model", "*.h5py"), ("All Files", "*.*")]
        )
        if path:
            model_entry.delete(0, tk.END)
            model_entry.insert(0, path)

    # --- Form frame ---
    form = tk.Frame(root, padx=10, pady=10)
    form.pack()

    # --- Training data path ---
    tk.Label(form, text="Training Data Path:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    training_entry = tk.Entry(form, width=50)
    training_entry.grid(row=0, column=1, padx=5, pady=5)
    tk.Button(form, text="Browse", command=select_training_data).grid(row=0, column=2, padx=5, pady=5)
    inputs["training_data_path"] = training_entry

    # --- Model save path ---
    tk.Label(form, text="Save Model Path:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
    model_entry = tk.Entry(form, width=50)
    model_entry.insert(0, "cnn_model_balanced.h5py")
    model_entry.grid(row=1, column=1, padx=5, pady=5)
    tk.Button(form, text="Browse", command=select_model_path).grid(row=1, column=2, padx=5, pady=5)
    inputs["save_model_path"] = model_entry

    # --- Default parameter values ---
    params = {
        "image_size": 75,
        "batch_size": 64,
        "epochs": 25
    }

    # --- Parameters section ---
    row = 2
    for key, default in params.items():
        tk.Label(form, text=f"{key}:").grid(row=row, column=0, sticky="e", padx=5, pady=5)
        entry = tk.Entry(form)
        entry.insert(0, str(default))
        entry.grid(row=row, column=1, padx=5, pady=5)
        inputs[key] = entry
        row += 1

    # --- Run button ---
    def run_clicked():
        try:
            training_path = inputs["training_data_path"].get()
            save_model_path = inputs["save_model_path"].get()
            img_size = int(inputs["image_size"].get())
            batch_size = int(inputs["batch_size"].get())
            epochs = int(inputs["epochs"].get())

            root.destroy()
            run_training(training_path, save_model_path, img_size, batch_size, epochs)
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    tk.Button(form, text="Run Training", command=run_clicked,
              bg="green", fg="white", padx=10, pady=5).grid(row=row, column=1, pady=10)

    root.mainloop()

    return inputs

#if __name__ == "__main__":
def main():
    gui_entry()
