import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model

model = load_model('Vgg.h5')

Y = {'blasti': 0,
     'bonegl': 1,
     'brhkyt': 2,
     'cbrtsh': 3,
     'cmnmyn': 4,
     'gretit': 5,
     'hilpig': 6,
     'himbul': 7,
     'himgri': 8,
     'hsparo': 9,
     'indvul': 10,
     'jglowl': 11,
     'lbicrw': 12,
     'mgprob': 13,
     'rebimg': 14,
     'wcrsrt': 15}

labels = list(Y.keys())


class ImageClassifierApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Classifier")
        self.load_button = tk.Button(self.master, text="Load Image", command=self.load_image)
        self.load_button.pack()
        self.label = tk.Label(self.master)
        self.label.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            image = Image.open(file_path)
            photo = ImageTk.PhotoImage(image)
            image_label.config(image=photo)
            image_label.image = photo
            padding_label = tk.Label(root, height=1)
            padding_label.pack()
            image = image.resize((224, 224))  # Resize image to match the input size of your CNN model
            image = np.array(image) / 255.0  # Normalize the image data
            image = np.expand_dims(image, axis=0)  # Add a batch dimension
            prediction = model.predict(image)
            predicted_class = labels[np.argmax(prediction)]
            class_label = "Bird" if predicted_class == 0 else "Non-Bird"
            self.label.config(text="Predicted Class: {}".format(predicted_class))


root = tk.Tk()
root.configure(bg="grey")
root.title("Bird Species Classification")
root.geometry("600x400")
heading_label = tk.Label(root, text="Bird Species Classification", font=("Arial", 22))
heading_label.pack(pady=20)
label = tk.Label(root, text="Upload your image")
label.pack()
image_label = tk.Label(root)
image_label.pack()
app = ImageClassifierApp(root)

root.mainloop()
