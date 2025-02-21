from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adam
from keras.utils import to_categorical
main = Tk()
main.title("PROFILE DUPLICATION DETECTION IN SOCIAL MEDIA")
main.geometry("1300x1200")
main.config(bg="lightgreen")

global filename
global X,Y
global X_train, X_test, y_train, y_test
global accuracy
global dataset
global model



def loadProfileDataset():    
    global filename, dataset
    outputarea.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    outputarea.insert(END, filename + " loaded\n\n")
    
    try:
        dataset = pd.read_csv(filename)
        outputarea.insert(END, "Dataset Loaded Successfully!\n\n")
        outputarea.insert(END, str(dataset.head()) + "\n\n")
        print(dataset.info())  # Debugging: Print dataset structure in terminal
    except Exception as e:
        outputarea.insert(END, "Error Loading Dataset: " + str(e) + "\n\n")
        print("Error:", e)

def preprocessDataset():
    global X, Y, dataset, X_train, X_test, y_train, y_test
    outputarea.delete('1.0', END)

    try:
        X = dataset.iloc[:, 0:8].values  # Use .iloc to avoid issues
        Y = dataset.iloc[:, 8].values  
        
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X, Y = X[indices], Y[indices]
        
        Y = to_categorical(Y, num_classes=2)  # Ensure Y is one-hot encoded properly

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        outputarea.insert(END, f"Dataset contains total profiles: {len(X)}\n")
        outputarea.insert(END, f"Training Profiles: {len(X_train)}\n")
        outputarea.insert(END, f"Testing Profiles: {len(X_test)}\n")
        print(f"X_train Shape: {X_train.shape}, Y_train Shape: {y_train.shape}")  # Debugging
    except Exception as e:
        outputarea.insert(END, "Error in Preprocessing: " + str(e) + "\n\n")
        print("Preprocessing Error:", e)

def executeANN():
    global model, accuracy, X_train, X_test, y_train, y_test
    outputarea.delete('1.0', END)

    try:
        model = Sequential([
            Dense(200, input_shape=(8,), activation='relu'),
            Dense(200, activation='relu'),
            Dense(2, activation='softmax')  # Ensure 2 output neurons
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print("ANN Model Summary:")
        model.summary()  # Debugging

        hist = model.fit(X_train, y_train, batch_size=5, epochs=200, verbose=2)

        results = model.evaluate(X_test, y_test)
        accuracy = hist.history
        acc = accuracy['accuracy'][-1] * 100  # Get last epoch accuracy

        outputarea.insert(END, f"Model Trained! Accuracy: {acc:.2f}%\n")
        print("Model Training Completed Successfully!")

    except Exception as e:
        outputarea.insert(END, "Error in Training: " + str(e) + "\n\n")
        print("Training Error:", e)

    
def graph():
    global accuracy
    if not accuracy:
        outputarea.insert(END, "Error: No accuracy data available!\n")
        return

    try:
        acc = accuracy['accuracy']
        loss = accuracy['loss']

        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy/Loss')
        plt.plot(acc, 'g-', label="Accuracy")
        plt.plot(loss, 'b-', label="Loss")
        plt.legend(loc='upper left')
        plt.title('ML Accuracy & Loss Graph')
        plt.show()
    except Exception as e:
        outputarea.insert(END, "Graph Error: " + str(e) + "\n\n")
        print("Graph Error:", e)


def predictProfile():
    outputarea.delete('1.0', END)
    global model

    try:
        filename = filedialog.askopenfilename(initialdir="Dataset")
        test = pd.read_csv(filename).iloc[:, 0:8].values  # Ensure correct shape
        
        predictions = model.predict(test)
        predicted_classes = np.argmax(predictions, axis=1)  # Convert softmax output to class

        for i in range(len(test)):
            msg = "Genuine" if predicted_classes[i] == 0 else "Duplicated"
            outputarea.insert(END, f"Profile {i+1}: {msg}\n")

    except Exception as e:
        outputarea.insert(END, "Prediction Error: " + str(e) + "\n\n")
        print("Prediction Error:", e)
  
        
def close():
    main.destroy()

font = ('times', 15, 'bold')
title = Label(main, text='PROFILE DUPLICATION DETECTION IN SOCIAL MEDIA')
#title.config(bg='powder blue', fg='olive drab')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Social Network Profiles Dataset", command=loadProfileDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
processButton.place(x=20,y=150)
processButton.config(font=ff)

annButton = Button(main, text="Run ML Algorithm", command=executeANN)
annButton.place(x=20,y=200)
annButton.config(font=ff)

graphButton = Button(main, text=" ML Accuracy & Loss Graph", command=graph)
graphButton.place(x=20,y=250)
graphButton.config(font=ff)

predictButton = Button(main, text="Predict Duplicate/Genuine Profile using ML", command=predictProfile)
predictButton.place(x=20,y=300)
predictButton.config(font=ff)

exitButton = Button(main, text="Logout", command=close)
exitButton.place(x=20,y=350)
exitButton.config(font=ff)


font1 = ('times', 12, 'bold')
outputarea = Text(main,height=30,width=85)
scroll = Scrollbar(outputarea)
outputarea.configure(yscrollcommand=scroll.set)
outputarea.place(x=400,y=100)
outputarea.config(font=font1)

main.config()
main.mainloop()
