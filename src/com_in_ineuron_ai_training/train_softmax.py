from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# from keras.models import load_model
import matplotlib.pyplot as plt
# from softmax import SoftMax
import numpy as np
import argparse
import pickle

# Construct the argumet parser and parse the argument
from src.com_in_ineuron_ai_detectfaces_mtcnn.Configurations import get_logger
from src.com_in_ineuron_ai_training.softmax import SoftMax


class TrainFaceRecogModel:

    def __init__(self, args):

        self.args = args
        self.logger = get_logger()
        # Load the face embeddings
        self.data = pickle.loads(open(args["embeddings"], "rb").read())

    def trainKerasModelForFaceRecognition(self):
        # Encode the labels
        le = LabelEncoder()
        labels = le.fit_transform(self.data["names"])
        num_classes = len(np.unique(labels))
        labels = labels.reshape(-1, 1)
        #one_hot_encoder = OneHotEncoder(categorical_features = [0])
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)


        labels = one_hot_encoder.fit_transform(labels)

        embeddings = np.array(self.data["embeddings"])

        # Initialize Softmax training model arguments
        BATCH_SIZE = 8
        EPOCHS = 5
        input_shape = embeddings.shape[1]

        # Build sofmax classifier
        softmax = SoftMax(input_shape=(input_shape,), num_classes=num_classes)
        model = softmax.build()

        # Create KFold
        cv = KFold(n_splits = 5, random_state = 42, shuffle=True)
        history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}


        # Train
        for train_idx, valid_idx in cv.split(embeddings):
            X_train, X_val, y_train, y_val = embeddings[train_idx], embeddings[valid_idx], labels[train_idx], labels[valid_idx]
            his = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_val, y_val))
            print(his.history.get('accuracy', 'Key not found'))


            history['accuracy'] += his.history['accuracy']
            history['val_accuracy'] += his.history['val_accuracy']
            history['loss'] += his.history['loss']
            history['val_loss'] += his.history['val_loss']

            self.logger.info(f"Train Acc: {his.history.get('accuracy', 'N/A')}, Val Acc: {his.history.get('val_accuracy', 'N/A')}")



        # write the face recognition model to output
        model.save(self.args['model'])
        f = open(self.args["le"], "wb")
        f.write(pickle.dumps(le))
        f.close()
