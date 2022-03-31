import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix ,classification_report,accuracy_score,f1_score,precision_score,recall_score
from scikitplot.metrics import plot_confusion_matrix
import seaborn as sn
import cv2 as cv
import pandas as pd
import numpy as np



class Metrics():
    def __init__(self,y_target,y_predict, classes):
        
        self.targets = y_target
        self.predicts = y_predict
        self.classes = classes

    def get_accuracy(self):
        accuracy = accuracy_score(self.targets, self.predicts)
        return accuracy
        
    def get_precision_score(self):
        precision = precision_score(self.targets, self.predicts,average='weighted')
        return precision

    def plot_confusion_matrix1(self):
        fig, axs = plt.subplots(figsize=(20, 20))
        plot_confusion_matrix(self.targets, self.predicts , ax=axs,normalize=True)
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=0)
        plt.yticks(tick_marks, self.classes)
        plt.savefig('results/confusion_matrix1.png')
        return fig

    def plot_confusion_matrix2(self):
        fig, axs = plt.subplots(figsize = (20,20))
        cf_matrix = confusion_matrix(self.targets, self.predicts)
        df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in self.classes], columns = [i for i in self.classes])
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig('results/confusion_matrix2.png')
        return fig

    def get_classification_report(self):
        class_report = classification_report(self.targets, self.predicts,target_names=self.classes)
        return class_report

    def show_images(self, image):
        fig, axs = plt.figure(figsize=(25,4))
        for idx in np.arange(5):
            ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks =[])
            image = image/2 + 0.5
            plt.imshow(np.transpose(image, (1,2,0)))
            ax.set_title(self.classes[self.targets[idx]])
        return fig
