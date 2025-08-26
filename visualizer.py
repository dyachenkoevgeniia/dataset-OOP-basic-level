import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
class Visualizer:
    def __init__(self,data):
        self.data=data

    def hist(self,column):
        sns.histplot(data=self.data, x=column, kde=True)
        plt.show()

    def scatter(self,x,y, hue=None):
        sns.scatterplot(x=x, y=y, data=self.data, hue=hue)
        plt.show()

    def heatmap_corr(self):
        corr=self.data.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.show()

    def boxplot(self,column):
        sns.boxplot(y=self.data[column])
        plt.show()

    def pairplot(self, columns):
        sns.pairplot(self.data[columns])

    def countplot(self,column):
        sns.countplot(x=column, data=self.data)
        plt.show()

    def violinplot(self,column):
        sns.violinplot(y=self.data[column])
        plt.show()

    def kdeplot(self,column):
        sns.kdeplot(self.data[column], fill=True)
        plt.show()

    def missing_values_heatmap(self,kind="heatmap"):
        if kind == "matrix":
            msno.matrix(self.data)
        elif kind == "bar":
            msno.bar(self.data)
        else:
            msno.heatmap(self.data)
        plt.show()