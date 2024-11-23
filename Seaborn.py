import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset("tips")
data1 = sns.load_dataset("iris")


#Первое знакомство
def titanic_():
    titanic = sns.load_dataset('titanic')
    sns.histplot(titanic['age'], kde=True)
    plt.show()


titanic_()

#Загрузка собственных наборов данных
import pandas as pd

custom_data = pd.read_csv("class.csv")


#Линейные графики
def lineplot_():
    sns.lineplot(x="total_bill", y="tip", data=data, color='red')
    plt.show()


lineplot_()


#Гистограммы
def histplot_():
    sns.histplot(data["total_bill"], bins=20, kde=True)
    plt.show()


histplot_()


#Диаграммы рассеяния/строка
def scatterplot():
    sns.scatterplot(x="total_bill", y="tip", data=data)
    plt.show()


scatterplot()


def scatterplot_1():
    titanic = sns.load_dataset("titanic")
    titanic.head()
    sns.set_palette('icefire')
    sns.stripplot(x="age", y="who", hue="alive", data=titanic, jitter=0.5, linewidth=1)
    plt.show()


scatterplot_1()


def scatterplot_2():
    titanic = sns.load_dataset("titanic")
    titanic.head()
    sns.set_palette("pastel")
    sns.stripplot(x="age", y="who", hue="alive", data=titanic)
    plt.show()


scatterplot_2()


#Ящик с усами(Boxplot)
def boxplot_():
    sns.boxplot(x="day", y="total_bill", data=data, color='bisque')
    plt.show()


boxplot_()


#Создание сложных визуализаций
#Совмещение графиков(График с линейной регрессией поверх диаграммы рассеяния)
def implot_():
    sns.lmplot(x="total_bill", y="tip", data=data, line_kws={'color': 'purple'})
    plt.show()


implot_()


#Использование FaceGrid
def face():
    g = sns.FacetGrid(data, col="smoker", row="time")
    g.map(sns.scatterplot, "total_bill", "tip", color='c')
    plt.show()


face()
#Тепловые карты
import numpy as np


def heatmap_():
    data = np.random.randint(low=1,
                             high=100,
                             size=(10, 10))
    annot = True
    cmap = 'Greens'
    hm = sns.heatmap(data=data,
                     annot=annot,
                     cmap=cmap)
    plt.show()


heatmap_()


#Примеры использования
#Анализ данных о чаевых
#Гистограмма распределения чаевых
def exp1():
    sns.histplot(data['tip'], bins=20, kde=True, color='slategrey')
    plt.show()


exp1()


#Диаграмма рассеяния чаевых и общего счёта
def exp2():
    sns.scatterplot(x='total_bill', y='tip', hue='time', data=data)
    sns.color_palette('pastel')
    plt.show()


exp2()


#Boxplot чаевых по дням недели
def exp3():
    sns.boxplot(x='day', y='tip', data=data, color='rebeccapurple')
    plt.show()


exp3()
