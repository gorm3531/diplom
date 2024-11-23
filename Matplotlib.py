import matplotlib.pyplot as plt


#Первый график
def first():
    x = [1, 2, 3, 4, 5]
    y = [25, 32, 34, 20, 25]
    plt.plot(x, y)
    plt.show()


#Второй график
def second():
    year = [1978, 1986, 1999, 2004]
    population = [2.12, 3.681, 5.312, 6.782]
    plt.plot(year, population)
    plt.show()


#Добавляем заголовки

def first2_0():
    x = [1, 2, 3, 4, 5]
    y = [25, 32, 34, 20, 25]
    plt.plot(x, y)
    plt.xlabel('Ось х')
    plt.ylabel('Ось y')
    plt.title('Линейный график')
    plt.show()


def second2_0():
    year = [1978, 1986, 1999, 2004]
    population = [2.12, 3.681, 5.312, 6.782]
    plt.plot(year, population)
    plt.xlabel('Год')
    plt.ylabel('Популяция')
    plt.title('Линейный график')
    plt.show()


#Добавим немного красок

def first3_0():
    x = [1, 2, 3, 4, 5]
    y = [25, 32, 34, 20, 25]
    plt.plot(x, y, color='red', marker='o', markersize=7)
    plt.xlabel('Ось х')
    plt.ylabel('Ось y')
    plt.title('Линейный график')
    plt.show()


first3_0()


def second3_0():
    year = [1978, 1986, 1999, 2004]
    population = [2.12, 3.681, 5.312, 6.782]
    plt.plot(year, population, color='yellowgreen', marker='o', markersize=5)
    plt.xlabel('Год')
    plt.ylabel('Популяция')
    plt.title('Линейный график')
    plt.show()


second3_0()


#Диаграмма рассеяния

def sctterplot_1():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [25, 32, 34, 20, 25, 23, 21, 33, 19, 28]
    plt.scatter(x, y, color='purple')
    plt.show()


sctterplot_1()


def scatterplot_2():
    year = [1950, 1975, 2000, 2018]
    population = [2.12, 3.681, 5.312, 6.981]
    plt.scatter(year, population, color='green')
    plt.show()


scatterplot_2()


#Гистограмма

def bar_1():
    x = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май']
    y = [2, 4, 3, 1, 7]
    plt.bar(x, y, label='Величина прибыли', color='brown')
    plt.xlabel('Месяц года')
    plt.ylabel('Прибыль, в млн руб.')
    plt.title('Пример гистограммы')
    plt.legend()
    plt.show()


bar_1()

#Рисование нескольких кривых (Здесь добавляется новая библиотека- numpy)
import numpy as np


def linspace_1():
    x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    cos, sin = np.cos(x), np.sin(x)
    plt.plot(x, cos, color='pink')
    plt.plot(x, sin, color='yellowgreen')
    plt.show()


linspace_1()


#Комбинируем графики(Столбчатая диаграмма и линейный график)
def combo():
    x = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май']
    y = [2, 4, 3, 1, 7]
    plt.bar(x, y, label='Величина прибыли', color='indigo')
    plt.plot(x, y, color='palevioletred', marker='o', markersize=7)
    plt.xlabel('Месяц года')
    plt.ylabel('Прибыль, в млн руб.')
    plt.title('Соединяем графики')
    plt.legend()
    plt.show()


combo()


#увеличим прозрачность столбчатой диаграммы
def combo_01():
    x = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май']
    y = [2, 4, 3, 1, 7]
    plt.bar(x, y, label='Величина прибыли', color='indigo', alpha=0.5)
    plt.plot(x, y, color='palevioletred', marker='o', markersize=7)
    plt.xlabel('Месяц года')
    plt.ylabel('Прибыль, в млн руб.')
    plt.title('Соединяем графики')
    plt.legend()
    plt.show()


combo_01()


#Круговая диаграмма
def vals_1():
    vals = [24, 17, 53, 21, 35]
    labels = ['Ford', 'Toyota', 'BMW', 'Audi', 'Jaguar']
    plt.pie(vals, labels=labels)
    plt.title('Распределение марок автомобилей на дороге')
    plt.show()


vals_1()


#Усложнённая круговая диаграмма
def vals_hard():
    names = 'Tom', 'Nino', 'Harry', 'Jill', 'Meredith', 'George'
    speed = [8, 7, 12, 4, 3, 2]
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'teal', 'lightsalmon']
    explode = (0.1, 0, 0, 0, 0, 0)
    plt.pie(speed, explode=explode, labels=names, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.show()


vals_hard()


#Сложные визуализации(Столбчатый график с накоплением)
def phone():
    labels = ['2017', '2018', '2019', '2020', '2021']
    android_users = [85, 85.1, 86, 86.2, 86]
    ios_users = [14.5, 14.8, 13, 13.8, 13.0]
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(labels, android_users, width, label='Android')
    ax.bar(labels, ios_users, width, bottom=android_users, label='ios')
    ax.set_ylabel('Соотношение, в %')
    ax.set_title('Распределение устройств на Android и iOS')
    ax.legend(loc='lower left', title='Устройства')
    plt.show()


phone()

#Тепловая карта(Здесь, понадобится добавить ещё 1 библиотеку)
#import numpy as np
import numpy.random


def heatmap_1():
    temperature = np.random.randn(4096)
    anger = np.random.randn(4096)

    heatmap, xedges, yedges = np.histogram2d(temperature, anger, bins=(64, 64))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.ylabel('Anger')
    plt.xlabel('Temp')
    plt.imshow(heatmap, extent=extent)
    plt.show()


heatmap_1()
