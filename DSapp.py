import streamlit as st
import numpy as np
import pandas as pd
#from scipy.stats import mode
import matplotlib.pyplot as plt
#from streamlit import session_state as ss
import seaborn as sns

def open_excel_data(path="Data/Salaries.xlsx"): # функция для открытия данных для зарплат
    df = pd.read_excel(path)
    return df

def open_csv_data(path="Data/inflation_data.csv"): # функция для открытия данных по инфляции
    inflation_df = pd.read_csv(path)
    return inflation_df

def inflation_convert(inflation_df): #преобразование данных по инфляции
    # Извлечение первого и последнего столбца
    first_column = inflation_df.iloc[:, 0]  # Первый столбец
    last_column = inflation_df.iloc[:, -1]  # Последний столбец

    # Создание нового DataFrame с первым и последним столбцом
    filtered_result = pd.concat([first_column, last_column], axis=1)
    filtered_result = filtered_result[(filtered_result['Год'] >= 2000) & (filtered_result['Год'] <= 2023)]

    #Создание сводной таблицы для переменной inflation_rate, в которой значения сгруппированы по столбцам “Год” и значениям в столбце “Всего”.
    inflation_rate = filtered_result.pivot_table(columns="Год", values="Всего")
    
    inflation_rate = inflation_rate.values[0][0:] # преобразование в массив
    
    #Вычисление накопленной инфляции по годам с округлением значений и сохранением в new_inflation.
    inflation=0 # суммарная инфляция по годам
    summarized_inflation=[] # массив для сохранения суммарной инфляции для каждого года
    for i in inflation_rate:
        inflation+=i
        summarized_inflation.append(inflation.round())
    
    return summarized_inflation # возвращаем массив суммарной инфляции

def plot_decorator(func): # декоратор для построения графика
    def wrapper(*args, **kwargs):
        df = open_excel_data()
        inflation_df = open_csv_data()
        years = df.columns[1:].astype(int) # получаем года из названий столбцов
        years = np.array(years)  # Преобразуем список в массив NumPy
        summarized_inflation = inflation_convert(inflation_df)  # преобразуем данные по инфляции
        if 'start_df' not in st.session_state:
            st.session_state.start_df = df # сохраняем данные в состоянии сессии
        activities = df['Экономическая деятельность'].values
        for i in activities:
            economic_activity = df[df['Экономическая деятельность'] == i].values[0][1:]
            #st.write(economic_activity)
            # Инициализируем переменную для совокупного коэффициента инфляции
            cumulative_inflation = 0

            # Рассчитываем реальную зарплату с учетом совокупной инфляции
            real_salary = []
            for nominal, cumulative_inflation in zip(economic_activity, summarized_inflation):
                #print(cumulative_inflation)
                real_salary.append(nominal/(1+cumulative_inflation/100))
            #print(real_salary)

            # Построение графика
            sns.set_theme(style="whitegrid")  # Выбор стиля
            plt.figure(figsize=(12, 6))  # Установка размера фигуры

            bar_width = 0.4  # Ширина столбцов

            # Выполнение основной логики построения графика
            #func(ax, years, economic_activity, real_salary, bar_width, i, *args, **kwargs)

            # Добавление подписей
            #plt.title(i)
            #plt.xlabel("Годы")
            #plt.ylabel("Зарплата (рубли)")  

            # Перемещаем легенду за пределы графика
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

            # Отображение сетки
            plt.grid(True)
            # Отображение графика
            fig, ax = plt.subplots()
            ax.set_title(i)
            ax.set_xlabel("Годы")
            func(ax, years, economic_activity, real_salary, bar_width, i, *args, **kwargs)
            st.pyplot(fig) # выводим график в Streamlit
            #plt.show()
    return wrapper

#@data_decorator
@plot_decorator
def plot_linear(ax, years, economic_activity, real_salary, bar_width, i): # функция для построения линейного графика зарплат
    sns.lineplot(x=years, y=economic_activity, label=f"{i} - Без учета инфляции") # строим график зп без учета инфляции
    sns.lineplot(x=years, y=real_salary, label=f"{i} - С учетом инфляции") # строим график зп с учетом инфляции
    plt.ylabel("Зарплата (рубли)")  
    print('Абсолютное значение повышения зарплаты БЕЗ учета инфляции для направления',i,':', round((economic_activity[-1]-economic_activity[0])))
    print('Абсолютное значение повышения зарплаты с учетом инфляции для направления',i,':', round((real_salary[-1]-real_salary[0])))

#@data_decorator
@plot_decorator
def plot_bar(ax, years, economic_activity, real_salary, bar_width, i): # функция для построения графика зарплат
    ax.bar(years, economic_activity, width=bar_width, label='Nominal Salary')
    #plt.bar(years - bar_width/2, economic_activity, width=bar_width, label='Nominal Salary')
    #plt.bar(years + bar_width/2, real_salary, width=bar_width, label='Real Salary')
    ax.bar(years + bar_width/2, real_salary, width=bar_width, label='Real Salary')
    plt.ylabel("Зарплата (рубли)")  
    
#@data_decorator
@plot_decorator
def plot_bar_inflation_percentage(ax, years, economic_activity, real_salary, bar_width, i): # функция для построения графика инфляции
    #расчет процентного изменения
    nominal_change = pd.Series(economic_activity).pct_change().dropna() * 100
    real_change = pd.Series(real_salary).pct_change().dropna() * 100
    years = np.array(years)  # Преобразуем список в массив NumPy
    # Построение столбцов для процентного изменения номинальной зарплаты
    plt.bar(years[1:] - bar_width/2, nominal_change, width=bar_width, label='Nominal Salary Change (%)')
    # Построение столбцов для процентного изменения реальной зарплаты
    plt.bar(years[1:] + bar_width/2, real_change, width=bar_width, label='Real Salary Change (%)')
    plt.ylabel("Изменение (%)")


def sidebar_input_features():    
    graph_choice = st.sidebar.selectbox("Выберите тип отображения графика", ['Столбчатый', 'Линейный']) # выбор типа графика
    return graph_choice

def process_side_bar_inputs():
    # Открываем данные и сохраняем в соответсвующую переменную
    """df = open_excel_data()
    inflation_df = open_csv_data()
    years = df.columns[1:].astype(int) # получаем года из названий столбцов
    summarized_inflation = inflation_convert(inflation_df) # преобразуем данные по инфляции"""
    graph_choice = sidebar_input_features() # обрабатываем входные данные для боковой панели
    if st.sidebar.button('Show Inflation Percentage Chart'): 
        st.session_state.show_inflation_chart = True
        st.session_state.show_salary_chart = False
        plot_bar_inflation_percentage()
        return
    if graph_choice == 'Линейный':
        plot_linear()
        st.session_state.show_salary_chart = True
        st.session_state.show_inflation_chart = False
        return
    elif graph_choice == 'Столбчатый':
        plot_bar()
        st.session_state.show_salary_chart = True
        st.session_state.show_inflation_chart = False
        return
    

if __name__ == "__main__":
    process_side_bar_inputs()
    
    