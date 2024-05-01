import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Создание экземпляров LabelEncoder и подготовка категориальных признаков
label_encoders = {}
categorical_cols = ['transmission', 'fuelType']
for col in categorical_cols:
    label_encoders[col] = LabelEncoder().fit(
        ['Manual', 'Semi-Auto', 'Automatic', 'Other']) if col == 'transmission' else LabelEncoder().fit(
        ['Petrol', 'Diesel', 'Hybrid', 'Other', 'Electric'])


# Загрузка данных
@st.cache_data
def load_data():
    return pd.read_csv("CarsData.csv")


# Функция загрузки обученной модели
@st.cache_data
def load_trained_model():
    return tf.keras.models.load_model("improved_model.h5", compile=False)


# Преобразование категориальных признаков в числовые
def preprocess_categorical_features(transmission, fuelType):
    global label_encoders
    transmission_encoded = label_encoders['transmission'].transform([transmission])[0]
    fuelType_encoded = label_encoders['fuelType'].transform([fuelType])[0]
    return transmission_encoded, fuelType_encoded


# Функция тестирования модели на введенных данных
def test_model_on_input(model, input_data):
    year, transmission, mileage, fuelType, tax, mpg, engineSize = input_data
    transmission_encoded, fuelType_encoded = preprocess_categorical_features(transmission, fuelType)
    input_array = np.array([year, transmission_encoded, mileage, fuelType_encoded, tax, mpg, engineSize]).reshape(1, -1)
    prediction = model.predict(input_array)
    return int(prediction) / 1000


# Функция для построения графиков
def charts_page(data):
    st.title('Графики')

    # График: Средняя цена автомобиля выпущенного в определенный год
    total_prices_per_year = data.groupby('year')['price'].mean().reset_index()
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(x='year', y='price', data=total_prices_per_year, palette='rainbow')
    ax.set_xticklabels(ax.get_xticklabels())
    plt.title('Средняя цена автомобиля выпущенного в определенный год')
    st.pyplot(plt)

    # График: Количество автомобилей производителя
    plt.figure(figsize=(14, 6))
    ax = sns.countplot(x='Manufacturer', data=data, palette='rainbow')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.title('Количество автомобилей производителя')
    st.pyplot(plt)

    # График: Средняя цена автомобиля для каждого производителя
    total_prices_per_manufacturer = data.groupby('Manufacturer')['price'].mean().reset_index()
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(x='Manufacturer', y='price', data=total_prices_per_manufacturer, palette='rainbow')
    ax.set_xticklabels(ax.get_xticklabels())
    plt.title('Средняя цена автомобиля для каждого производителя')
    st.pyplot(plt)


# Функция для отображения описания программы
def description_page():
    st.title('Описание программы')
    st.write('Данная программа предназначена для тестирования модели машинного обучения, которая предсказывает цену '
             'автомобиля на основе его характеристик. Вы можете ввести данные о годе выпуска, типе трансмиссии, пробеге,'
             ' типе топлива, налоге, расходе топлива и объеме двигателя, а затем нажать кнопку "Тестировать модель",'
             ' чтобы увидеть предсказанную цену автомобиля. Также доступны графики, отображающие среднюю цену автомобиля'
             ' по годам, количество автомобилей по производителям и среднюю цену автомобиля для каждого производителя. '
             'Также есть страница с выводами.')
    st.write('Для перехода между страницами используйте боковую панель навигации.')


# Функция для отображения выводов по графикам
def conclusions_page():
    st.title('Выводы по графикам')
    st.markdown("**График средней цены автомобиля выпущенного в определённый год**: имеет левостороннее распределение,"
                " однако, при этом, есть выброс в виде цены автомобилей за 1970 год.")
    st.markdown("**График количества автомобилей производителя**: имеет неоднородное распределение,"
                " наибольшее количество автомобилей имеет производитель - ford, наименьшее - hyundi.")
    st.markdown("**График средней цены автомобиля производителя**: имеет неоднородное распределение,"
                " производители Audi, BMW, и merc имеют примерно одинаковую и самую высокую среднюю стоимость своих (от"
                " 23 до 25 тыс. евро) автомобилей. Производитель vauxhall имеет самую низкую среднюю стоимость (11 тыс. евро)"
                " автомобилей, остальные же производители расположились в среднем ценовом сегменте (от 13 до 16 тыс. евро)"
                " и также имеют примерно одинаковую стоимость.")


# Основная функция
def main():
    data = load_data()
    st.sidebar.title("Навигация")
    page = st.sidebar.selectbox("Выбор страницы", ["Описание", "Модель", "Графики", "Выводы"])
    if page == "Описание":
        description_page()
    elif page == "Модель":
        st.title('Тестирование модели для предсказания автомобилей')
        # Загрузка модели
        model = load_trained_model()

        st.write('Модель загружена. Введите данные для тестирования.')

        # Виджеты для ввода данных
        year = st.number_input('Год выпуска:', min_value=1900, max_value=2024, step=1)
        transmission = st.selectbox('Тип трансмиссии:', ['Manual', 'Semi-Auto', 'Automatic', 'Other'])
        mileage = st.number_input('Пробег (в милях):', min_value=0, step=100)
        fuelType = st.selectbox('Тип топлива:', ['Petrol', 'Diesel', 'Hybrid', 'Other', 'Electric'])
        tax = st.number_input('Налог (в фунтах):', min_value=0, step=1)
        mpg = st.number_input('Расход топлива (миль на галлон):', min_value=0.0, step=0.1)
        engineSize = st.number_input('Объем двигателя (в литрах):', min_value=0.0, step=0.1)

        # Кнопка для тестирования модели
        if st.button('Тестировать модель'):
            prediction = test_model_on_input(model, [year, transmission, mileage, fuelType, tax, mpg, engineSize])
            st.write('Предсказанная цена:', prediction)
    elif page == "Графики":
        charts_page(data)
    elif page == "Описание":
        description_page()
    elif page == "Выводы":
        conclusions_page()


if __name__ == "__main__":
    main()
