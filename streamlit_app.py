from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sb

st.title('🤖 Machine Learning App')
st.info('This is app predicts the annual salary of a person whether is is more than 50K dollars or less !')

with st.expander("Data"):
    st.write('**Raw Data**')
    data = pd.read_csv("census.csv")
    data

with st.expander("Data Visualization and analysis"):
    plt.figure(figsize=[25, 80])
    for i in range(data.shape[1] - 1):
        plt.subplot(data.shape[1] - 1, 1, i+1)
        richPeople_data = data[data['income'] == ">50K"].copy()
        sb.countplot(data=richPeople_data, x=data.columns[i])
        plt.xlabel(f"{data.columns[i]}"); plt.ylabel("count")
        if(i == 0):
            plt.title("People earn more than 50K dollars")
        if(data[data.columns[i]].value_counts().count() > 16):
            plt.xticks(rotation=45)

    # Display the plot in Streamlit
    st.pyplot(plt)  # Use Streamlit's function to render the plot
    st.write("""### These graphs show that
1- People who are older than 30 years old and younger than 50 years old are the most profitable for more than 50 thousand annually

2- People who have his private buisness are the most profitable for more than 50 thousand annually

3- People whose education level is 'Bachelors' are the most profitable for more than 50 thousand annually, followed by those with 'HS-grade', then those with 'Some-college'

4- People whose education num is '13' are the most profitable for more than 50 thousand annually, followed by those with '9', then those with '10'

5- People who is married are the most profitable for more than 50 thousand annually

6- People with occupation 'Exec-managerial' and 'Prof-Speciality' are the most profitable for more than 50 thousand annually

7- People with race 'white' are the most profitable for more than 50 thousand annually

8- Males are the most profitable for more than 50 thousand annually

9- People with 'capital-gain' 0 are the most profitable for more than 50 thousand annually

10- People with 'capital-loss' 0 are the most profitable for more than 50 thousand annually

11- People who work 40 hours per week are the most profitable for more than 50 thousand annually

12- People whose native-country is 'united-states' are the most profitable for more than 50 thousand annually""")
    
with st.sidebar:
    st.header('Input Features')
    age = st.slider('age (year)', 1, 90)
    workclass = st.selectbox('workclass', data['workclass'].unique())
    education_level = st.selectbox('education_level', data['education_level'].unique())
    education_num = st.slider('education-num (integer)', 1, 20)
    marital_status = st.selectbox('marital-status', data['marital-status'].unique())
    occupation = st.selectbox('occupation', data['occupation'].unique())
    relationship = st.selectbox('relationship', data['relationship'].unique())
    race = st.selectbox('race', data['race'].unique())
    sex = st.selectbox('sex', data['sex'].unique())
    capital_gain = st.slider('capital-gain (dolar)', 1.0, 50000.0)
    capital_loss = st.slider('capital-loss (dolar)', 1.0, 50000.0)
    hours_per_week = st.slider('hours-per-week (hours)', 1, 100)
    native_country = st.selectbox('native-country', data['native-country'].unique())

    # Create a DataFrame for the input features
    features = {'age': age,
            'workclass': workclass,
            'education_level': education_level,
            'education-num': education_num,
            'marital-status': marital_status,
            'occupation': occupation,
            'relationship': relationship,
            'race': race,
            'sex': sex,
            'capital-gain': capital_gain,
            'capital-loss': capital_loss,
            'hours-per-week': hours_per_week,
            'native-country': native_country
            }
    input_df = pd.DataFrame(features, index=[0])
    # input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input features'):
    input_df


    
