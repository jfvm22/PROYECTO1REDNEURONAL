import streamlit as st 
import numpy as np 
import pandas as pd
import plotly.offline as py 
#py.init_notebook_mode(connected=True) # this code, allow us to work with offline plotly version
import plotly.graph_objs as go # it's like "plt" of matplot
import plotly.tools as tls # It's useful to we get some tools of plotly
#import matplotlib.pyplot as plt
#import seaborn as sns
#librerias de modelado
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold,cross_val_score
#librerias de redes neuronales
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score, classification_report

#definir funciones
def get_eda(dataset):
    # Distribución de Creditos por Tipo de Casa
    trace0 = go.Bar(
        x=dataset[dataset["risk"] == 'good']["housing"].value_counts().index.values,
        y=dataset[dataset["risk"] == 'good']["housing"].value_counts().values,
        name='Good credit'
    )

    trace1 = go.Bar(
        x=dataset[dataset["risk"] == 'bad']["housing"].value_counts().index.values,
        y=dataset[dataset["risk"] == 'bad']["housing"].value_counts().values,
        name="Bad Credit"
    )

    data = [trace0, trace1]

    layout = go.Layout(
        title='Housing Distribution'
    )

    fig = go.Figure(data=data, layout=layout)

    st.plotly_chart(fig)

    # Distribución de Creditos por Genero
    trace0 = go.Bar(
        x=dataset[dataset["risk"] == 'good']["sex"].value_counts().index.values,
        y=dataset[dataset["risk"] == 'good']["sex"].value_counts().values,
        name='Good credit'
    )

    trace1 = go.Bar(
        x=dataset[dataset["risk"] == 'bad']["sex"].value_counts().index.values,
        y=dataset[dataset["risk"] == 'bad']["sex"].value_counts().values,
        name="Bad Credit"
    )

    data = [trace0, trace1]

    layout = go.Layout(
        title='Gender Distribution'
    )

    fig = go.Figure(data=data, layout=layout)

    st.plotly_chart(fig)

    # Distribución de Creditos por Job
    trace0 = go.Bar(
        x=dataset[dataset["risk"] == 'good']["job"].value_counts().index.values,
        y=dataset[dataset["risk"] == 'good']["job"].value_counts().values,
        name='Good credit'
    )

    trace1 = go.Bar(
        x=dataset[dataset["risk"] == 'bad']["job"].value_counts().index.values,
        y=dataset[dataset["risk"] == 'bad']["job"].value_counts().values,
        name="Bad Credit"
    )

    data = [trace0, trace1]

    layout = go.Layout(
        title='Job Distribution'
    )

    fig = go.Figure(data=data, layout=layout)

    st.plotly_chart(fig)

    # Distribución de Creditos por Cuentas de ahorro
    trace0 = go.Bar(
        x=dataset[dataset["risk"] == 'good']["saving_accounts"].value_counts().index.values,
        y=dataset[dataset["risk"] == 'good']["saving_accounts"].value_counts().values,
        name='Good credit'
    )

    trace1 = go.Bar(
        x=dataset[dataset["risk"] == 'bad']["saving_accounts"].value_counts().index.values,
        y=dataset[dataset["risk"] == 'bad']["saving_accounts"].value_counts().values,
        name="Bad Credit"
    )

    data = [trace0, trace1]

    layout = go.Layout(
        title='Saving Accounts Distribution'
    )

    fig = go.Figure(data=data, layout=layout)

    st.plotly_chart(fig)


    # Distribución de Creditos por Cuentas de Crédiro
    trace0 = go.Bar(
        x=dataset[dataset["risk"] == 'good']["checking account"].value_counts().index.values,
        y=dataset[dataset["risk"] == 'good']["checking account"].value_counts().values,
        name='Good credit'
    )

    trace1 = go.Bar(
        x=dataset[dataset["risk"] == 'bad']["checking account"].value_counts().index.values,
        y=dataset[dataset["risk"] == 'bad']["checking account"].value_counts().values,
        name="Bad Credit"
    )

    data = [trace0, trace1]

    layout = go.Layout(
        title='Checking Account Distribution'
    )

    fig = go.Figure(data=data, layout=layout)

    st.plotly_chart(fig)

    # Distribución de Creditos por duración
    trace0 = go.Bar(
        x=dataset[dataset["risk"] == 'good']["duration"].value_counts().index.values,
        y=dataset[dataset["risk"] == 'good']["duration"].value_counts().values,
        name='Good credit'
    )

    trace1 = go.Bar(
        x=dataset[dataset["risk"] == 'bad']["duration"].value_counts().index.values,
        y=dataset[dataset["risk"] == 'bad']["duration"].value_counts().values,
        name="Bad Credit"
    )

    data = [trace0, trace1]

    layout = go.Layout(
        title='Duration Distribution'
    )

    fig = go.Figure(data=data, layout=layout)   
    st.plotly_chart(fig)
    
    # Distribución de Creditos por Propósito
    trace0 = go.Bar(
        x=dataset[dataset["risk"] == 'good']["purpose"].value_counts().index.values,
        y=dataset[dataset["risk"] == 'good']["purpose"].value_counts().values,
        name='Good credit'
    )

    trace1 = go.Bar(
        x=dataset[dataset["risk"] == 'bad']["purpose"].value_counts().index.values,
        y=dataset[dataset["risk"] == 'bad']["purpose"].value_counts().values,
        name="Bad Credit"
    )

    data = [trace0, trace1]

    layout = go.Layout(
        title='Purpose Distribution'
    )

    fig = go.Figure(data=data, layout=layout)

    st.plotly_chart(fig)

#crear una funcion para aplicar dummies 
def one_hot_encoder(df, nan_as_category = False):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category, drop_first=True)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def feature_engineering(dataset):
    #crear categorias por edad
    interval = (18,25,35,60, 120)
    cats = ['Student', 'Young', 'Adult', 'Senior']
    dataset['Age_cat'] = pd.cut(dataset.age, interval, labels=cats)

    #reemplazar los valores nan
    dataset['saving_accounts'] = dataset['saving_accounts'].fillna('no_inf')
    dataset['checking account'] = dataset['checking account'].fillna('no_inf')

    #convertir a dummies las variables categoricas
    dataset = dataset.merge(pd.get_dummies(dataset.purpose, drop_first=True, prefix='purpose'), left_index=True, right_index=True)


    #aplicar dummies
    dataset = dataset.merge(pd.get_dummies(dataset.sex, prefix='Sex'), left_index=True, right_index=True)
    dataset = dataset.merge(pd.get_dummies(dataset.housing, drop_first=True, prefix='Housing'), left_index=True, right_index=True)
    dataset = dataset.merge(pd.get_dummies(dataset["saving_accounts"], drop_first=True, prefix='Savings'), left_index=True, right_index=True)
    dataset = dataset.merge(pd.get_dummies(dataset.risk, prefix='Risk'), left_index=True, right_index=True)
    dataset = dataset.merge(pd.get_dummies(dataset["checking account"], drop_first=True, prefix='Check'), left_index=True, right_index=True)
    dataset = dataset.merge(pd.get_dummies(dataset["Age_cat"], prefix='Age_cat'), left_index=True, right_index=True)

    #eliminar las variables anteriores
    del dataset["Unnamed: 0"]
    del dataset["saving_accounts"]
    del dataset["checking account"]
    del dataset["purpose"]
    del dataset["sex"]
    del dataset["housing"]
    del dataset["Age_cat"]
    del dataset["risk"]
    del dataset["Risk_good"]
    return dataset    
    
def modelling(dataset):
    #aplicamos una funcion logaritmo para ajustar los valores
    dataset['credit amount'] = np.log(dataset['credit amount'])

    # separamos la variable objetivo (y) de las variables predictoras (X)
    X = dataset.drop('Risk_bad', axis=1).values
    y = dataset['Risk_bad'].values
    
    # Spliting X and y into train and test version
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)

    # Prepapar los modelos
    #arreglo para almacenar los modelos
    models = []
    #agregamos cada uno de los métodos
    models.append(('LGR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('SVM', SVC(gamma='auto')))
    #models.append(('XGBM', XGBClassifier()))
    #models.append(('LGBM', LGBMClassifier()))

    # Entrenamos y validamos cada modelo
    # arreglo para analizar los resultados
    results = []
    names = []
    scoring = 'recall'

    for name, model in models:
            kfold = KFold(n_splits=10, random_state=None)
            cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
    #crear dataset de resultados
    resultsDF = pd.DataFrame (results, columns = ['V0','V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9'])
    
    resultsBox = pd.DataFrame (results, columns = ['V0','V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9'])
    resultsDF['Model'] = names
    #graficar resultados
    fig = go.Figure()
    for i in range(7):
        fig.add_trace(go.Box(y=resultsBox[i:i+1].to_numpy()[0], name=names[i] ))
    st.plotly_chart(fig)
    return X_train, X_test, y_train, y_test

#definimos el modelo
def nn_model(learning_rate, y_train_categorical):
    NN_model = Sequential()

    # The Input Layer :
    NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

    # The Hidden Layers :
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

    # The Output Layer :
    NN_model.add(Dense(np.unique(y_train_categorical).shape[0] , kernel_initializer='normal',activation='sigmoid'))

    # Compile the network :
    optimizer = Adam(learning_rate=1e-5)
    NN_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
    NN_model.summary()
    return NN_model

def TrainningNN(X_train, X_test, y_train, y_test):
    y_test_categorical = to_categorical(y_test, num_classes=2, dtype='float32')
    y_train_categorical = to_categorical( y_train, num_classes=2, dtype='float32')

    #convertir tensor a numpy
    #X_train = np.array(X_train)
    X_train = np.asarray(X_train).astype(np.float32)
    
    #semilla para aleatorios
    np.random.seed(7)

    NN_model = nn_model(1e-4, y_train_categorical)
    nb_epochs = 100
    NN_model.fit(X_train, y_train_categorical, epochs=nb_epochs, batch_size=50)

    #convertir tensor en numpy array
    #X_test = np.array(X_test)
    X_test = np.asarray(X_test).astype(np.float32)

    NNpredictions = NN_model.predict(X_test)
    

    NN_prediction = list()
    for i in range(len(NNpredictions)):
        NN_prediction.append(np.argmax(NNpredictions[i]))

    # Validation of the results
    st.write("Accuracy:")
    st.write(accuracy_score(y_test, NN_prediction))
    
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, NN_prediction))
    cm = confusion_matrix(y_test, NN_prediction)
    heatmap = go.Heatmap(z=cm,
                     x=['Good', 'Bad'],
                     y=['Good', 'Bad'],
                     colorscale='Viridis')
    # Crear un objeto figura
    fig = go.Figure(data=[heatmap])
    # Utilizar st.plotly_chart para mostrar la figura en Streamlit
    st.plotly_chart(fig)

    #st.write("fbeta score:")
    #st.write(fbeta_score(y_test, NN_prediction, beta=2))
    #st.write("Classification Report:")
    #st.write(classification_report(y_test, NN_prediction))
    
    # Generate ROC curve values: fpr, tpr, thresholds
    fpr, tpr, thresholds = roc_curve(y_test, NNpredictions[:, 1])
    lr_auc = roc_auc_score(y_test, NNpredictions[:, 1])
    # Plot ROC curve
    fig = go.Figure()
    # Curva de habilidad nula
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='No Skill: ROC AUC=%.3f' % (0.5)))
    # Curva ROC del modelo
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='Logistic: ROC AUC=%.3f' % (lr_auc)))
    # Configura el diseño del gráfico
    fig.update_layout(xaxis_title='False Positive Rate',
                  yaxis_title='True Positive Rate',
                  title='ROC Curve',
                  showlegend=True)
    # Muestra la figura en Streamlit
    st.plotly_chart(fig)
    
    return NN_model

def predictionForm(modelNN):
    option_sex = ['male', 'female']
    option_job = [0,1,2,3]
    option_housing = ['own', 'rent']
    option_saving = ['moderate', 'quite rich', 'rich','little']
    option_checking = ['moderate', 'rich','little']
    option_duration = [6,7,8,9,10,11,12,15,18,24,27,30,36,42,45,48,60]
    option_purpose = ['radio/TV', 'education', 'furniture/equipment', 'car',
       'domestic appliances', 'repairs', 'vacation/others']

    age = st.text_input('Edad')
    select_sex = st.selectbox('Género', option_sex)
    select_job = st.selectbox('Trabajo', option_job)
    select_housing = st.selectbox('Vivienda', option_housing)
    select_saving = st.selectbox('Cuenta de Ahorro', option_saving)
    select_checking = st.selectbox('Cuenta de Crédito', option_checking)
    credit_amount = st.text_input('Monto del Crédito')
    select_duration = st.selectbox('Duración', option_duration)
    select_purpose = st.selectbox('Propósito', option_purpose)

    if st.button('Enviar'):
        prediction(age,select_sex,select_job,select_housing,select_saving,select_checking,credit_amount, select_duration, select_purpose)    
    
def prediction(age,sex,job,housing,saving,checking,amount,duration,purpose):
    # Crear dataframe
    data = {
        "age": [age], 
        "sex": [sex], 
        "job": [job],
        "housing": [housing], 
        "saving_accounts": [saving],
        "checking_account": [checking], 
        "credit_amount": [amount], 
        "duration": [duration], 
        "purpose": [purpose]
    }

    dataset = pd.DataFrame(data)
    datasetNew = dataset.copy()


    # ============ featuring para el dataset a predecir  ============
    
    #asignar categorias por edad
    if int(age) >= 18 and int(age) <= 25:
        Age_cat='Student'
    elif int(age) >= 28 and int(age) <= 35:
        Age_cat='Young'
    elif int(age) >= 35 and int(age) <= 60:
        Age_cat='Adult'
    else:
        Age_cat='Senior'
    datasetNew['Age_cat'] = Age_cat
    
  #asignar el proposito del crédito
    purpose_car=0
    purpose_domestic=0
    purpose_education=0
    purpose_furniture=0
    purpose_radio=0
    purpose_repairs=0
    purpose_vacation=0
    if purpose ==  "radio/TV":
        purpose_radio = 1
    elif purpose ==  "education":
        purpose_education = 1
    elif purpose ==  "furniture/equipment":
        purpose_furniture = 1
    elif purpose ==  "car":
        purpose_car = 1
    elif purpose ==  "vacation/others":
        purpose_vacation = 1
    elif purpose ==  "domestic appliances":
        purpose_domestic = 1
    else:
        purpose_repairs = 1
    dataPurpose = {
        "purpose_car":[purpose_car],
        "purpose_domestic appliances":[purpose_domestic], 
        "purpose_education":[purpose_education],
        "purpose_furniture/equipment":[purpose_furniture], 
        "purpose_radio/TV":[purpose_radio], 
        "purpose_repairs":[purpose_repairs],
        "purpose_vacation/others":[purpose_vacation]
    }
    dataPurpose = pd.DataFrame(dataPurpose)
    datasetNew = pd.concat([datasetNew, dataPurpose], axis=1)  
    
    #asignar genero
    Sex_female=0
    Sex_male=0
    if sex == "female":
        Sex_female=1
    else:
        Sex_male=1
    dataGender = {
        "Sex_female":[Sex_female],
        "Sex_male":[Sex_male]
    }    
    dataGender = pd.DataFrame(dataGender)
    datasetNew = pd.concat([datasetNew, dataGender], axis=1)
    
    #asignar vivienda
    Housing_own=0
    Housing_rent=0
    if housing == "own":
        Housing_own=1
    else:
        Housing_rent=1
    dataHousing = {
        "Housing_own":[Housing_own],
        "Housing_rent":[Housing_rent]
    }    
    dataHousing = pd.DataFrame(dataHousing)
    datasetNew = pd.concat([datasetNew, dataHousing], axis=1)

    #asignar Cuenta de Ahorros
    Savings_moderate=0
    Savings_no_inf=0
    Savings_quite=0 
    Savings_rich=0
    if saving == "moderate":
        Savings_moderate=1
    elif saving == "quite rich":
        Savings_quite=1
    elif saving == "rich":
        Savings_rich=1
    else:
        Savings_no_inf=1
    dataSaving = {
        "Savings_moderate":[Savings_moderate],
        "Savings_no_inf":[Savings_no_inf],
        "Savings_quite rich":[Savings_quite],
        "Savings_rich":[Savings_rich]
    }    
    dataSaving = pd.DataFrame(dataSaving)
    datasetNew = pd.concat([datasetNew, dataSaving], axis=1)   
  
    #asignar Cuenta de Credito 
    Check_moderate=0
    Check_no_inf=0 
    Check_rich=0
    if checking == "moderate":
        Check_moderate=1
    elif checking == "rich":
        Check_rich=1
    else:
        Check_no_inf=1
    dataChecking = {
        "Check_moderate":[Check_moderate],
        "Check_no_inf":[Check_no_inf],
        "Check_rich":[Check_rich]
    }    
    dataChecking = pd.DataFrame(dataChecking)
    datasetNew = pd.concat([datasetNew, dataChecking], axis=1)   

    #asignar categoria edad 
    Age_cat_Student=0
    Age_cat_Young=0
    Age_cat_Adult=0
    Age_cat_Senior=0
    if Age_cat == "Student":
        Age_cat_Student=1
    elif Age_cat == "Young":
        Age_cat_Young=1
    elif Age_cat == "Adult":
        Age_cat_Adult ==1
    else:
        Age_cat_Senior=1
    dataCatAge = {
        "Age_cat_Student":[Age_cat_Student], 
        "Age_cat_Young":[Age_cat_Young],
        "Age_cat_Adult":[Age_cat_Adult], 
        "Age_cat_Senior":[Age_cat_Senior]
    }    
    dataCatAge = pd.DataFrame(dataCatAge)
    datasetNew = pd.concat([datasetNew, dataCatAge], axis=1) 
    #eliminar columnas 
    del datasetNew["saving_accounts"]
    del datasetNew["checking_account"]
    del datasetNew["purpose"]
    del datasetNew["sex"]
    del datasetNew["housing"]
    del datasetNew["Age_cat"]
    #aplicamos una funcion logaritmo para ajustar los valores
    datasetNew['credit_amount'] = np.log(int(datasetNew['credit_amount']))
    #convertir variables a numpy
    X_values = datasetNew.values
    X_predict = X_values.astype(float)

    # Mostrar prediccion
    st.write("### Predicción del Crédito")
    # Mostrar dataframe
    st.write(dataset)
    # Calcular Prediccion
    valuePredict = modelNN.predict(X_predict)
    st.write(valuePredict)
    
    if valuePredict[0][0] > valuePredict[0][1]:
        st.write("#### Crédito Malo")
    else:
        st.write("#### Crédito Bueno")


#writing simple text 

st.title("Credit Card App")

    
# ============ Aplicación Principal  ============
        
# Definir las opciones de página
pages = ["Cargar Datos", "Explorar Datos", "Feature Engineering", "Modelado", "Neural Network", "Prediccion"]


# Mostrar un menú para seleccionar la página
selected_page = st.sidebar.multiselect("Seleccione una página", pages)

# Condicionales para mostrar la página seleccionada
if "Cargar Datos" in selected_page:
    st.write("""
    ## Cargar Datos""")
    # Cargar archivo CSV usando file uploader
    uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])
    # Si el archivo se cargó correctamente
    if uploaded_file is not None:
    # Leer archivo CSV usando Pandas
        dataset = pd.read_csv(uploaded_file)
    # Mostrar datos en una tabla
        st.write(dataset)
        
if "Explorar Datos" in selected_page:
    st.write("""
    ## Explore Data
    Distributions""")
    if uploaded_file is not None:
        get_eda(dataset)
        
if "Feature Engineering" in selected_page:
    st.write("""
    ## Feature Engineering
    New datset""")
    if uploaded_file is not None:
        dataset = feature_engineering(dataset)
        st.write(dataset)

if "Modelado" in selected_page:
    st.write("""
    ## Entrenamiento con diferentes modelos
    Resultados""")
    if uploaded_file is not None:
        X_train, X_test, y_train, y_test = modelling(dataset)
        
if "Neural Network" in selected_page:
    st.write("""
    ## Neural Network
    Resultados""")
    if uploaded_file is not None:
        st.write(tf.__version__)
        modelNN = TrainningNN(X_train, X_test, y_train, y_test)
        
if "Prediccion" in selected_page:
    st.write("""
    ## Predicción de un Crédito
    Capture los datos""")
    if uploaded_file is not None:
        predictionForm(modelNN) 
