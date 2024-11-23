import streamlit as st 
import numpy as np 
import pandas as pd
import plotly.offline as py 
#py.init_notebook_mode(connected=True) # this code, allow us to work with offline plotly version
import plotly.graph_objs as go # it's like "plt" of matplot
import plotly.tools as tls # It's useful to we get some tools of plotly

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

if "Modelado" in selected_page:
    st.write("""
    ## Entrenamiento con diferentes modelos
    Resultados""")

        
if "Neural Network" in selected_page:
    st.write("""
    ## Neural Network
    Resultados""")

        
if "Prediccion" in selected_page:
    st.write("""
    ## Predicción de un Crédito
    Capture los datos""")
 
