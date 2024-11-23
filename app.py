import streamlit as st 
import pandas as pd

#writing simple text 

st.title("Credit Card App")

# ============ Aplicación Principal  ============
        
# Definir las opciones de página
pages = ["Cargar Datos"]


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
        
