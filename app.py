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

