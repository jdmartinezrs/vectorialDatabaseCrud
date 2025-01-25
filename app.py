import streamlit as st
import pandas as pd
from pyvis.network import Network
import streamlit.components.v1 as components
import chromadb

# Configuración de la página
st.set_page_config(layout="wide")
st.title("Visualización de Partidos de Fútbol - FIFA 2022")

# 1. Conectar a ChromaDB y obtener la colección
client = chromadb.PersistentClient(path="./chroma_db")  # Usar persistencia local
collection = client.get_or_create_collection(name="partidos_fifa_2022")

# 2. Obtener todos los datos de la colección
partidos_chroma = collection.get(include=["documents", "metadatas"])

# 3. Convertir los datos a un DataFrame de Pandas
partidos = []
for i in range(len(partidos_chroma["ids"])):
    partido = {
        "id": partidos_chroma["ids"][i],
        "texto": partidos_chroma["documents"][i],
        **partidos_chroma["metadatas"][i]
    }
    partidos.append(partido)

df_partidos = pd.DataFrame(partidos)

# Filtros en la barra lateral
st.sidebar.header("Filtros")

# Checkbox para activar/desactivar cada filtro
filtro_estadio = st.sidebar.checkbox("Filtrar por estadio", value=True)
filtro_resultado = st.sidebar.checkbox("Filtrar por resultado", value=True)
filtro_etapa = st.sidebar.checkbox("Filtrar por etapa", value=True)

# Selectores condicionales
estadio = None
resultado = None
etapa = None

if filtro_estadio:
    estadio = st.sidebar.selectbox("Selecciona un estadio", df_partidos["estadio"].unique())
if filtro_resultado:
    resultado = st.sidebar.selectbox("Selecciona un resultado", df_partidos["resultado"].unique())
if filtro_etapa:
    etapa = st.sidebar.selectbox("Selecciona una etapa", df_partidos["etapa"].unique())

# Filtrar los partidos según los filtros seleccionados
partidos_filtrados = df_partidos.copy()

if filtro_estadio and estadio:
    partidos_filtrados = partidos_filtrados[partidos_filtrados["estadio"] == estadio]
if filtro_resultado and resultado:
    partidos_filtrados = partidos_filtrados[partidos_filtrados["resultado"] == resultado]
if filtro_etapa and etapa:
    partidos_filtrados = partidos_filtrados[partidos_filtrados["etapa"] == etapa]

# Mostrar los partidos filtrados
st.subheader("Partidos filtrados")
st.write(partidos_filtrados)

# Preparar los datos para el gráfico
nodes = []  # Lista de nodos (equipos)
edges = []  # Lista de aristas (partidos)

# Crear nodos únicos para los equipos
equipos_unicos = list(set(df_partidos["equipo_local"].tolist() + df_partidos["equipo_visitante"].tolist()))
for i, equipo in enumerate(equipos_unicos):
    nodes.append({
        "id": i,
        "label": equipo,
        "title": equipo,
        "value": 1,  # Tamaño del nodo
        "shape": "dot"  # Forma del nodo
    })

# Crear aristas para los partidos
for _, partido in partidos_filtrados.iterrows():
    src = equipos_unicos.index(partido["equipo_local"])
    dst = equipos_unicos.index(partido["equipo_visitante"])
    edges.append({
        "source": src,
        "target": dst,
        "label": partido["resultado"],  # Resultado como etiqueta de la arista
        "title": f"Estadio: {partido['estadio']}, Etapa: {partido['etapa']}"  # Información adicional
    })

# Crear el gráfico con Pyvis
net = Network(height="500px", width="100%", notebook=True)
for node in nodes:
    net.add_node(node["id"], label=node["label"], title=node["title"], shape=node["shape"])
for edge in edges:
    net.add_edge(edge["source"], edge["target"], label=edge["label"], title=edge["title"])

# Guardar el gráfico en un archivo HTML
net.save_graph("grafo.html")

# Mostrar el gráfico en Streamlit
with open("grafo.html", "r", encoding="utf-8") as f:
    html_content = f.read()
components.html(html_content, height=600)