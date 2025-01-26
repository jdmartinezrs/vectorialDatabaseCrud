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
filtro_fecha = st.sidebar.checkbox("Filtrar por fecha", value=True)

# Selectores condicionales
estadio = None
resultado = None
etapa = None
fecha = None

# Obtener opciones únicas para los filtros
if filtro_estadio:
    estadio = st.sidebar.selectbox("Selecciona un estadio", df_partidos["estadio"].unique())
if filtro_resultado:
    resultado = st.sidebar.selectbox("Selecciona un resultado", df_partidos["resultado"].unique())
if filtro_etapa:
    etapa = st.sidebar.selectbox("Selecciona una etapa", df_partidos["etapa"].unique())
if filtro_fecha:
    # Extraer fechas únicas y ordenarlas
    fechas_unicas = sorted(df_partidos["fecha"].unique())
    fecha = st.sidebar.selectbox("Selecciona una fecha", fechas_unicas)

# Filtrar los partidos según los filtros seleccionados
partidos_filtrados = df_partidos.copy()

if filtro_estadio and estadio:
    partidos_filtrados = partidos_filtrados[partidos_filtrados["estadio"] == estadio]
if filtro_resultado and resultado:
    partidos_filtrados = partidos_filtrados[partidos_filtrados["resultado"] == resultado]
if filtro_etapa and etapa:
    partidos_filtrados = partidos_filtrados[partidos_filtrados["etapa"] == etapa]
if filtro_fecha and fecha:
    partidos_filtrados = partidos_filtrados[partidos_filtrados["fecha"] == fecha]

# Mostrar los partidos filtrados
st.subheader("Partidos filtrados")
st.write(partidos_filtrados)

# 4. Preparar los datos para el gráfico
nodes = []  # Lista de nodos (equipos)
edges = []  # Lista de aristas (partidos)

# Diccionario de banderas (URLs de imágenes de banderas)
banderas = {
    "Argentina": "https://flagcdn.com/ar.svg",
    "Francia": "https://flagcdn.com/fr.svg",
    "Brasil": "https://flagcdn.com/br.svg",
    "Alemania": "https://flagcdn.com/de.svg",
    "España": "https://flagcdn.com/es.svg",
    "Portugal": "https://flagcdn.com/pt.svg",
    "México": "https://flagcdn.com/mx.svg",
    "Qatar": "https://flagcdn.com/qa.svg",
    "Ecuador": "https://flagcdn.com/ec.svg",
    "Senegal": "https://flagcdn.com/sn.svg",
    "Países Bajos": "https://flagcdn.com/nl.svg",
    "Inglaterra": "https://flagcdn.com/gb.svg",
    "Estados Unidos": "https://flagcdn.com/us.svg",
    "Gales": "https://flagcdn.com/gb-wls.svg",
    "Irán": "https://flagcdn.com/ir.svg",
    "Arabia Saudita": "https://flagcdn.com/sa.svg",
    "Polonia": "https://flagcdn.com/pl.svg",
    "Australia": "https://flagcdn.com/au.svg",
    "Dinamarca": "https://flagcdn.com/dk.svg",
    "Túnez": "https://flagcdn.com/tn.svg",
    "Costa Rica": "https://flagcdn.com/cr.svg",
    "Japón": "https://flagcdn.com/jp.svg",
    "Bélgica": "https://flagcdn.com/be.svg",
    "Canadá": "https://flagcdn.com/ca.svg",
    "Marruecos": "https://flagcdn.com/ma.svg",
    "Croacia": "https://flagcdn.com/hr.svg",
    "Suiza": "https://flagcdn.com/ch.svg",
    "Camerún": "https://flagcdn.com/cm.svg",
    "Serbia": "https://flagcdn.com/rs.svg",
    "Uruguay": "https://flagcdn.com/uy.svg",
    "Corea del Sur": "https://flagcdn.com/kr.svg",
    "Ghana": "https://flagcdn.com/gh.svg",
}

# Crear nodos únicos para los equipos
equipos_unicos = list(set(df_partidos["equipo_local"].tolist() + df_partidos["equipo_visitante"].tolist()))
for i, equipo in enumerate(equipos_unicos):
    # Obtener la URL de la bandera del equipo
    bandera_url = banderas.get(equipo, "https://flagcdn.com/xx.svg")  # Bandera por defecto si no se encuentra
    nodes.append({
        "id": i,
        "label": equipo,
        "title": equipo,
        "value": 1,  # Tamaño del nodo
        "shape": "circularImage",  # Usar imagen dentro de un círculo
        "image": bandera_url,  # URL de la bandera
        "size": 25,  # Tamaño de la imagen
        "borderWidth": 0,  # Eliminar el borde del círculo
        "color": {
            "border": "transparent",  # Hacer el borde transparente
            "background": "transparent"  # Hacer el fondo transparente
        }
    })

# Crear aristas para los partidos
for _, partido in partidos_filtrados.iterrows():
    src = equipos_unicos.index(partido["equipo_local"])
    dst = equipos_unicos.index(partido["equipo_visitante"])
    edges.append({
        "source": src,
        "target": dst,
        "label": partido["resultado"],  # Resultado como etiqueta de la arista
        "title": f"Estadio: {partido['estadio']}, Etapa: {partido['etapa']}",  # Información adicional
        "width": 2,  # Grosor de la línea
        "color": "#000000"  # Color de la línea
    })

# Crear el gráfico con Pyvis
net = Network(height="500px", width="100%", notebook=True)
for node in nodes:
    net.add_node(node["id"], label=node["label"], title=node["title"], shape=node["shape"], image=node["image"], size=node["size"], borderWidth=node["borderWidth"], color=node["color"])
for edge in edges:
    net.add_edge(edge["source"], edge["target"], label=edge["label"], title=edge["title"], width=edge["width"], color=edge["color"])

# Configuración adicional del gráfico
net.toggle_physics(True)  # Activar física para mejor distribución
net.set_edge_smooth('dynamic')  # Suavizar las aristas

# Guardar el gráfico en un archivo HTML
net.save_graph("grafo.html")

# Mostrar el gráfico en Streamlit
with open("grafo.html", "r", encoding="utf-8") as f:
    html_content = f.read()
components.html(html_content, height=600)