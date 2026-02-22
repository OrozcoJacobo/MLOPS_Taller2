# MLOPS_TALLER2 – Desarrollo en Contenedores con uv y Docker Compose

## Descripción

Este proyecto implementa un entorno de desarrollo para Machine Learning utilizando Docker Compose y uv para la gestión de dependencias.

Se definen dos servicios dentro del mismo entorno:

- Un servicio de entrenamiento con JupyterLab.
- Un servicio de inferencia con FastAPI.

Los modelos entrenados en el entorno de Jupyter son almacenados en un volumen compartido, permitiendo que la API los consuma sin necesidad de reconstruir imágenes ni reinstalar dependencias.

---

## Arquitectura

El entorno está compuesto por dos servicios principales:

### 1. Servicio API (FastAPI)

- Expone un endpoint `/predict` para realizar inferencias.
- Carga dinámicamente los modelos definidos en `models/registry.json`.
- Registra logs de predicción en `logs/predictions.log`.
- Corre en el puerto `8000`.

### 2. Servicio JupyterLab

- Permite entrenar modelos dentro del contenedor.
- Utiliza uv para instalar dependencias desde `pyproject.toml`.
- Guarda los modelos entrenados en el volumen compartido.
- Corre en el puerto `8888`.

### Volúmenes Compartidos

- `models/` → compartido entre Jupyter y API.
- `logs/` → utilizado por la API para registrar predicciones.

### Flujo de Trabajo

1. Se entrena un modelo en Jupyter.
2. El modelo se guarda en `/workspace/models`.
3. Se actualiza `registry.json`.
4. La API puede consumir inmediatamente el nuevo modelo.

---

## Gestión de Dependencias con uv

El proyecto utiliza:

- `pyproject.toml`
- `uv.lock`

Las dependencias se instalan dentro de los contenedores utilizando:

```bash
uv sync
```

Esto garantiza entornos reproducibles y consistentes entre servicios.

---

## Cómo ejecutar el proyecto

### 1. Construir las imágenes

```bash
docker compose build
```

Si se requiere reconstrucción completa:

```bash
docker compose build --no-cache
```

### 2. Levantar los servicios

```bash
docker compose up
```

---

## Accesos

- API:  
  http://localhost:8000/docs

- JupyterLab:  
  http://localhost:8888

---

## Entrenamiento de nuevos modelos

1. Acceder a JupyterLab.
2. Ejecutar el notebook `entrenamiento_pinguinos.ipynb`.
3. Los modelos se guardarán automáticamente en la carpeta `models/`.
4. El archivo `registry.json` se actualiza con los modelos disponibles.
5. La API puede utilizarlos sin necesidad de reiniciar el contenedor.

---

## Estructura del Proyecto

```
MLOPS_TALLER2/
│
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.jupyter
├── pyproject.toml
├── uv.lock
│
├── penguin_predict/
│   └── main.py
│
├── notebooks/
│   └── entrenamiento_pinguinos.ipynb
│
├── models/
│   ├── *.joblib
│   └── registry.json
│
└── logs/
    └── predictions.log
```