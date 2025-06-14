from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pyngrok import conf, ngrok
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Cargar modelo
pipeline = joblib.load("modelo_credito.pkl")

# Crear app FastAPI
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar Jinja2
templates = Jinja2Templates(directory="templates")

# Ruta de inicio (renderiza index.html)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.get("/ingresodata", response_class=HTMLResponse)
async def ingreso_data(request: Request):
 return templates.TemplateResponse("ingresodata.html", {"request": request})


# Definición del esquema de entrada
class FormData(BaseModel):
    cedula: str              
    edad: int
    genero: str
    estado_civil: str
    nivel_educativo: str
    ciudad: str
    ingresos_mensuales: float
    gastos_mensuales: float
    tiene_empleo: str
    tipo_empleo: str
    score_crediticio: float = None
    monto_solicitado: float
    plazo_meses: int
    tipo_credito: str

# Endpoint para predicción
@app.post("/predecir")
def predecir(data: FormData):
    df = pd.DataFrame([{
        "edad": data.edad,
        "ingresos_mensuales": data.ingresos_mensuales,
        "gastos_mensuales": data.gastos_mensuales,
        "score_crediticio": data.score_crediticio or 0,
        "monto_solicitado": data.monto_solicitado,
        "plazo_meses": data.plazo_meses,
        "genero": data.genero,
        "estado_civil": data.estado_civil,
        "nivel_educativo": data.nivel_educativo,
        "ciudad": data.ciudad,
        "tiene_empleo": data.tiene_empleo,
        "tipo_empleo": data.tipo_empleo,
        "tipo_credito": data.tipo_credito,
    }])

    pred = pipeline.predict(df)[0]
    return {"aprobado": bool(pred)}

if __name__ == "__main__":
    try:
        conf.get_default().config_path = "ngrok.yml"
        public_url = ngrok.connect(8000)
        print("⚡ API pública en:", public_url)
    except Exception as e:
        print("Ngrok no disponible:", e)

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
