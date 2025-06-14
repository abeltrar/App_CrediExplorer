from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from pyngrok import conf, ngrok
import joblib
import pandas as pd
import numpy as np

# Cargar modelo y columnas esperadas
modelo = joblib.load("modelo_credito.pkl")
columnas_modelo = joblib.load("columnas_modelo.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ingresodata", response_class=HTMLResponse)
async def ingreso_data(request: Request):
    return templates.TemplateResponse("ingresodata.html", {"request": request})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

class FormData(BaseModel):
    cedula: int
    edad: int
    genero: str
    estado_civil: str
    nivel_educativo: str
    ciudad: str
    ingresos_mensuales: float
    gastos_mensuales: float
    tiene_empleo: bool
    tipo_empleo: str
    score_crediticio: float
    monto_solicitado: float
    plazo_meses: int
    tipo_credito: str

# Endpoint de predicción
@app.post("/predecir")
def predecir(data: FormData):
    sueldo_min, sueldo_max = 0, 35000000
    gasto_min, gasto_max = 0, 25000000
    solicitado_min, solicitado_max = 500000, 100000000

    df_input = pd.DataFrame([{
        "edad": data.edad,
        "score_crediticio": data.score_crediticio or 0,
        "plazo_meses": data.plazo_meses,
        "Sueldo": (data.ingresos_mensuales - sueldo_min) / (sueldo_max - sueldo_min),
        "gasto": (data.gastos_mensuales - gasto_min) / (gasto_max - gasto_min),
        "solicitado": (data.monto_solicitado - solicitado_min) / (solicitado_max - solicitado_min),
        f"estado_civil_{data.estado_civil}": 1,
        f"genero_{data.genero}": 1,
        f"nivel_educativo_{data.nivel_educativo}": 1,
        f"ciudad_{data.ciudad}": 1,
        f"tipo_empleo_{data.tipo_empleo}": 1,
        f"tiene_empleo_{data.tiene_empleo}": 1,
        f"tipo_credito_{data.tipo_credito}": 1,
        "banco_Banco1": 0,
        "banco_Banco2": 0
    }])

    # Asegurar todas las columnas requeridas por el modelo estén presentes
    for col in columnas_modelo:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[columnas_modelo]

    pred = modelo.predict(df_input)[0]
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
