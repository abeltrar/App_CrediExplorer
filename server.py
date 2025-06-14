from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pyngrok import conf, ngrok
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

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




@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("❌ Error de validación:", exc.errors())
    return JSONResponse(
    status_code=422,
    content={"detail": exc.errors()},
    )


# Definición del esquema de entrada
class FormData(BaseModel): 
    edad: int
    genero: str
    estado_civil: str
    nivel_educativo: str
    ciudad: str
    ingresos_mensuales: float
    gastos_mensuales: float
    tiene_empleo: str
    tipo_empleo: str
    score_crediticio: float 
    monto_solicitado: float
    plazo_meses: int
    tipo_credito: str
    banco: str  


# Endpoint para predicción

@app.post("/predecir")
def predecir(data: FormData):
    df = pd.DataFrame([{
        "edad": data.edad,
        "score_crediticio": data.score_crediticio or 0,
        "plazo_meses": data.plazo_meses,
        "sueldo en pesos": data.ingresos_mensuales,
        "gasto en pesos": data.gastos_mensuales,
        "monto_solicitado (Cop)": data.monto_solicitado,

        # One-hot encoding manual
        "estado_civil_Casado": 1 if data.estado_civil.lower() == "casado" else 0,
        "estado_civil_Divorciado": 1 if data.estado_civil.lower() == "divorciado" else 0,
        "estado_civil_Soltero": 1 if data.estado_civil.lower() == "soltero" else 0,

        "genero_F": 1 if data.genero.upper() == "F" else 0,
        "genero_M": 1 if data.genero.upper() == "M" else 0,

        "nivel_educativo_Primaria": 1 if data.nivel_educativo.lower() == "primaria" else 0,
        "nivel_educativo_Secundaria": 1 if data.nivel_educativo.lower() == "secundaria" else 0,
        "nivel_educativo_Universitaria": 1 if data.nivel_educativo.lower() == "universitaria" else 0,

        "ciudad_Barranquilla": 1 if data.ciudad.lower() == "barranquilla" else 0,
        "ciudad_Bogota": 1 if data.ciudad.lower() == "bogota" else 0,
        "ciudad_Cali": 1 if data.ciudad.lower() == "cali" else 0,
        "ciudad_Medellin": 1 if data.ciudad.lower() == "medellin" else 0,

        "tipo_empleo_Formal": 1 if data.tipo_empleo.lower() == "formal" else 0,
        "tipo_empleo_Independiente": 1 if data.tipo_empleo.lower() == "independiente" else 0,
        "tipo_empleo_Informal": 1 if data.tipo_empleo.lower() == "informal" else 0,

        "tiene_empleo_True": 1 if data.tiene_empleo.lower() == "si" else 0,
        "tiene_empleo_False": 1 if data.tiene_empleo.lower() == "no" else 0,

        "tipo_credito_Consumo": 1 if data.tipo_credito.lower() == "consumo" else 0,
        "tipo_credito_Hipotecario": 1 if data.tipo_credito.lower() == "hipotecario" else 0,
        "tipo_credito_Vehiculo": 1 if data.tipo_credito.lower() == "vehiculo" else 0,

        "banco_Banco A": 1 if hasattr(data, "banco") and data.banco == "Banco A" else 0,
        "banco_Banco B": 1 if hasattr(data, "banco") and data.banco == "Banco B" else 0,
        "banco_Banco C": 1 if hasattr(data, "banco") and data.banco == "Banco C" else 0,
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
