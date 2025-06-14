from fastapi import FastAPI
from pydantic import BaseModel
from pyngrok import ngrok
import joblib
import pandas as pd

pipeline = joblib.load("modelo_credito.pkl")

app = FastAPI()

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
    public_url = ngrok.connect(8000)
    print("⚡ API pública en:", public_url)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)