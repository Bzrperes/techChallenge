"""API da Etapa 3: uvicorn src.api:app --reload."""
from contextlib import asynccontextmanager
from pathlib import Path
import logging
from fastapi import Depends, FastAPI, HTTPException, Request
from src.config import MODEL_PATH
from src.predict import carregar_artefato, prever
from src.schemas import Cliente, Predicao, Saude


def obter_artefato(request: Request) -> dict:
    artefato = request.app.state.artefato
    if artefato is None:
        raise HTTPException(status_code=503, detail="Modelo indisponível. Consulte /health e os logs do servidor.")
    return artefato


def criar_app(model_path: str | Path = MODEL_PATH) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.artefato = None
        try:
            app.state.artefato = carregar_artefato(model_path)
        except Exception:
            logging.getLogger(__name__).exception("Não foi possível carregar o campeão da Etapa 2")
        yield
        app.state.artefato = None

    app = FastAPI(title="Churn — Etapa 3", version="2.0.0", lifespan=lifespan)

    @app.get("/health", response_model=Saude)
    def health(request: Request):
        return Saude(status="ok", modelo_carregado=request.app.state.artefato is not None)

    @app.post("/predict", response_model=Predicao)
    def predict(cliente: Cliente, artefato: dict = Depends(obter_artefato)):
        resultado = prever(artefato, [cliente.para_registro()])[0]
        return Predicao(**resultado, limiar=artefato["limiar"], modelo=artefato["modelo"])

    return app


app = criar_app()
