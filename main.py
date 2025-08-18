from typing import Dict
from fastapi import FastAPI
import uvicorn

from agents import get_macro_analysis_agent
from src.protocols import MacroAnalysisRequest, MacroAnalysisResponse

app = FastAPI()


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/macro_analysis")
async def macro_analysis(request: MacroAnalysisRequest) -> MacroAnalysisResponse:
    pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
