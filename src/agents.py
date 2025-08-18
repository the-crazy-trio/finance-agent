from typing import Any, Dict


class MacroAnalysisAgent:
    """
    Данный агент проводит макроанализ текущий обстановки в России.
    А именно, предоставляет:
    - Реальный ВВП
    - Реальный ВНП
    - Ключевую ставку
    - Доходность ОФЗ на разных периодах
    - Спрэды (КБД с Мосбиржи или ЦБ)
    - Оценка уровня инфляции
    - Индекс инфляционных ожиданий
    - PMI
    - PPI
    
    """
    def __init__(self) -> None:
        pass

    async def ainvoke(self, request: Dict[str, Any]) -> Dict[str, str]:
        pass


macro_analysis_agent = MacroAnalysisAgent()


def get_macro_analysis_agent() -> MacroAnalysisAgent:
    return macro_analysis_agent
