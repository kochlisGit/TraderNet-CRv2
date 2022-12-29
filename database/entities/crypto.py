class Crypto:
    def __init__(self, symbol: str, name: str, start_year: int):
        self._symbol = symbol
        self._name = name
        self._start_year = start_year

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def name(self) -> str:
        return self._name

    @property
    def start_year(self) -> int:
        return self._start_year
