class Termination:
    """
    Base class for termination conditions.
    """
    def __init__(self, **kwargs):
        self.maxparameter = {}
        for k, v in kwargs.items():
            self.maxparameter[k] = v

    def get(self, key: str) -> float | None:
        return self.maxparameter.get(key, None)
    
    def set(self, key: str, value: float) -> None:
        self.maxparameter[key] = value

    def is_terminated(self, **kwargs) -> bool:
        for k, v in kwargs.items():
            if k in self.maxparameter and v >= self.maxparameter[k]:
                return True
        return False