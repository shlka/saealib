"""
Termination module.

Termination class defines criteria to stop the optimization process.
"""
class Termination:
    """
    Termination class to determine when to stop the optimization process.

    Attributes
    ----------
    maxparameter : dict[str, float]
        Dictionary to store maximum parameters for termination.
    """
    def __init__(self, **kwargs):
        """
        Initialize Termination instance.

        Parameters
        ----------
        kwargs : dict[str, float]
            Maximum parameters for termination.
        """
        self.maxparameter = {}
        for k, v in kwargs.items():
            self.maxparameter[k] = v

    def get(self, key: str) -> float | None:
        """
        Getter for maximum parameter.

        Parameters
        ----------
        key : str
            The key to retrieve the maximum parameter for.
        
        Returns
        -------
        float | None
            The maximum parameter for the given key, or None if not found.
        """
        return self.maxparameter.get(key, None)
    
    def set(self, key: str, value: float) -> None:
        """
        Setter for maximum parameter.

        Parameters
        ----------
        key : str
            The key to set the maximum parameter for.
        value : float
            The maximum parameter value to set.
        """
        self.maxparameter[key] = value

    def is_terminated(self, **kwargs) -> bool:
        """
        Check if termination criteria are met.

        Parameters
        ----------
        kwargs : dict[str, float]
            Current parameters to check against maximum parameters.
        """
        for k, v in kwargs.items():
            if k in self.maxparameter and v >= self.maxparameter[k]:
                return True
        return False