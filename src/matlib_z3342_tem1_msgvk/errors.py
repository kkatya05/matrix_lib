class ShapeMismatchError(ValueError):
    """Raised when matrix or vector dimensions are incompatible."""
    pass


class SingularMatrixError(ValueError):
    """Raised when a linear system cannot be solved due to singularity."""
    pass