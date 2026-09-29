"""Domain-specific errors used by the screening pipeline."""


class FraudScreeningError(Exception):
    """Base class for pipeline errors."""


class InputValidationError(FraudScreeningError):
    """Input transaction is missing, malformed, or semantically invalid."""


class HistoryValidationError(FraudScreeningError):
    """Historical context violates the causal-history rules."""


class PreprocessingContractError(FraudScreeningError):
    """Preprocessing input/output does not match the frozen representation."""


class ArtifactNotFoundError(FraudScreeningError):
    """A required official artifact cannot be found."""


class ArtifactFingerprintError(FraudScreeningError):
    """An artifact fingerprint does not match the expected identity."""


class ArtifactCompatibilityError(FraudScreeningError):
    """An artifact loads but is incompatible with the expected model state."""


class InferenceContractError(FraudScreeningError):
    """Inference output violates the stable screening interface."""
