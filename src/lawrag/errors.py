"""Exception hierarchy (SDD 3.8.4). Each class carries the API error code it maps to."""


class LawRagError(Exception):
    code = "INTERNAL_ERROR"
    http_status = 500

    def __init__(self, message: str = "", **details):
        super().__init__(message)
        self.message = message
        self.details = details


class ConfigError(LawRagError):
    code = "CONFIG_ERROR"


class IngestionError(LawRagError):
    code = "INGESTION_ERROR"


class ParseError(IngestionError):
    code = "PARSE_ERROR"


class ValidationError(IngestionError):
    code = "VALIDATION_ERROR"


class GraphUnavailable(LawRagError):
    code = "NEO4J_UNAVAILABLE"
    http_status = 503


class NotFound(LawRagError):
    code = "NOT_FOUND"
    http_status = 404


class BadIdentifier(LawRagError):
    code = "MALFORMED_UID"
    http_status = 400


class Conflict(LawRagError):
    code = "CONFLICT"
    http_status = 409


class RetrievalError(LawRagError):
    code = "RETRIEVAL_ERROR"


class LLMError(LawRagError):
    code = "LLM_ERROR"
    http_status = 503


class LLMTimeout(LLMError):
    code = "LLM_TIMEOUT"
    http_status = 504


class LLMOutputError(LLMError):
    code = "LLM_OUTPUT_ERROR"


class BudgetExceeded(LLMError):
    code = "BUDGET_EXCEEDED"
    http_status = 429


class VerificationError(LawRagError):
    code = "VERIFICATION_ERROR"
