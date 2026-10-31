"""Commitment record schema, mirroring the annotation format."""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Force(str, Enum):
    suggestion = "suggestion"
    intention = "intention"
    conditional_promise = "conditional_promise"
    firm_commitment = "firm_commitment"


class Status(str, Enum):
    proposed = "proposed"
    active = "active"
    revised = "revised"
    completed = "completed"
    cancelled = "cancelled"
    uncertain = "uncertain"


class Evidence(BaseModel):
    source: str = "transcript"
    text: str


class Commitment(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    uid: Optional[str] = None
    commitment_id: str
    version: int = 1
    owner: Optional[str] = None
    action: str
    object: Optional[str] = None
    commitment_force: Force
    conditions_cnf: list[list[str]] = []
    condition: Optional[str] = None
    deadline: Optional[str] = None
    status: Status
    evidence: list[Evidence]
    revision_of: Optional[str] = None
    uncertainty: bool = False

    @field_validator("evidence", mode="before")
    @classmethod
    def coerce_evidence(cls, v):
        # Accept a bare string or a single span as well as a list of spans.
        if isinstance(v, (str, dict)):
            v = [v]
        return [{"text": e} if isinstance(e, str) else e for e in v]

    @field_validator("conditions_cnf", mode="before")
    @classmethod
    def coerce_cnf(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [[v]]
        return [[g] if isinstance(g, str) else g for g in v]

    @model_validator(mode="after")
    def fill_condition(self):
        # Flatten CNF into a string so it can be compared with gold `condition`.
        if self.condition is None and self.conditions_cnf:
            self.condition = " AND ".join(" OR ".join(g) for g in self.conditions_cnf)
        return self

    @property
    def evidence_text(self):
        return " ".join(e.text for e in self.evidence)

    @field_validator("commitment_force", "status", mode="before")
    @classmethod
    def normalize_enum(cls, v):
        if isinstance(v, str):
            return v.strip().lower().replace(" ", "_").replace("-", "_")
        return v
