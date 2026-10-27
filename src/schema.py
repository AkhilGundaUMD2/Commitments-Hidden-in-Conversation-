"""Commitment record schema, mirroring the annotation format."""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator


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
    text: str


class Commitment(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    uid: Optional[str] = None
    commitment_id: str
    owner: Optional[str] = None
    action: str
    object: Optional[str] = None
    commitment_force: Force
    condition: Optional[str] = None
    deadline: Optional[str] = None
    status: Status
    evidence: Evidence
    revision_of: Optional[str] = None
    uncertainty: bool = False

    @field_validator("evidence", mode="before")
    @classmethod
    def coerce_evidence(cls, v):
        # Models sometimes return evidence as a bare string.
        if isinstance(v, str):
            return {"text": v}
        return v

    @field_validator("commitment_force", "status", mode="before")
    @classmethod
    def normalize_enum(cls, v):
        if isinstance(v, str):
            return v.strip().lower().replace(" ", "_").replace("-", "_")
        return v
