"""Structured catalog models used by retrieval and ranking."""

from pydantic import BaseModel, ConfigDict, Field


class CatalogAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    url: str
    description: str = ""
    skills: list[str] = Field(default_factory=list)
    test_type_codes: list[str] = Field(default_factory=list)
    test_type_labels: list[str] = Field(default_factory=list)
    duration_minutes: int | None = None
    job_roles: list[str] = Field(default_factory=list)
    remote_testing_supported: bool | None = None
    remote_testing_detail: str = ""
    languages: list[str] = Field(default_factory=list)
    adaptive_irt: str = ""
    source_section: str = "individual_test_solutions"


class CatalogMeta(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_name: str
    embedding_dim: int
    num_items: int
