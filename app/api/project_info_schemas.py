from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReleaseFeature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: str = Field(
        pattern=r"^\d+\.\d+-\d{2,}$",
        examples=["1.2-04"],
    )
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    category: str = Field(min_length=1, examples=["developer"])
    change_type: Literal["new", "improved", "fixed", "security", "deprecated"]


class ReleaseInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(pattern=r"^\d+\.\d+\.\d+$", examples=["1.2.0"])
    title: str = Field(min_length=1)
    release_date: Optional[date] = None
    status: Literal["current", "previous", "planned", "deprecated"]
    summary: str = Field(min_length=1)
    features: list[ReleaseFeature] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_feature_numbers(self) -> ReleaseInfo:
        expected_prefix = ".".join(self.version.split(".")[:2]) + "-"
        feature_numbers = [f.number for f in self.features]

        if len(feature_numbers) != len(set(feature_numbers)):
            raise ValueError(f"duplicate feature number in release {self.version}")

        invalid = [n for n in feature_numbers if not n.startswith(expected_prefix)]
        if invalid:
            raise ValueError(
                f"feature numbers for release {self.version} must start with {expected_prefix}: {invalid}"
            )
        return self


class ProjectMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    name_en: str = Field(min_length=1)
    description: str = Field(min_length=1)
    current_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    api_version: str = Field(pattern=r"^v\d+$", examples=["v1"])


class ProjectInfoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: ProjectMetadata
    releases: list[ReleaseInfo]

    @model_validator(mode="after")
    def _validate_current_release(self) -> ProjectInfoResponse:
        versions = [r.version for r in self.releases]
        if len(versions) != len(set(versions)):
            raise ValueError("نسخه‌های انتشار باید یکتا باشند")

        matching = [r for r in self.releases if r.version == self.project.current_version]
        if len(matching) != 1:
            raise ValueError("current_version باید دقیقاً با یک release مطابقت داشته باشد")
        if matching[0].status != "current":
            raise ValueError("release منطبق با current_version باید status=current داشته باشد")
        if sum(1 for r in self.releases if r.status == "current") != 1:
            raise ValueError("دقیقاً یک release باید status=current داشته باشد")
        return self


class CurrentReleaseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: ProjectMetadata
    release: ReleaseInfo