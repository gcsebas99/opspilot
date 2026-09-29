from typing import Any

from pydantic import BaseModel, Field


class Grade(BaseModel):
    """One grader's verdict on one trial."""

    name: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)


class Check(BaseModel):
    """One named assertion inside a grader -- kept individually so a failing
    grade says exactly *which* expectation broke, not just "trajectory: 0.83"."""

    name: str
    passed: bool
    detail: str = ""


def grade_from_checks(name: str, checks: list[Check]) -> Grade:
    passed_count = sum(c.passed for c in checks)
    return Grade(
        name=name,
        passed=passed_count == len(checks),
        score=passed_count / len(checks) if checks else 1.0,
        details={"checks": [c.model_dump() for c in checks]},
    )
