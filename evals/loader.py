from pathlib import Path
from typing import Literal

import yaml
from pydantic import ValidationError

from evals.models import EvalCase

GOLDEN_DIR = Path(__file__).parent / "golden"

Suite = Literal["golden", "smoke", "adversarial"]


def load_cases(suite: Suite = "golden", golden_dir: Path = GOLDEN_DIR) -> list[EvalCase]:
    """Load and validate every case in `golden_dir`.

    `suite="golden"` returns all of them; any other suite name is a tag
    filter over the same set (`smoke`/`adversarial` are conventions, not
    special-cased here -- a case tagged "smoke" is in the smoke suite,
    full stop). A YAML file that doesn't validate against EvalCase raises
    immediately, naming the file -- a broken case should fail the moment
    someone tries to use the dataset, not silently vanish from a suite.
    """
    cases: list[EvalCase] = []
    for path in sorted(golden_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        try:
            cases.append(EvalCase.model_validate(raw))
        except ValidationError as exc:
            raise ValueError(f"invalid eval case in {path}:\n{exc}") from exc

    ids = [case.id for case in cases]
    duplicates = {case_id for case_id in ids if ids.count(case_id) > 1}
    if duplicates:
        raise ValueError(f"duplicate eval case ids in {golden_dir}: {sorted(duplicates)}")

    if suite == "golden":
        return cases
    return [case for case in cases if suite in case.tags]
