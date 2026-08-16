"""Eligibility schema, and the flattening that makes it portable across providers.

Pydantic's ``model_json_schema()`` emits ``$defs`` + ``$ref`` for every nested
model and ``anyOf`` for every ``Optional`` field. Support for those varies by
provider, and OpenRouter passes the schema straight through to whichever one
serves the request — so the schema is authored normally and flattened on the way
out. A flat, closed schema is the subset everything accepts.

The 64-key ceiling was a Databricks Foundation Model API limit and is no longer
binding, but it is kept as a cheap guard: exceeding it means the taxonomy grew
past what a single structured-output call should carry, which is worth failing
on locally rather than discovering as an opaque 400 mid-run.

Two consequences shape the models below:

* **No ``Optional`` fields.** "unclear" is a ``Literal`` enum value on a required
  field, never ``None`` — an absent field and an uncertain answer are different
  facts, and only one of them survives the round trip.
* **No nested model reuse across fields**, since each reference is inlined.
"""

from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, Field

# Derived from data, not invented: an initial nine-category taxonomy put 23.6%
# of 1,027 criteria from 40 real trials into "other". Clustering that bucket
# showed it was dominated by categories the taxonomy simply lacked —
# reproductive 29%, consent 16%, procedure 9%, washout 8%. Those are now their
# own kinds. A criterion type that lands in "other" cannot be matched against a
# patient later, so the bucket is a direct cap on recall.
CriterionKind = Literal[
    # What the patient has
    "condition",
    "stage",
    "biomarker",
    "comorbidity",
    "infection",
    # Measurements
    "lab",
    "organ_function",
    "ecog",
    "age",
    "life_expectancy",
    "measurable_disease",
    # Treatment history and constraints
    "prior_therapy",
    "concurrent_therapy",
    "procedure",
    "washout",
    "allergy",
    # Administrative — rarely patient-matchable, but must not pollute the rest
    "reproductive",
    "consent",
    "compliance",
    "other",
]

_UNSUPPORTED_KEYWORDS = frozenset(
    {"anyOf", "oneOf", "allOf", "pattern", "prefixItems", "$defs", "default", "format"}
)

MAX_SCHEMA_KEYS = 64


class Criterion(BaseModel):
    """One eligibility predicate, lifted out of the prose."""

    criterion_id: str = Field(description="Stable id within this trial, e.g. 'inc-1' or 'exc-3'.")
    text: str = Field(description="The criterion restated as a single self-contained sentence.")
    kind: CriterionKind = Field(description="Which category of predicate this is.")
    is_exclusion: bool = Field(description="True if this criterion excludes rather than includes.")


class ExtractedEligibility(BaseModel):
    """The full structured result for one trial."""

    criteria: list[Criterion] = Field(description="Every inclusion and exclusion criterion found.")


class SchemaTooLargeError(ValueError):
    """The flattened schema exceeds the key budget the API enforces."""


def _resolve(node: Any, defs: dict[str, Any]) -> Any:
    """Recursively inline ``$ref`` targets and drop unsupported keywords."""
    if isinstance(node, list):
        return [_resolve(item, defs) for item in node]
    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        if name not in defs:
            msg = f"Cannot resolve $ref to {name!r}"
            raise ValueError(msg)
        return _resolve(copy.deepcopy(defs[name]), defs)

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _UNSUPPORTED_KEYWORDS:
            continue
        out[key] = _resolve(value, defs)

    # The API requires objects to be closed and to list their required keys.
    if out.get("type") == "object" and "properties" in out:
        out["additionalProperties"] = False
        out["required"] = list(out["properties"])
    return out


def _count_keys(node: Any) -> int:
    if isinstance(node, list):
        return sum(_count_keys(item) for item in node)
    if not isinstance(node, dict):
        return 0
    return len(node) + sum(_count_keys(value) for value in node.values())


def flatten_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Produce a JSON Schema the Foundation Model API will accept.

    Raises ``SchemaTooLargeError`` when the result exceeds the documented key
    budget — better to fail here than to get an opaque 400 mid-backfill.
    """
    raw = model.model_json_schema()
    defs = raw.get("$defs", {})
    flat = _resolve(raw, defs)
    flat.pop("title", None)

    keys = _count_keys(flat)
    if keys > MAX_SCHEMA_KEYS:
        msg = f"Flattened schema has {keys} keys, over the {MAX_SCHEMA_KEYS} the API allows"
        raise SchemaTooLargeError(msg)
    return flat


def response_format(model: type[BaseModel], name: str) -> dict[str, Any]:
    """The ``response_format`` payload for a strict structured-output call."""
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": flatten_schema(model)},
    }
