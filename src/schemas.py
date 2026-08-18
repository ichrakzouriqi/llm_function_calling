"""
schemas.py – Pydantic data models for the call-me-maybe project.

Two models:
  - FunctionDefinition : one entry from functions_definition.json
  - FunctionCall       : one output entry  (prompt / name / parameters)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# FunctionDefinition
# ---------------------------------------------------------------------------

class ParameterSpec(BaseModel):
    """Schema for a single parameter: just its JSON type."""

    type: str  # "string" | "number" | "integer" | "boolean"


class ReturnSpec(BaseModel):
    """Schema for the return type (we store it but never decode it)."""

    type: str


class FunctionDefinition(BaseModel):
    """
    Mirrors one object inside functions_definition.json.

    Example JSON:
        {
            "name": "fn_add_numbers",
            "description": "Add two numbers together and return their sum.",
            "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
            "returns": {"type": "number"}
        }
    """

    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ReturnSpec

    def ordered_parameters(self) -> list[tuple[str, str]]:
        """
        Return (param_name, json_type) pairs in declaration order.

        Python dicts preserve insertion order (3.7+), so the order matches
        the JSON file — which is what the grader expects.
        """
        return [(pname, pspec.type) for pname, pspec in self.parameters.items()]


# ---------------------------------------------------------------------------
# FunctionCall
# ---------------------------------------------------------------------------

class FunctionCall(BaseModel):
    """
    One output entry – exactly three keys required by the grader:
        prompt     : the original question (copied verbatim)
        name       : the chosen function name
        parameters : dict mapping param names to typed Python values
    """

    prompt: str
    name: str
    parameters: dict[str, Any]
