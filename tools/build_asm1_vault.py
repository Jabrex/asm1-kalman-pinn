from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
EXPECTED_USED_RANGE = "A1:AC91"
INDEXED_COLORS = {
    "10": "#FF0000",
    "11": "#00FF00",
    "12": "#0000FF",
    "13": "#FFFF00",
    "40": "#00CCFF",
    "57": "#339966",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def col_number(letters: str) -> int:
    result = 0
    for char in letters:
        result = result * 26 + ord(char) - 64
    return result


def col_letters(number: int) -> str:
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def split_cell(reference: str) -> tuple[int, int]:
    match = re.fullmatch(r"([A-Z]+)(\d+)", reference.replace("$", ""))
    if not match:
        raise ValueError(f"Invalid cell reference: {reference}")
    return int(match.group(2)), col_number(match.group(1))


def shift_formula(formula: str, master: str, follower: str) -> str:
    master_row, master_col = split_cell(master)
    follower_row, follower_col = split_cell(follower)
    row_shift = follower_row - master_row
    col_shift = follower_col - master_col
    pattern = re.compile(r"(?<![A-Za-z0-9_.])(?P<abs_col>\$?)(?P<col>[A-Z]{1,3})(?P<abs_row>\$?)(?P<row>\d+)")

    def replace(match: re.Match[str]) -> str:
        col = col_number(match.group("col"))
        row = int(match.group("row"))
        if not match.group("abs_col"):
            col += col_shift
        if not match.group("abs_row"):
            row += row_shift
        return f"{match.group('abs_col')}{col_letters(col)}{match.group('abs_row')}{row}"

    return pattern.sub(replace, formula)


def artifact_value(values: list[list[object]], reference: str) -> object:
    row, col = split_cell(reference)
    try:
        return values[row - 1][col - 1]
    except IndexError:
        return None


def source_cell(reference: str) -> str:
    return f"Sheet1!{reference}"


def code_id_for_symbol(symbol: str) -> str:
    replacements = {"μ": "mu", "η": "eta"}
    result = "".join(replacements.get(char, char) for char in symbol.strip())
    result = re.sub(r"[^A-Za-z0-9]+", "_", result).strip("_")
    result = re.sub(r"_+", "_", result)
    if result and result[0].isdigit():
        result = f"v_{result}"
    return result


def read_xml(archive: zipfile.ZipFile, name: str) -> ET.Element:
    return ET.fromstring(archive.read(name))


def parse_styles(archive: zipfile.ZipFile) -> tuple[list[dict[str, str | None]], list[int]]:
    root = read_xml(archive, "xl/styles.xml")
    fills: list[dict[str, str | None]] = []
    for fill in root.findall(f"./{{{MAIN_NS}}}fills/{{{MAIN_NS}}}fill"):
        pattern = fill.find(f"{{{MAIN_NS}}}patternFill")
        fg = pattern.find(f"{{{MAIN_NS}}}fgColor") if pattern is not None else None
        indexed = fg.get("indexed") if fg is not None else None
        rgb = fg.get("rgb") if fg is not None else None
        if rgb and len(rgb) == 8:
            rgb = f"#{rgb[2:]}"
        elif rgb:
            rgb = f"#{rgb}"
        color = INDEXED_COLORS.get(indexed or "", rgb)
        fills.append(
            {
                "pattern": pattern.get("patternType") if pattern is not None else None,
                "indexed": indexed,
                "rgb": rgb,
                "color": color,
            }
        )
    xfs = root.findall(f"./{{{MAIN_NS}}}cellXfs/{{{MAIN_NS}}}xf")
    return fills, [int(xf.get("fillId", "0")) for xf in xfs]


def parse_formula_manifest(
    sheet_root: ET.Element, values: list[list[object]]
) -> tuple[list[dict[str, object]], dict[str, str]]:
    formula_nodes: list[tuple[str, ET.Element]] = []
    masters: dict[str, tuple[str, str]] = {}
    for cell in sheet_root.findall(f".//{{{MAIN_NS}}}c"):
        formula = cell.find(f"{{{MAIN_NS}}}f")
        if formula is None:
            continue
        reference = cell.get("r")
        if reference is None:
            continue
        formula_nodes.append((reference, formula))
        if formula.get("t") == "shared" and formula.text:
            masters[formula.get("si", "")] = (reference, formula.text)

    manifest: list[dict[str, object]] = []
    by_cell: dict[str, str] = {}
    for reference, formula in formula_nodes:
        formula_type = formula.get("t", "normal")
        shared_index = formula.get("si")
        shared_master = None
        if formula_type == "shared" and formula.text:
            expanded = formula.text
            kind = "shared_master"
            shared_master = reference
        elif formula_type == "shared":
            master_ref, master_formula = masters[shared_index or ""]
            expanded = shift_formula(master_formula, master_ref, reference)
            kind = "shared_follower"
            shared_master = master_ref
        else:
            expanded = formula.text or ""
            kind = "normal"
        source_formula = f"={expanded}"
        by_cell[reference] = source_formula
        manifest.append(
            {
                "cell": source_cell(reference),
                "source_formula": source_formula,
                "formula_kind": kind,
                "shared_index": int(shared_index) if shared_index is not None else None,
                "shared_master": source_cell(shared_master) if shared_master else None,
                "evaluated_value": artifact_value(values, reference),
            }
        )
    return manifest, by_cell


def parse_named_ranges(archive: zipfile.ZipFile, values: list[list[object]]) -> list[dict[str, object]]:
    root = read_xml(archive, "xl/workbook.xml")
    result: list[dict[str, object]] = []
    for item in root.findall(f".//{{{MAIN_NS}}}definedName"):
        formula_text = item.text or ""
        match = re.fullmatch(r"'?([^']+)'?!\$?([A-Z]+)\$?(\d+)", formula_text)
        cell = f"{match.group(2)}{match.group(3)}" if match else None
        result.append(
            {
                "name": item.get("name"),
                "scope": "workbook" if item.get("localSheetId") is None else f"sheet:{item.get('localSheetId')}",
                "formula": f"={formula_text}" if not formula_text.startswith("=") else formula_text,
                "source_cell": source_cell(cell) if cell else None,
                "evaluated_value": artifact_value(values, cell) if cell else None,
                "code_id": code_id_for_symbol(item.get("name", "")),
            }
        )
    return result


def parse_legacy_artifacts(archive: zipfile.ZipFile, sheet_root: ET.Element) -> list[dict[str, object]]:
    rel_root = read_xml(archive, "xl/worksheets/_rels/sheet1.xml.rels")
    relations = {
        rel.get("Id"): rel.get("Target")
        for rel in rel_root.findall(f"{{{PKG_REL_NS}}}Relationship")
    }
    artifacts: list[dict[str, object]] = []
    seen: set[tuple[str | None, str | None, str]] = set()
    for node in sheet_root.findall(f".//{{{MAIN_NS}}}oleObject"):
        rel_id = node.get(f"{{{REL_NS}}}id")
        target = relations.get(rel_id or "", "")
        normalized = str(PurePosixPath("xl/worksheets") / target)
        parts: list[str] = []
        for part in PurePosixPath(normalized).parts:
            if part == "..":
                if parts:
                    parts.pop()
            elif part != ".":
                parts.append(part)
        archive_path = "/".join(parts)
        identity = (node.get("shapeId"), rel_id, archive_path)
        if identity in seen:
            continue
        seen.add(identity)
        payload = archive.read(archive_path)
        artifacts.append(
            {
                "id": f"legacy_equation_{len(artifacts) + 1}",
                "program_id": node.get("progId"),
                "shape_id": node.get("shapeId"),
                "relationship_id": rel_id,
                "archive_path": archive_path,
                "size_bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "anchor_cell": "Y73",
                "width": 0,
                "height": 0,
                "data_role": "none; hidden zero-size legacy OLE object",
            }
        )
    return artifacts


def aliases_for_component(original: str, standardised: str, continuity: str) -> list[str]:
    aliases = [original, standardised, continuity]
    if original == "SNO":
        aliases.extend(["SNOx", "SNOX"])
    if original == "SNH":
        aliases.extend(["SNHx", "SNHX"])
    return list(dict.fromkeys(alias for alias in aliases if alias))


def aliases_for_parameter(original: str, standardised: str) -> list[str]:
    aliases = [original, standardised]
    if original == "kh":
        aliases.append("kH")
    if original == "Ks":
        aliases.append("KS")
    return list(dict.fromkeys(alias.strip() for alias in aliases if alias))


def normalize_expression(expression: str, alias_to_code: dict[str, str]) -> str:
    aliases = sorted(alias_to_code, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(alias) for alias in aliases))
    normalized = pattern.sub(lambda match: alias_to_code[match.group(0)], expression)
    return normalized.replace("[", "(").replace("]", ")")


def evaluate_expression(expression: str, values: dict[str, float]) -> float:
    if not re.fullmatch(r"[A-Za-z0-9_+\-*/(). ]+", expression):
        raise ValueError(f"Unsafe or unsupported coefficient expression: {expression}")
    return float(eval(expression, {"__builtins__": {}}, values))


def matrix_cell(
    reference: str,
    raw: object,
    formula_by_cell: dict[str, str],
    alias_to_code: dict[str, str],
    parameter_values: dict[str, float],
) -> dict[str, object]:
    effective: object = 0 if raw is None else raw
    code_expression: object = effective
    if isinstance(effective, str):
        code_expression = normalize_expression(effective, alias_to_code)
        evaluated = evaluate_expression(code_expression, parameter_values)
    elif isinstance(effective, (int, float)):
        evaluated = float(effective)
    else:
        evaluated = None
    return {
        "raw_cell": raw,
        "effective_coefficient": effective,
        "code_expression": code_expression,
        "evaluated_value": evaluated,
        "source_cell": source_cell(reference),
        "source_formula": formula_by_cell.get(reference),
    }


def extract_matrix(
    values: list[list[object]],
    start_row: int,
    start_col: int,
    row_count: int,
    col_count: int,
    formula_by_cell: dict[str, str],
    alias_to_code: dict[str, str],
    parameter_values: dict[str, float],
) -> list[list[dict[str, object]]]:
    result: list[list[dict[str, object]]] = []
    for row in range(start_row, start_row + row_count):
        current: list[dict[str, object]] = []
        for col in range(start_col, start_col + col_count):
            reference = f"{col_letters(col)}{row}"
            current.append(
                matrix_cell(
                    reference,
                    artifact_value(values, reference),
                    formula_by_cell,
                    alias_to_code,
                    parameter_values,
                )
            )
        result.append(current)
    return result


def matrix_values(cells: list[list[dict[str, object]]]) -> list[list[float]]:
    return [[float(cell["evaluated_value"]) for cell in row] for row in cells]


def build_schema() -> dict[str, object]:
    scalar = {"type": ["string", "number", "null"]}
    matrix_cell_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "raw_cell",
            "effective_coefficient",
            "code_expression",
            "evaluated_value",
            "source_cell",
            "source_formula",
        ],
        "properties": {
            "raw_cell": scalar,
            "effective_coefficient": {"type": ["string", "number"]},
            "code_expression": {"type": ["string", "number"]},
            "evaluated_value": {"type": ["number", "null"]},
            "source_cell": {"type": "string"},
            "source_formula": {"type": ["string", "null"]},
        },
    }

    def matrix_schema(rows: int, cols: int) -> dict[str, object]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["row_labels", "column_labels", "cells", "source_range"],
            "properties": {
                "row_labels": {"type": "array", "minItems": rows, "maxItems": rows, "items": {"type": "string"}},
                "column_labels": {"type": "array", "minItems": cols, "maxItems": cols, "items": {"type": "string"}},
                "cells": {
                    "type": "array",
                    "minItems": rows,
                    "maxItems": rows,
                    "items": {
                        "type": "array",
                        "minItems": cols,
                        "maxItems": cols,
                        "items": {"$ref": "#/definitions/matrixCell"},
                    },
                },
                "source_range": {"type": "string"},
            },
        }

    component = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "index",
            "source_text",
            "original_notation",
            "standardised_notation",
            "standardised_matrix_header",
            "continuity_alias",
            "code_id",
            "aliases",
            "description",
            "unit",
            "source_cell",
            "standardised_notation_source_cell",
            "standardised_matrix_header_source_cell",
            "state_table_status",
        ],
        "properties": {
            "index": {"type": "integer"},
            "source_text": {"type": "string"},
            "original_notation": {"type": "string"},
            "standardised_notation": {"type": "string"},
            "standardised_matrix_header": {"type": "string"},
            "continuity_alias": {"type": "string"},
            "code_id": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]*$"},
            "aliases": {"type": "array", "items": {"type": "string"}},
            "description": {"type": ["string", "null"]},
            "unit": {"type": ["string", "null"]},
            "source_cell": {"type": "string"},
            "standardised_notation_source_cell": {"type": "string"},
            "standardised_matrix_header_source_cell": {"type": "string"},
            "state_table_status": {"enum": ["listed", "added_term_not_listed"]},
        },
    }
    parameter = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "source_text",
            "original_notation",
            "standardised_notation",
            "code_id",
            "aliases",
            "description",
            "unit",
            "source_cells",
            "source_formula",
            "evaluated_value",
        ],
        "properties": {
            "source_text": {"type": "string"},
            "original_notation": {"type": "string"},
            "standardised_notation": {"type": "string"},
            "code_id": {"type": "string"},
            "aliases": {"type": "array", "items": {"type": "string"}},
            "description": {"type": "string"},
            "unit": {"type": "string"},
            "source_cells": {
                "type": "object",
                "additionalProperties": False,
                "required": ["symbol", "value"],
                "properties": {"symbol": {"type": "string"}, "value": {"type": "string"}},
            },
            "source_formula": {"type": ["string", "null"]},
            "evaluated_value": {"type": "number"},
        },
    }
    rate = {
        "type": "object",
        "additionalProperties": False,
        "required": ["source_text", "code_expression", "source_cell", "source_kind"],
        "properties": {
            "source_text": {"type": "string"},
            "code_expression": {"type": "string"},
            "source_cell": {"type": "string"},
            "source_kind": {"const": "text_expression"},
        },
    }
    process = {
        "type": "object",
        "additionalProperties": False,
        "required": ["process_id", "name", "source_cells", "rates"],
        "properties": {
            "process_id": {"type": "integer", "minimum": 1, "maximum": 8},
            "name": {"type": "string"},
            "source_cells": {
                "type": "object",
                "additionalProperties": False,
                "required": ["original", "standardised"],
                "properties": {"original": {"type": "string"}, "standardised": {"type": "string"}},
            },
            "rates": {
                "type": "object",
                "additionalProperties": False,
                "required": ["original", "standardised"],
                "properties": {"original": rate, "standardised": rate},
            },
        },
    }
    formula_cell = {
        "type": "object",
        "additionalProperties": False,
        "required": ["cell", "source_formula", "formula_kind", "shared_index", "shared_master", "evaluated_value"],
        "properties": {
            "cell": {"type": "string"},
            "source_formula": {"type": "string"},
            "formula_kind": {"enum": ["normal", "shared_master", "shared_follower"]},
            "shared_index": {"type": ["integer", "null"]},
            "shared_master": {"type": ["string", "null"]},
            "evaluated_value": scalar,
        },
    }
    correction_legend = {
        "type": "object",
        "additionalProperties": False,
        "required": ["source_text", "color", "indexed_color"],
        "properties": {
            "source_text": {"type": "string"},
            "color": {"type": "string"},
            "indexed_color": {"type": "integer"},
        },
    }
    source_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "path",
            "relative_to_vault",
            "sha256",
            "size_bytes",
            "captured_utc",
            "worksheet",
            "used_range",
            "formula_cells",
            "calc_chain_count",
            "formula_error_count",
            "corrections_legend",
        ],
        "properties": {
            "path": {"type": "string"},
            "relative_to_vault": {"type": "string"},
            "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "size_bytes": {"type": "integer"},
            "captured_utc": {"type": "string"},
            "worksheet": {"const": "Sheet1"},
            "used_range": {"const": "A1:AC91"},
            "formula_cells": {"type": "array", "minItems": 56, "maxItems": 56, "items": formula_cell},
            "calc_chain_count": {"const": 56},
            "formula_error_count": {"const": 0},
            "corrections_legend": {"type": "array", "minItems": 3, "maxItems": 3, "items": correction_legend},
        },
    }
    alias_entry = {
        "type": "object",
        "additionalProperties": False,
        "required": ["source_symbol", "code_id"],
        "properties": {"source_symbol": {"type": "string"}, "code_id": {"type": "string"}},
    }
    equivalence = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "corrected_matrix_mismatch_count",
            "composition_matrix_mismatch_count",
            "rate_expression_mismatch_count",
        ],
        "properties": {
            "corrected_matrix_mismatch_count": {"const": 0},
            "composition_matrix_mismatch_count": {"const": 0},
            "rate_expression_mismatch_count": {"const": 0},
        },
    }
    notations_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["original", "standardised", "continuity", "alias_map", "equivalence"],
        "properties": {
            "original": {"type": "string"},
            "standardised": {"type": "string"},
            "continuity": {"type": "string"},
            "alias_map": {"type": "array", "items": alias_entry},
            "equivalence": equivalence,
        },
    }

    def numeric_grid(rows: int, cols: int) -> dict[str, object]:
        return {
            "type": "array",
            "minItems": rows,
            "maxItems": rows,
            "items": {
                "type": "array",
                "minItems": cols,
                "maxItems": cols,
                "items": {"type": "number"},
            },
        }

    continuity_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "residuals",
            "workbook_residuals",
            "max_abs_residual",
            "tolerance",
            "passes",
            "matches_workbook",
            "calculation",
        ],
        "properties": {
            "residuals": numeric_grid(8, 3),
            "workbook_residuals": numeric_grid(8, 3),
            "max_abs_residual": {"type": "number", "maximum": 1e-15},
            "tolerance": {"const": 1e-15},
            "passes": {"const": True},
            "matches_workbook": {"const": True},
            "calculation": {"type": "string"},
        },
    }
    kinetic_cell_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["category", "color", "marker", "source_cell"],
        "properties": {
            "category": {"type": ["string", "null"]},
            "color": {"type": ["string", "null"]},
            "marker": {"type": ["string", "null"]},
            "source_cell": {"type": "string"},
        },
    }
    category_counts_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["consumed", "biomass", "other_required", "inhibitory"],
        "properties": {
            "consumed": {"const": 13},
            "biomass": {"const": 8},
            "other_required": {"const": 4},
            "inhibitory": {"const": 1},
        },
    }
    kinetic_legend_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["category", "color", "component_function", "rate_consequence"],
        "properties": {
            "category": {"enum": ["consumed", "biomass", "other_required", "inhibitory"]},
            "color": {"type": "string"},
            "component_function": {"type": "string"},
            "rate_consequence": {"type": "string"},
        },
    }
    kinetic_checking_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["row_labels", "column_labels", "cells", "category_counts", "starred_missing_terms", "source_range", "legend"],
        "properties": {
            "row_labels": {"type": "array", "minItems": 8, "maxItems": 8, "items": {"type": "string"}},
            "column_labels": {"type": "array", "minItems": 14, "maxItems": 14, "items": {"type": "string"}},
            "cells": {
                "type": "array",
                "minItems": 8,
                "maxItems": 8,
                "items": {
                    "type": "array",
                    "minItems": 14,
                    "maxItems": 14,
                    "items": kinetic_cell_schema,
                },
            },
            "category_counts": category_counts_schema,
            "starred_missing_terms": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"enum": ["X82", "X84"]},
            },
            "source_range": {"type": "string"},
            "legend": {"type": "array", "minItems": 4, "maxItems": 4, "items": kinetic_legend_schema},
        },
    }
    named_range_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "scope", "formula", "source_cell", "evaluated_value", "code_id"],
        "properties": {
            "name": {"type": "string"},
            "scope": {"type": "string"},
            "formula": {"type": "string"},
            "source_cell": {"type": ["string", "null"]},
            "evaluated_value": scalar,
            "code_id": {"type": "string"},
        },
    }
    anomaly_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "source_text", "source_cells", "handling"],
        "properties": {
            "id": {"type": "string"},
            "source_text": {"type": "string"},
            "source_cells": {"type": "array", "items": {"type": "string"}},
            "handling": {"type": "string"},
        },
    }
    legacy_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "id",
            "program_id",
            "shape_id",
            "relationship_id",
            "archive_path",
            "size_bytes",
            "sha256",
            "anchor_cell",
            "width",
            "height",
            "data_role",
        ],
        "properties": {
            "id": {"type": "string"},
            "program_id": {"const": "Equation.3"},
            "shape_id": {"type": "string"},
            "relationship_id": {"type": "string"},
            "archive_path": {"type": "string"},
            "size_bytes": {"type": "integer"},
            "sha256": {"type": "string"},
            "anchor_cell": {"const": "Y73"},
            "width": {"const": 0},
            "height": {"const": 0},
            "data_role": {"type": "string"},
        },
    }
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "asm1.schema.json",
        "title": "ASM1 canonical model data",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "source",
            "notations",
            "components",
            "parameters",
            "processes",
            "matrices",
            "continuity",
            "kinetic_checking",
            "named_ranges",
            "source_anomalies",
            "legacy_artifacts",
        ],
        "properties": {
            "schema_version": {"const": "1.0.0"},
            "source": source_schema,
            "notations": notations_schema,
            "components": {"type": "array", "minItems": 14, "maxItems": 14, "items": component},
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["stoichiometric", "kinetic"],
                "properties": {
                    "stoichiometric": {"type": "array", "minItems": 10, "maxItems": 10, "items": parameter},
                    "kinetic": {"type": "array", "minItems": 15, "maxItems": 15, "items": parameter},
                },
            },
            "processes": {"type": "array", "minItems": 8, "maxItems": 8, "items": process},
            "matrices": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "corrected_original",
                    "corrected_standardised",
                    "composition_original",
                    "composition_standardised",
                    "continuity_composition",
                    "numeric_stoichiometry",
                ],
                "properties": {
                    "corrected_original": matrix_schema(8, 14),
                    "corrected_standardised": matrix_schema(8, 14),
                    "composition_original": matrix_schema(3, 14),
                    "composition_standardised": matrix_schema(3, 14),
                    "continuity_composition": matrix_schema(14, 3),
                    "numeric_stoichiometry": matrix_schema(8, 14),
                },
            },
            "continuity": continuity_schema,
            "kinetic_checking": kinetic_checking_schema,
            "named_ranges": {"type": "array", "minItems": 10, "maxItems": 10, "items": named_range_schema},
            "source_anomalies": {"type": "array", "items": anomaly_schema},
            "legacy_artifacts": {"type": "array", "minItems": 3, "maxItems": 3, "items": legacy_schema},
        },
        "definitions": {"matrixCell": matrix_cell_schema},
    }


def md_value(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def frontmatter(model: dict[str, object]) -> str:
    return (
        "---\n"
        "model: ASM1\n"
        f"schema_version: \"{model['schema_version']}\"\n"
        f"source_sha256: {model['source']['sha256']}\n"
        "canonical_data: data/asm1.json\n"
        "---\n\n"
    )


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    head = "| " + " | ".join(headers) + " |\n"
    separator = "| " + " | ".join("---" for _ in headers) + " |\n"
    body = "".join("| " + " | ".join(md_value(value) for value in row) + " |\n" for row in rows)
    return head + separator + body


def matrix_markdown(matrix: dict[str, object]) -> str:
    headers = ["Process / conserved quantity", *matrix["column_labels"]]
    rows = []
    for label, cells in zip(matrix["row_labels"], matrix["cells"]):
        rows.append([label, *[(cell["raw_cell"] if cell["raw_cell"] is not None else "∅ (0)") for cell in cells]])
    return matrix_markdown_table(headers, rows)


def matrix_markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    return markdown_table(headers, rows)


def generate_notes(model: dict[str, object], output: Path) -> None:
    prefix = frontmatter(model)
    source_hash = model["source"]["sha256"]
    readme = prefix + f"""# ASM1 Canonical Knowledge Base

This vault records the ASM1 workbook without silently correcting source notation. Machine consumers must use [`data/asm1.json`](data/asm1.json); Markdown files are generated views of that JSON.

## Navigation

- [[State Variables]]
- [[Parameters]]
- [[Processes and Rates]]
- [[Corrected Matrices]]
- [[Composition and Continuity]]
- [[Kinetic Checking Matrix]]
- [[Source Manifest]]
- [[Audit Report]]

## Contract

- Operational component vector: **14** components (13 listed state variables plus matrix-only `SN2`).
- Processes: **8**.
- Parameters: **10 stoichiometric + 15 kinetic**.
- Source SHA-256: `{source_hash}`.
- Source text has precedence over normalized aliases. `code_id` and `code_expression` are additive implementation aids.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    component_rows = [
        [
            item["index"],
            f"`{item['original_notation']}`",
            f"`{item['standardised_notation']}`",
            f"`{item['standardised_matrix_header']}`",
            f"`{item['continuity_alias']}`",
            f"`{item['code_id']}`",
            item["description"],
            item["unit"],
            item["state_table_status"],
            item["source_cell"],
        ]
        for item in model["components"]
    ]
    state_note = prefix + "# State Variables\n\n"
    state_note += "The source table lists 13 state variables. `SN2` is preserved as the fourteenth operational component because it is explicitly added to both corrected matrices.\n\n"
    state_note += markdown_table(
        ["#", "Original", "State-table standardised", "Matrix header", "Continuity", "Code ID", "Description", "Unit", "Status", "Source"],
        component_rows,
    )
    (output / "State Variables.md").write_text(state_note, encoding="utf-8")

    parameter_note = prefix + "# Parameters\n\n"
    for title, key in (("Stoichiometric parameters", "stoichiometric"), ("Kinetic parameters", "kinetic")):
        parameter_note += f"## {title}\n\n"
        rows = []
        for item in model["parameters"][key]:
            rows.append(
                [
                    f"`{item['original_notation']}`",
                    f"`{item['standardised_notation']}`",
                    f"`{item['code_id']}`",
                    item["description"],
                    item["unit"],
                    md_value(item["evaluated_value"]),
                    f"`{item['source_formula']}`" if item["source_formula"] else "—",
                    item["source_cells"]["value"],
                ]
            )
        parameter_note += markdown_table(
            ["Original", "Standardised", "Code ID", "Description", "Unit", "Full value", "Source formula", "Source"],
            rows,
        ) + "\n"
    (output / "Parameters.md").write_text(parameter_note, encoding="utf-8")

    process_note = prefix + "# Processes and Rates\n\n"
    process_note += "Rate cells are source text, not Excel formulas. Each expression is preserved character-for-character.\n\n"
    for process in model["processes"]:
        process_note += f"## {process['process_id']}. {process['name']}\n\n"
        for notation in ("original", "standardised"):
            rate = process["rates"][notation]
            process_note += f"### {notation.title()} source expression — `{rate['source_cell']}`\n\n```text\n{rate['source_text']}\n```\n\n"
            process_note += f"Code-safe expression:\n\n```text\n{rate['code_expression']}\n```\n\n"
    (output / "Processes and Rates.md").write_text(process_note, encoding="utf-8")

    corrected_note = prefix + "# Corrected Matrices\n\n"
    corrected_note += "`∅ (0)` means the workbook cell is blank while its effective coefficient is zero. A literal `0` remains distinguishable.\n\n"
    corrected_note += "## Corrected Matrix: Original Notation\n\n" + matrix_markdown(model["matrices"]["corrected_original"]) + "\n"
    corrected_note += "## Corrected Matrix: Standardised Notation\n\n" + matrix_markdown(model["matrices"]["corrected_standardised"]) + "\n"
    (output / "Corrected Matrices.md").write_text(corrected_note, encoding="utf-8")

    continuity_note = prefix + "# Composition and Continuity\n\n"
    continuity_note += "## Original composition matrix\n\n" + matrix_markdown(model["matrices"]["composition_original"]) + "\n"
    continuity_note += "## Standardised composition matrix\n\n" + matrix_markdown(model["matrices"]["composition_standardised"]) + "\n"
    continuity_note += "## Continuity composition matrix\n\n" + matrix_markdown(model["matrices"]["continuity_composition"]) + "\n"
    continuity_note += "## Numeric stoichiometry\n\n" + matrix_markdown(model["matrices"]["numeric_stoichiometry"]) + "\n"
    residual_rows = [
        [process["process_id"], process["name"], *[format(value, ".17g") for value in residual]]
        for process, residual in zip(model["processes"], model["continuity"]["residuals"])
    ]
    continuity_note += "## Independently recomputed residuals\n\n"
    continuity_note += markdown_table(["#", "Process", "COD", "N", "Charge"], residual_rows)
    continuity_note += f"\nMaximum absolute residual: `{model['continuity']['max_abs_residual']:.17g}`; acceptance tolerance: `{model['continuity']['tolerance']:.1e}`.\n"
    (output / "Composition and Continuity.md").write_text(continuity_note, encoding="utf-8")

    kinetic_note = prefix + "# Kinetic Checking Matrix\n\n"
    kinetic_note += markdown_table(
        ["Category", "Source color", "Meaning", "Rate consequence"],
        [
            [item["category"], item["color"], item["component_function"], item["rate_consequence"]]
            for item in model["kinetic_checking"]["legend"]
        ],
    )
    kinetic_rows = []
    for process, row in zip(model["processes"], model["kinetic_checking"]["cells"]):
        kinetic_rows.append(
            [
                process["name"],
                *[
                    (f"{cell['category']}*" if cell["marker"] else cell["category"]) if cell["category"] else "—"
                    for cell in row
                ],
            ]
        )
    kinetic_note += "\n" + markdown_table(
        ["Process", *model["kinetic_checking"]["column_labels"]], kinetic_rows
    )
    kinetic_note += "\n`X82` and `X84` are green consumed-component cells marked `*`. The workbook states that these missing alkalinity kinetic terms have not been corrected; no term is invented here.\n"
    (output / "Kinetic Checking Matrix.md").write_text(kinetic_note, encoding="utf-8")

    manifest_note = prefix + "# Source Manifest\n\n"
    manifest_note += markdown_table(
        ["Property", "Value"],
        [
            ["Source path", model["source"]["path"]],
            ["SHA-256", source_hash],
            ["Size", model["source"]["size_bytes"]],
            ["Worksheet", model["source"]["worksheet"]],
            ["Used range", model["source"]["used_range"]],
            ["Formula cells", len(model["source"]["formula_cells"])],
            ["Calculation-chain cells", model["source"]["calc_chain_count"]],
            ["Named ranges", len(model["named_ranges"])],
            ["Formula errors", model["source"]["formula_error_count"]],
        ],
    )
    manifest_note += "\n## Named ranges\n\n"
    manifest_note += markdown_table(
        ["Name", "Formula", "Value", "Source"],
        [[item["name"], item["formula"], item["evaluated_value"], item["source_cell"]] for item in model["named_ranges"]],
    )
    manifest_note += "\n## Formula cells\n\n"
    manifest_note += markdown_table(
        ["Cell", "Kind", "Formula", "Value"],
        [
            [item["cell"], item["formula_kind"], f"`{item['source_formula']}`", item["evaluated_value"]]
            for item in model["source"]["formula_cells"]
        ],
    )
    manifest_note += "\n## Source anomalies preserved\n\n"
    manifest_note += markdown_table(
        ["ID", "Source text", "Cells", "Handling"],
        [[item["id"], item["source_text"], ", ".join(item["source_cells"]), item["handling"]] for item in model["source_anomalies"]],
    )
    manifest_note += "\n## Legacy artifacts\n\n"
    manifest_note += markdown_table(
        ["ID", "Program", "Archive path", "Size", "SHA-256", "Anchor", "Role"],
        [
            [item["id"], item["program_id"], item["archive_path"], item["size_bytes"], item["sha256"], item["anchor_cell"], item["data_role"]]
            for item in model["legacy_artifacts"]
        ],
    )
    manifest_note += "\n## Literature cross-check\n\n- Hauduc et al. (2010): https://pubmed.ncbi.nlm.nih.gov/20182061/\n- Corominas et al. (2010): https://publications.polymtl.ca/18466/\n"
    (output / "Source Manifest.md").write_text(manifest_note, encoding="utf-8")


def build_model(snapshot: Path, artifact: dict[str, object], source_path: Path, source_hash: str) -> dict[str, object]:
    values = artifact["values"]
    with zipfile.ZipFile(snapshot) as archive:
        sheet_root = read_xml(archive, "xl/worksheets/sheet1.xml")
        fills, xfs = parse_styles(archive)
        formula_manifest, formula_by_cell = parse_formula_manifest(sheet_root, values)
        named_ranges = parse_named_ranges(archive, values)
        legacy_artifacts = parse_legacy_artifacts(archive, sheet_root)
        calc_chain_count = 0
        if "xl/calcChain.xml" in archive.namelist():
            calc_root = read_xml(archive, "xl/calcChain.xml")
            calc_chain_count = len(calc_root.findall(f".//{{{MAIN_NS}}}c"))
        style_by_cell = {
            cell.get("r"): int(cell.get("s", "0"))
            for cell in sheet_root.findall(f".//{{{MAIN_NS}}}c")
            if cell.get("r")
        }

    original_headers = [artifact_value(values, f"{col_letters(col)}11") for col in range(12, 26)]
    standard_headers = [artifact_value(values, f"{col_letters(col)}28") for col in range(12, 26)]
    continuity_headers = [artifact_value(values, f"{col_letters(col)}62") for col in range(12, 26)]
    state_rows = {
        artifact_value(values, f"D{row}"): row
        for row in range(12, 25)
        if artifact_value(values, f"D{row}")
    }
    components: list[dict[str, object]] = []
    for index, (original, standardised_matrix_header, continuity) in enumerate(
        zip(original_headers, standard_headers, continuity_headers), start=1
    ):
        if original in state_rows:
            row = state_rows[original]
            standardised = artifact_value(values, f"E{row}")
            description = artifact_value(values, f"C{row}")
            unit = artifact_value(values, f"F{row}")
            status = "listed"
            cell = f"D{row}"
            standardised_cell = f"E{row}"
        else:
            standardised = standardised_matrix_header
            description = None
            unit = None
            status = "added_term_not_listed"
            cell = "Y11"
            standardised_cell = "Y28"
        standardised_matrix_cell = f"{col_letters(11 + index)}28"
        components.append(
            {
                "index": index,
                "source_text": original,
                "original_notation": original,
                "standardised_notation": standardised,
                "standardised_matrix_header": standardised_matrix_header,
                "continuity_alias": continuity,
                "code_id": code_id_for_symbol(continuity),
                "aliases": list(
                    dict.fromkeys(
                        aliases_for_component(original, standardised, continuity)
                        + [standardised_matrix_header]
                    )
                ),
                "description": description,
                "unit": unit,
                "source_cell": source_cell(cell),
                "standardised_notation_source_cell": source_cell(standardised_cell),
                "standardised_matrix_header_source_cell": source_cell(standardised_matrix_cell),
                "state_table_status": status,
            }
        )

    def parameter_rows(start: int, stop: int) -> list[dict[str, object]]:
        result = []
        for row in range(start, stop + 1):
            original = artifact_value(values, f"D{row}")
            standardised = artifact_value(values, f"E{row}")
            aliases = aliases_for_parameter(original, standardised)
            result.append(
                {
                    "source_text": original,
                    "original_notation": original,
                    "standardised_notation": standardised,
                    "code_id": code_id_for_symbol(original),
                    "aliases": aliases,
                    "description": artifact_value(values, f"C{row}"),
                    "unit": artifact_value(values, f"F{row}"),
                    "source_cells": {"symbol": source_cell(f"D{row}"), "value": source_cell(f"G{row}")},
                    "source_formula": formula_by_cell.get(f"G{row}"),
                    "evaluated_value": artifact_value(values, f"G{row}"),
                }
            )
        return result

    stoichiometric_parameters = parameter_rows(25, 34)
    kinetic_parameters = parameter_rows(35, 49)
    all_parameters = [*stoichiometric_parameters, *kinetic_parameters]
    alias_to_code: dict[str, str] = {}
    for component in components:
        for alias in component["aliases"]:
            alias_to_code[alias] = component["code_id"]
    for parameter in all_parameters:
        for alias in parameter["aliases"]:
            alias_to_code[alias] = parameter["code_id"]
    parameter_values = {parameter["code_id"]: float(parameter["evaluated_value"]) for parameter in all_parameters}

    process_names = [artifact_value(values, f"K{row}") for row in range(12, 20)]
    processes = []
    for index, name in enumerate(process_names, start=1):
        original_row = 11 + index
        standard_row = 28 + index
        original_rate = artifact_value(values, f"Z{original_row}")
        standard_rate = artifact_value(values, f"Z{standard_row}")
        processes.append(
            {
                "process_id": index,
                "name": name,
                "source_cells": {"original": source_cell(f"K{original_row}"), "standardised": source_cell(f"K{standard_row}")},
                "rates": {
                    "original": {
                        "source_text": original_rate,
                        "code_expression": normalize_expression(original_rate, alias_to_code),
                        "source_cell": source_cell(f"Z{original_row}"),
                        "source_kind": "text_expression",
                    },
                    "standardised": {
                        "source_text": standard_rate,
                        "code_expression": normalize_expression(standard_rate, alias_to_code),
                        "source_cell": source_cell(f"Z{standard_row}"),
                        "source_kind": "text_expression",
                    },
                },
            }
        )

    original_cells = extract_matrix(values, 12, 12, 8, 14, formula_by_cell, alias_to_code, parameter_values)
    standard_cells = extract_matrix(values, 29, 12, 8, 14, formula_by_cell, alias_to_code, parameter_values)
    composition_original = extract_matrix(values, 21, 12, 3, 14, formula_by_cell, alias_to_code, parameter_values)
    composition_standard = extract_matrix(values, 38, 12, 3, 14, formula_by_cell, alias_to_code, parameter_values)
    numeric_cells = extract_matrix(values, 63, 12, 8, 14, formula_by_cell, alias_to_code, parameter_values)

    continuity_rows = [47, 48, 49, 50, *range(52, 62)]
    continuity_labels = [artifact_value(values, f"Z{row}") for row in continuity_rows]
    if continuity_labels != continuity_headers:
        raise ValueError(f"Continuity component ordering mismatch: {continuity_labels} != {continuity_headers}")
    continuity_cells: list[list[dict[str, object]]] = []
    for row in continuity_rows:
        continuity_cells.append(
            [
                matrix_cell(
                    f"{col_letters(col)}{row}",
                    artifact_value(values, f"{col_letters(col)}{row}"),
                    formula_by_cell,
                    alias_to_code,
                    parameter_values,
                )
                for col in range(27, 30)
            ]
        )

    numeric_values = matrix_values(numeric_cells)
    composition_values = matrix_values(continuity_cells)
    residuals = []
    for row in numeric_values:
        residual_row = []
        for quantity in range(3):
            total = 0.0
            for col in range(14):
                total += row[col] * composition_values[col][quantity]
            residual_row.append(total)
        residuals.append(residual_row)
    workbook_residuals = [
        [float(artifact_value(values, f"{col_letters(col)}{row}")) for col in range(27, 30)]
        for row in range(63, 71)
    ]
    max_abs_residual = max(abs(value) for row in residuals for value in row)

    kinetic_columns = [artifact_value(values, f"{col_letters(col)}81") for col in range(12, 26)]
    color_category = {
        "#00FF00": "consumed",
        "#00CCFF": "biomass",
        "#FFFF00": "other_required",
        "#FF0000": "inhibitory",
    }
    kinetic_cells: list[list[dict[str, object]]] = []
    category_counts: Counter[str] = Counter()
    starred: list[str] = []
    for row in range(82, 90):
        current = []
        for col in range(12, 26):
            reference = f"{col_letters(col)}{row}"
            style = style_by_cell.get(reference, 0)
            fill_id = xfs[style] if style < len(xfs) else 0
            color = fills[fill_id]["color"] if fill_id < len(fills) else None
            category = color_category.get(color)
            marker = artifact_value(values, reference) == "*"
            if category:
                category_counts[category] += 1
            if marker:
                starred.append(reference)
            current.append(
                {
                    "category": category,
                    "color": color if category else None,
                    "marker": "*" if marker else None,
                    "source_cell": source_cell(reference),
                }
            )
        kinetic_cells.append(current)

    errors = {"#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A"}
    formula_error_count = sum(1 for row in values for value in row if isinstance(value, str) and value in errors)
    matrices = {
        "corrected_original": {
            "row_labels": process_names,
            "column_labels": original_headers,
            "cells": original_cells,
            "source_range": "Sheet1!L12:Y19",
        },
        "corrected_standardised": {
            "row_labels": process_names,
            "column_labels": standard_headers,
            "cells": standard_cells,
            "source_range": "Sheet1!L29:Y36",
        },
        "composition_original": {
            "row_labels": ["COD", "N", "Charge"],
            "column_labels": original_headers,
            "cells": composition_original,
            "source_range": "Sheet1!L21:Y23",
        },
        "composition_standardised": {
            "row_labels": ["COD", "N", "Charge"],
            "column_labels": standard_headers,
            "cells": composition_standard,
            "source_range": "Sheet1!L38:Y40",
        },
        "continuity_composition": {
            "row_labels": continuity_labels,
            "column_labels": ["COD", "N", "Charge"],
            "cells": continuity_cells,
            "source_range": "Sheet1!Z47:AC61 (row 51 is visual spacing)",
        },
        "numeric_stoichiometry": {
            "row_labels": process_names,
            "column_labels": continuity_headers,
            "cells": numeric_cells,
            "source_range": "Sheet1!L63:Y70",
        },
    }
    anomalies = [
        {"id": "pocess_rate_typo", "source_text": "Pocess rate", "source_cells": ["Sheet1!Z11", "Sheet1!Z28"], "handling": "preserved; canonical concept is process_rate"},
        {"id": "kh_case", "source_text": "kh in parameter table; kH in process rates", "source_cells": ["Sheet1!D35", "Sheet1!Z18", "Sheet1!Z19"], "handling": "preserved; all aliases map to k_h"},
        {"id": "ks_case", "source_text": "Ks in parameter table; KS in process rates", "source_cells": ["Sheet1!D40", "Sheet1!Z12", "Sheet1!Z13"], "handling": "preserved; all aliases map to K_S"},
        {"id": "matrix_only_sn2", "source_text": "SN2 occurs in matrices but has no state-variable table row", "source_cells": ["Sheet1!Y11", "Sheet1!Y28"], "handling": "preserved as operational component 14 with added_term_not_listed status"},
        {"id": "sno_case_variants", "source_text": "SNOx / SNOX / S_NO", "source_cells": ["Sheet1!E19", "Sheet1!T28", "Sheet1!T62"], "handling": "preserved as aliases of S_NO"},
        {"id": "snh_case_variants", "source_text": "SNHx / SNHX / S_NH", "source_cells": ["Sheet1!E18", "Sheet1!U28", "Sheet1!U62"], "handling": "preserved as aliases of S_NH"},
        {"id": "named_range_aliases", "source_text": "iNO3,N2 / ASM1_i_NOx.N2 and NO3 / NOx variants", "source_cells": ["Sheet1!D30", "Sheet1!G30"], "handling": "source names retained; code identifiers are separate"},
        {"id": "oxygen_unit_prefix", "source_text": "- g COD.m-3", "source_cells": ["Sheet1!F14"], "handling": "unit text preserved verbatim"},
        {"id": "double_negative_process3", "source_text": "-(-iCOD_NO3-YA)/YA", "source_cells": ["Sheet1!S14"], "handling": "expression preserved and independently evaluated"},
        {"id": "text_rate_cells", "source_text": "All 16 rate expressions are text, not Excel formulas", "source_cells": ["Sheet1!Z12:Z19", "Sheet1!Z29:Z36"], "handling": "source_kind is text_expression"},
        {"id": "comma_and_dot_identifiers", "source_text": "Commas and dots are part of symbols and named ranges", "source_cells": ["Sheet1!D22", "Sheet1!D30", "Sheet1!G30"], "handling": "longest-match alias table; never split as CSV"},
        {"id": "missing_alkalinity_kinetics", "source_text": "* Missing kinetic terms that have not been corrected", "source_cells": ["Sheet1!X82", "Sheet1!X84", "Sheet1!K91:Y91"], "handling": "flagged; no kinetic term invented"},
        {"id": "knh_h_special_value", "source_text": "*same KNH,H value as ASM2d has been chosen", "source_cells": ["Sheet1!D44:G44", "Sheet1!C51"], "handling": "source note and value preserved"},
        {"id": "standardised_parameter_trailing_spaces", "source_text": "YOHO\u2420 and YANO\u2420 contain a trailing U+0020 space", "source_cells": ["Sheet1!E25", "Sheet1!E27"], "handling": "raw standardised_notation preserves the final space; aliases and code identifiers are trimmed separately"},
    ]
    alias_entries = [
        {"source_symbol": alias, "code_id": code}
        for alias, code in sorted(alias_to_code.items(), key=lambda item: (-len(item[0]), item[0]))
    ]
    corrected_matrix_mismatches = sum(
        original_cells[row][col]["code_expression"] != standard_cells[row][col]["code_expression"]
        for row in range(8)
        for col in range(14)
    )
    composition_matrix_mismatches = sum(
        composition_original[row][col]["code_expression"] != composition_standard[row][col]["code_expression"]
        for row in range(3)
        for col in range(14)
    )
    rate_expression_mismatches = sum(
        process["rates"]["original"]["code_expression"]
        != process["rates"]["standardised"]["code_expression"]
        for process in processes
    )
    return {
        "schema_version": "1.0.0",
        "source": {
            "path": str(source_path.resolve()),
            "relative_to_vault": "../asm1.xlsx",
            "sha256": source_hash,
            "size_bytes": snapshot.stat().st_size,
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "worksheet": "Sheet1",
            "used_range": EXPECTED_USED_RANGE,
            "formula_cells": formula_manifest,
            "calc_chain_count": calc_chain_count,
            "formula_error_count": formula_error_count,
            "corrections_legend": [
                {"source_text": "added term", "color": "#FF0000", "indexed_color": 10},
                {"source_text": "corrected term", "color": "#339966", "indexed_color": 57},
                {"source_text": "term to be careful with", "color": "#0000FF", "indexed_color": 12},
            ],
        },
        "notations": {
            "original": "Corrected Matrix: Original Notation",
            "standardised": "Corrected Matrix: Standardised Notation (Corominas et al., 2010)",
            "continuity": "Underscore/dot labels used in the numeric continuity block",
            "alias_map": alias_entries,
            "equivalence": {
                "corrected_matrix_mismatch_count": corrected_matrix_mismatches,
                "composition_matrix_mismatch_count": composition_matrix_mismatches,
                "rate_expression_mismatch_count": rate_expression_mismatches,
            },
        },
        "components": components,
        "parameters": {"stoichiometric": stoichiometric_parameters, "kinetic": kinetic_parameters},
        "processes": processes,
        "matrices": matrices,
        "continuity": {
            "residuals": residuals,
            "workbook_residuals": workbook_residuals,
            "max_abs_residual": max_abs_residual,
            "tolerance": 1e-15,
            "passes": max_abs_residual <= 1e-15,
            "matches_workbook": all(
                math.isclose(residuals[row][col], workbook_residuals[row][col], rel_tol=0, abs_tol=1e-30)
                for row in range(8)
                for col in range(3)
            ),
            "calculation": "numeric_stoichiometry (8x14) multiplied by continuity_composition (14x3)",
        },
        "kinetic_checking": {
            "row_labels": process_names,
            "column_labels": kinetic_columns,
            "cells": kinetic_cells,
            "category_counts": {
                "consumed": category_counts["consumed"],
                "biomass": category_counts["biomass"],
                "other_required": category_counts["other_required"],
                "inhibitory": category_counts["inhibitory"],
            },
            "starred_missing_terms": starred,
            "source_range": "Sheet1!K81:Y89",
            "legend": [
                {"category": "consumed", "color": "#00FF00", "component_function": "Consumed component (every state variable with a negative sign)", "rate_consequence": "Limitation monod function"},
                {"category": "biomass", "color": "#00CCFF", "component_function": "Biomass involved in the process", "rate_consequence": "proportional to the biomass concentration"},
                {"category": "other_required", "color": "#FFFF00", "component_function": "Other required component", "rate_consequence": "Limitation monod function"},
                {"category": "inhibitory", "color": "#FF0000", "component_function": "Inhibitory component", "rate_consequence": "Inhibitory monod function"},
            ],
        },
        "named_ranges": named_ranges,
        "source_anomalies": anomalies,
        "legacy_artifacts": legacy_artifacts,
    }


def run_build(args: argparse.Namespace) -> None:
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    node = Path(args.node).resolve()
    node_modules = Path(args.node_modules).resolve()
    extractor = Path(__file__).with_name("extract_asm1_artifact.mjs")
    if not source.is_file():
        raise FileNotFoundError(source)
    if not extractor.is_file():
        raise FileNotFoundError(extractor)
    output.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="asm1-build-") as temp_name:
        temp = Path(temp_name)
        snapshot = temp / "asm1-snapshot.xlsx"
        shutil.copyfile(source, snapshot)
        source_hash = sha256_bytes(snapshot.read_bytes())
        runtime = temp / "node-runtime"
        runtime.mkdir()
        junction = runtime / "node_modules"
        subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(node_modules)], check=True, capture_output=True)
        runtime_extractor = runtime / extractor.name
        shutil.copyfile(extractor, runtime_extractor)
        artifact_json = temp / "artifact.json"
        subprocess.run(
            [str(node), str(runtime_extractor), str(snapshot), str(artifact_json)],
            cwd=runtime,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        artifact = json.loads(artifact_json.read_text(encoding="utf-8"))
        if artifact["worksheet"] != "Sheet1" or artifact["used_range"] != EXPECTED_USED_RANGE:
            raise ValueError(f"Unexpected artifact workbook shape: {artifact['worksheet']} {artifact['used_range']}")
        model = build_model(snapshot, artifact, source, source_hash)
        schema = build_schema()
        (output / "data" / "asm1.json").write_text(
            json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (output / "data" / "asm1.schema.json").write_text(
            json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        generate_notes(model, output)
        final_snapshot = temp / "asm1-final.xlsx"
        shutil.copyfile(source, final_snapshot)
        final_hash = sha256_bytes(final_snapshot.read_bytes())
        if final_hash != source_hash:
            raise RuntimeError(f"Source workbook changed during extraction: {source_hash} -> {final_hash}")
        print(json.dumps({"status": "built", "source_sha256": source_hash, "output": str(output)}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a canonical ASM1 Obsidian knowledge base from asm1.xlsx")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--node-modules", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_build(parse_args())
