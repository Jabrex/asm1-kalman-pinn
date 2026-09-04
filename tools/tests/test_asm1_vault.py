import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(r"C:\Users\musta\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe")
NODE = Path(r"C:\Users\musta\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
NODE_MODULES = Path(r"C:\Users\musta\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules")
SOURCE = ROOT / "asm1.xlsx"
EXPECTED_SOURCE_SHA256 = "dff2424c5fa1ed83846ebac7269ac3284317dc8799f18d7edaabb18d60ba892a"


ORIGINAL_RATES = [
    "μH*[SS/(KS+SS)]*[SO/(KO,H+SO)]*[SNH/(KNH,H+SNH)]*XB,H",
    "μH*[SS/(KS+SS)]*[KO,H/(KO,H+SO)]*[SNO/(KNO+SNO)]*[SNH/(KNH,H+SNH)]*ηg*XB,H",
    "μA*[SNH/(KNH+SNH)]*[SO/(KO,A+SO)]*XB,A",
    "bH*XB,H",
    "bA*XB,A",
    "ka*SND*XB,H",
    "kH*[(XS/XB,H)/(KX+XS/XB,H)]*([SO/(KO,H+SO)]+ηh*[KO,H/(KO,H+SO)]*[SNO/(KNO+SNO)])*XB,H",
    "kH*(XND/XS)*[(XS/XB,H)/(KX+XS/XB,H)]*([SO/(KO,H+SO)]+ηh*[KO,H/(KO,H+SO)]*[SNO/(KNO+SNO)])*XB,H",
]

STANDARDISED_RATES = [
    "μOHO,Max*[SB/(KSB,OHO+SB)]*[SO2/(KO2,OHO+SO2)]*[SNHX/(KNHx,OHO+SNHX)]*XOHO",
    "μOHO,Max*[SB/(KSB,OHO+SB)]*[KO2,OHO/(KO2,OHO+SO2)]*[SNOx/(KNOx,OHO+SNOx)]*[SNHX/(KNHx,OHO+SNHX)]*nμOHO,Ax*XOHO",
    "μANO,Max*[SNHX/(KNHx,ANO+SNHX)]*[SO2/(KO2,ANO+SO2)]*XANO",
    "bOHO*XOHO",
    "bANO*XANO",
    "qam*SB,N*XOHO",
    "qXCB_SB,hyd*[(XCB/XOHO)/(KXCB,hyd+XCB/XOHO)]*([SO2/(KO2,OHO+SO2)]+nqhyd,Ax*[KO2,OHO/(KO2,OHO+SO2)]*[SNOx/(KNOx,OHO+SNOx)])*XOHO",
    "qXCB_SB,hyd*(XCB,N/XCB)*[(XCB/XOHO)/(KXCB,hyd+XCB/XOHO)]*([SO2/(KO2,OHO+SO2)]+nqhyd,Ax*[KO2,OHO/(KO2,OHO+SO2)]*[SNOx/(KNOx,OHO+SNOx)])*XOHO",
]


class ASM1VaultBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="asm1-vault-test-")
        cls.output = Path(cls._tmp.name) / "vault"
        cls.source_hash_before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        command = [
            str(PYTHON),
            str(ROOT / "tools" / "build_asm1_vault.py"),
            "--source",
            str(SOURCE),
            "--output",
            str(cls.output),
            "--node",
            str(NODE),
            "--node-modules",
            str(NODE_MODULES),
        ]
        cls.build = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        if cls.build.returncode == 0:
            cls.model = json.loads((cls.output / "data" / "asm1.json").read_text(encoding="utf-8"))
            cls.schema = json.loads((cls.output / "data" / "asm1.schema.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_build_command_succeeds_without_mutating_source(self):
        self.assertEqual(self.build.returncode, 0, self.build.stderr)
        self.assertEqual(self.source_hash_before, EXPECTED_SOURCE_SHA256)
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), EXPECTED_SOURCE_SHA256)

    def test_machine_contract_has_all_required_dimensions(self):
        self.assertEqual(len(self.model["components"]), 14)
        self.assertEqual(len(self.model["parameters"]["stoichiometric"]), 10)
        self.assertEqual(len(self.model["parameters"]["kinetic"]), 15)
        self.assertEqual(len(self.model["processes"]), 8)
        for key in ("corrected_original", "corrected_standardised", "numeric_stoichiometry"):
            matrix = self.model["matrices"][key]["cells"]
            self.assertEqual((len(matrix), len(matrix[0])), (8, 14))
        for key in ("composition_original", "composition_standardised"):
            matrix = self.model["matrices"][key]["cells"]
            self.assertEqual((len(matrix), len(matrix[0])), (3, 14))
        continuity_comp = self.model["matrices"]["continuity_composition"]["cells"]
        self.assertEqual((len(continuity_comp), len(continuity_comp[0])), (14, 3))
        residuals = self.model["continuity"]["residuals"]
        self.assertEqual((len(residuals), len(residuals[0])), (8, 3))
        kinetic = self.model["kinetic_checking"]["cells"]
        self.assertEqual((len(kinetic), len(kinetic[0])), (8, 14))

    def test_preserves_all_rate_text_character_for_character(self):
        self.assertEqual([p["rates"]["original"]["source_text"] for p in self.model["processes"]], ORIGINAL_RATES)
        self.assertEqual(
            [p["rates"]["standardised"]["source_text"] for p in self.model["processes"]],
            STANDARDISED_RATES,
        )

    def test_original_and_standardised_notations_are_symbolically_equivalent(self):
        self.assertEqual(
            self.model["notations"]["equivalence"],
            {
                "corrected_matrix_mismatch_count": 0,
                "composition_matrix_mismatch_count": 0,
                "rate_expression_mismatch_count": 0,
            },
        )

    def test_code_rate_expressions_are_ascii_and_numerically_evaluable(self):
        environment = {
            item["code_id"]: 1.0
            for item in [
                *self.model["components"],
                *self.model["parameters"]["stoichiometric"],
                *self.model["parameters"]["kinetic"],
            ]
        }
        for process in self.model["processes"]:
            for notation in ("original", "standardised"):
                expression = process["rates"][notation]["code_expression"]
                self.assertTrue(expression.isascii())
                self.assertNotIn("[", expression)
                self.assertNotIn("]", expression)
                result = eval(expression, {"__builtins__": {}}, environment)
                self.assertIsInstance(result, float)

    def test_preserves_source_quirks_and_blank_zero_distinction(self):
        sn2 = next(c for c in self.model["components"] if c["original_notation"] == "SN2")
        self.assertEqual(sn2["state_table_status"], "added_term_not_listed")
        self.assertEqual(sn2["source_cell"], "Sheet1!Y11")
        first_blank = self.model["matrices"]["corrected_original"]["cells"][0][0]
        self.assertIsNone(first_blank["raw_cell"])
        self.assertEqual(first_blank["effective_coefficient"], 0)
        explicit_zero = self.model["matrices"]["composition_original"]["cells"][1][0]
        self.assertEqual(explicit_zero["raw_cell"], 0)
        self.assertEqual(explicit_zero["effective_coefficient"], 0)
        anomaly_ids = {item["id"] for item in self.model["source_anomalies"]}
        self.assertTrue({"pocess_rate_typo", "kh_case", "ks_case", "matrix_only_sn2"}.issubset(anomaly_ids))

    def test_preserves_trailing_spaces_in_standardised_parameter_symbols(self):
        stoichiometric = {
            item["original_notation"]: item
            for item in self.model["parameters"]["stoichiometric"]
        }
        self.assertEqual(stoichiometric["YH"]["standardised_notation"], "YOHO ")
        self.assertEqual(stoichiometric["YA"]["standardised_notation"], "YANO ")
        self.assertEqual(stoichiometric["YH"]["aliases"], ["YH", "YOHO"])
        self.assertEqual(stoichiometric["YA"]["aliases"], ["YA", "YANO"])
        anomaly = next(
            item
            for item in self.model["source_anomalies"]
            if item["id"] == "standardised_parameter_trailing_spaces"
        )
        self.assertEqual(anomaly["source_cells"], ["Sheet1!E25", "Sheet1!E27"])

    def test_keeps_state_table_notation_separate_from_matrix_header_case(self):
        sno = next(c for c in self.model["components"] if c["original_notation"] == "SNO")
        snh = next(c for c in self.model["components"] if c["original_notation"] == "SNH")
        self.assertEqual((sno["standardised_notation"], sno["standardised_matrix_header"]), ("SNOx", "SNOX"))
        self.assertEqual((snh["standardised_notation"], snh["standardised_matrix_header"]), ("SNHx", "SNHX"))

    def test_continuity_and_kinetic_semantics_match_source(self):
        self.assertLessEqual(self.model["continuity"]["max_abs_residual"], 1e-15)
        self.assertEqual(self.model["continuity"]["max_abs_residual"], 5.551115123125783e-17)
        self.assertEqual(
            self.model["kinetic_checking"]["category_counts"],
            {"consumed": 13, "biomass": 8, "other_required": 4, "inhibitory": 1},
        )
        self.assertEqual(self.model["kinetic_checking"]["starred_missing_terms"], ["X82", "X84"])

    def test_source_manifest_and_markdown_views_are_complete(self):
        self.assertEqual(self.model["source"]["sha256"], EXPECTED_SOURCE_SHA256)
        self.assertEqual(len(self.model["source"]["formula_cells"]), 56)
        self.assertEqual(len(self.model["named_ranges"]), 10)
        self.assertEqual(len(self.model["legacy_artifacts"]), 3)
        expected_notes = {
            "README.md",
            "State Variables.md",
            "Parameters.md",
            "Processes and Rates.md",
            "Corrected Matrices.md",
            "Composition and Continuity.md",
            "Kinetic Checking Matrix.md",
            "Source Manifest.md",
        }
        self.assertEqual({path.name for path in self.output.glob("*.md")}, expected_notes)
        for note in expected_notes:
            text = (self.output / note).read_text(encoding="utf-8")
            self.assertIn(EXPECTED_SOURCE_SHA256, text)
        self.assertTrue((self.output / "data" / "asm1.schema.json").is_file())

    def test_every_declared_object_schema_rejects_unknown_properties(self):
        unchecked = []

        def visit(node, path):
            if isinstance(node, dict):
                if node.get("type") == "object" and node.get("additionalProperties") is not False:
                    unchecked.append(path)
                for key, value in node.items():
                    visit(value, f"{path}/{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    visit(value, f"{path}/{index}")

        visit(self.schema, "#")
        self.assertEqual(unchecked, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
