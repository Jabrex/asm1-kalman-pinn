import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";


const [sourcePath, outputPath] = process.argv.slice(2);
if (!sourcePath || !outputPath) {
  throw new Error("Usage: node extract_asm1_artifact.mjs <source.xlsx> <output.json>");
}

const input = await FileBlob.load(sourcePath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheets = JSON.parse(
  (await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 4000 })).ndjson
    .split("\n")
    .filter(Boolean)
    .map((line) => line.trim())
    .filter((line) => line.startsWith("{"))
    .reduce((acc, line) => {
      const item = JSON.parse(line);
      if (item.kind === "sheet") acc.push(item);
      return acc;
    }, [])
    .map((item) => JSON.stringify(item))
    .join(",")
    .replace(/^/, "[")
    .replace(/$/, "]"),
);

if (sheets.length !== 1 || sheets[0].name !== "Sheet1") {
  throw new Error(`Expected exactly one worksheet named Sheet1; received ${JSON.stringify(sheets)}`);
}

const sheet = workbook.worksheets.getItem("Sheet1");
const used = sheet.getRange("A1:AC91");
const errorScan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "ASM1 formula error scan",
});

const payload = {
  source_path: sourcePath,
  worksheet: "Sheet1",
  used_range: "A1:AC91",
  values: used.values,
  formulas: used.formulas,
  formula_infos: used.formulaInfos,
  error_scan_ndjson: errorScan.ndjson,
};

await fs.writeFile(outputPath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
