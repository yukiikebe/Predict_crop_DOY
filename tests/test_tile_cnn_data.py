from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import torch

from doy_prediction.tile_cnn_data import (
    build_tile_crop_records,
    get_feature_names,
    observations_to_tensor,
)
from doy_prediction.tile_cnn_model import TileCNNRegressor, normalized_to_doy


class TileCnnDataTests(unittest.TestCase):
    def test_build_records_multiple_crops(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "outputs" / "2020_AR" / "0_1"
            base.mkdir(parents=True)
            write_test_workbook(
                base / "harvest_summary_all_crops.xlsx",
                rows=[
                    ["Corn", "2020-01-05", 0.10, "", "", "", "", "", "", ""],
                    ["Corn", "2020-01-10", 0.20, "", "", "", "", "", "", ""],
                    ["Corn", "2020-01-15", 0.30, "", "", "Start", "", "", "", ""],
                    ["Corn", "2020-12-31", 0.40, "", "", "End", "", "", "", ""],
                    ["Rice", "2020-02-01", 0.15, 0.01, 0.03, "", "", "", "", ""],
                    ["Rice", "2020-02-10", "", 0.02, 0.04, "Start", "", "", "", ""],
                    ["Rice", "2020-11-20", 0.55, 0.11, 0.30, "End", "", "", "", ""],
                ],
            )

            records = build_tile_crop_records(
                base.parent.parent,
                crops=["Corn", "Rice"],
                years=[2020],
                feature_set="all_indices",
                min_points=2,
            )

            self.assertEqual(len(records), 2)
            corn = next(record for record in records if record.crop == "Corn")
            rice = next(record for record in records if record.crop == "Rice")
            self.assertEqual(corn.start_doy, 15)
            self.assertEqual(corn.end_doy, 366)
            self.assertEqual(rice.x.shape, (4, 73))

    def test_feature_options_and_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "outputs" / "2021_AR" / "0_2"
            base.mkdir(parents=True)
            write_test_workbook(
                base / "harvest_summary_all_crops.xlsx",
                rows=[
                    ["Rice", "2021-01-05", "", 0.10, 0.20, "", "", "", "", ""],
                    ["Rice", "2021-01-15", 0.30, "", "", "Start", "", "", "", ""],
                    ["Rice", "2021-11-20", 0.40, 0.20, 0.10, "End", "", "", "", ""],
                ],
            )

            ndvi_records = build_tile_crop_records(
                base.parent.parent,
                crops=["Rice"],
                years=[2021],
                feature_set="ndvi_only",
                min_points=2,
            )
            all_records = build_tile_crop_records(
                base.parent.parent,
                crops=["Rice"],
                years=[2021],
                feature_set="all_indices",
                min_points=2,
            )

            self.assertEqual(len(ndvi_records), 1)
            self.assertEqual(ndvi_records[0].x.shape, (2, 73))
            self.assertEqual(all_records[0].x.shape, (4, 73))

    def test_observations_to_tensor_fixed_length(self) -> None:
        tensor = observations_to_tensor(
            [
                (5, {"NDVI": 0.2, "NDWI": 0.1, "EVI": np.nan}),
                (33, {"NDVI": 0.5, "NDWI": np.nan, "EVI": 0.3}),
            ],
            feature_names=get_feature_names("all_indices"),
        )
        self.assertEqual(tensor.shape, (4, 73))
        self.assertGreater(tensor[-1, 0], 0.0)
        self.assertEqual(float(tensor[-1, 10]), 0.0)

    def test_model_forward_and_postprocess(self) -> None:
        model = TileCNNRegressor(in_channels=4)
        x = torch.randn(3, 4, 73)
        out = model(x)
        self.assertEqual(tuple(out.shape), (3, 2))
        doy = normalized_to_doy(out)
        self.assertEqual(doy.shape, (3, 2))
        self.assertTrue(np.all(doy[:, 0] <= doy[:, 1]))


def write_test_workbook(path: Path, rows: list[list[object]]) -> None:
    headers = ["Crop", "Date", "NDVI", "NDWI", "EVI", "Harvest", "start_rule", "div_start", "div_end", "IoU"]
    worksheet_rows = [headers, *rows]

    row_xml = []
    for row_idx, row in enumerate(worksheet_rows, start=1):
        cells = []
        for col_idx, value in enumerate(row, start=1):
            ref = f"{column_name(col_idx)}{row_idx}"
            if value == "":
                cells.append(f'<c r="{ref}" t="inlineStr"></c>')
                continue
            if isinstance(value, (int, float)):
                cells.append(f'<c r="{ref}" t="n"><v>{value}</v></c>')
            else:
                cells.append(
                    f'<c r="{ref}" t="inlineStr"><is><t>{value}</t></is></c>'
                )
        row_xml.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>'
        f'{"".join(row_xml)}'
        '</sheetData>'
        '</worksheet>'
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
            '</Types>',
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
            '</Relationships>',
        )
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>'
            '</workbook>',
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '</Relationships>',
        )
        zf.writestr(
            "docProps/core.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" />',
        )
        zf.writestr(
            "docProps/app.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" />',
        )
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def column_name(index: int) -> str:
    out = []
    while index:
        index, rem = divmod(index - 1, 26)
        out.append(chr(ord("A") + rem))
    return "".join(reversed(out))


if __name__ == "__main__":
    unittest.main()
