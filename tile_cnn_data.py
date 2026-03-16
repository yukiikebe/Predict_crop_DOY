from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Iterable, Optional
from xml.etree import ElementTree as ET
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset


YEAR_DIR_RE = re.compile(r"^(?P<year>\d{4})_[A-Za-z]+$")
WORKBOOK_NAME = "harvest_summary_all_crops.xlsx"
BIN_SIZE_DAYS = 5
NUM_BINS = 73

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "ndvi_only": ("NDVI",),
    "all_indices": ("NDVI", "NDWI", "EVI"),
}


@dataclass(frozen=True)
class TileCropRecord:
    year: int
    tile: str
    crop: str
    x: np.ndarray
    start_doy: Optional[int]
    end_doy: Optional[int]
    source_workbook: str
    feature_set: str
    num_observations: int


class TileCropDataset(Dataset):
    def __init__(self, records: Iterable[TileCropRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        target = np.array(
            [
                0.0 if record.start_doy is None else record.start_doy / 366.0,
                0.0 if record.end_doy is None else record.end_doy / 366.0,
            ],
            dtype=np.float32,
        )
        return {
            "x": torch.from_numpy(record.x.astype(np.float32)),
            "y": torch.from_numpy(target),
            "year": record.year,
            "tile": record.tile,
            "crop": record.crop,
            "start_doy": record.start_doy,
            "end_doy": record.end_doy,
            "source_workbook": record.source_workbook,
            "feature_set": record.feature_set,
            "num_observations": record.num_observations,
        }


def get_feature_names(feature_set: str) -> tuple[str, ...]:
    try:
        return FEATURE_SETS[feature_set]
    except KeyError as exc:
        raise ValueError(
            f"Unknown feature_set={feature_set!r}. Expected one of {sorted(FEATURE_SETS)}."
        ) from exc


def build_tile_crop_records(
    outputs_path: str | Path,
    *,
    crops: Optional[Iterable[str]] = None,
    years: Optional[Iterable[int]] = None,
    feature_set: str = "all_indices",
    min_points: int = 2,
    require_labels: bool = True,
) -> list[TileCropRecord]:
    outputs_path = Path(outputs_path)
    feature_names = get_feature_names(feature_set)
    crop_filter = set(crops) if crops is not None else None
    year_filter = set(int(year) for year in years) if years is not None else None

    records: list[TileCropRecord] = []
    for workbook_path, year, tile in iter_workbook_paths(outputs_path, years=year_filter):
        rows = read_all_crops_workbook(workbook_path)
        grouped: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            crop_name = str(row.get("Crop", "")).strip()
            if not crop_name:
                continue
            if crop_filter is not None and crop_name not in crop_filter:
                continue
            grouped.setdefault(crop_name, []).append(row)

        for crop_name, crop_rows in grouped.items():
            record = build_record_from_rows(
                crop_rows,
                year=year,
                tile=tile,
                crop=crop_name,
                source_workbook=workbook_path,
                feature_set=feature_set,
                feature_names=feature_names,
                min_points=min_points,
                require_labels=require_labels,
            )
            if record is not None:
                records.append(record)

    records.sort(key=lambda record: (record.crop, record.year, record.tile))
    return records


def iter_workbook_paths(
    outputs_path: str | Path,
    *,
    years: Optional[Iterable[int]] = None,
) -> list[tuple[Path, int, str]]:
    outputs_path = Path(outputs_path)
    if not outputs_path.exists():
        raise FileNotFoundError(f"Path not found: {outputs_path}")

    year_filter = set(int(year) for year in years) if years is not None else None
    items: list[tuple[Path, int, str]] = []

    year_dir = parse_year_dir(outputs_path)
    if year_dir is not None:
        if year_filter is None or year_dir in year_filter:
            for tile_dir in sorted(path for path in outputs_path.iterdir() if path.is_dir()):
                workbook_path = tile_dir / WORKBOOK_NAME
                if workbook_path.exists():
                    items.append((workbook_path, year_dir, tile_dir.name))
        return items

    for candidate in sorted(path for path in outputs_path.iterdir() if path.is_dir()):
        candidate_year = parse_year_dir(candidate)
        if candidate_year is None:
            continue
        if year_filter is not None and candidate_year not in year_filter:
            continue
        for tile_dir in sorted(path for path in candidate.iterdir() if path.is_dir()):
            workbook_path = tile_dir / WORKBOOK_NAME
            if workbook_path.exists():
                items.append((workbook_path, candidate_year, tile_dir.name))
    return items


def parse_year_dir(path: str | Path) -> Optional[int]:
    match = YEAR_DIR_RE.match(Path(path).name)
    if match is None:
        return None
    return int(match.group("year"))


def build_record_from_rows(
    rows: list[dict[str, object]],
    *,
    year: int,
    tile: str,
    crop: str,
    source_workbook: Path,
    feature_set: str,
    feature_names: tuple[str, ...],
    min_points: int,
    require_labels: bool,
) -> Optional[TileCropRecord]:
    start_doy = None
    end_doy = None
    observations: list[tuple[int, dict[str, float]]] = []

    for row in rows:
        date_str = str(row.get("Date", "")).strip()
        if not date_str:
            continue
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue

        doy = int(dt.strftime("%j"))
        harvest_flag = str(row.get("Harvest", "")).strip()
        if harvest_flag == "Start":
            start_doy = doy
        elif harvest_flag == "End":
            end_doy = doy

        feature_values = {
            feature_name: to_float(row.get(feature_name))
            for feature_name in feature_names
        }
        if any(np.isfinite(value) for value in feature_values.values()):
            observations.append((doy, feature_values))

    if require_labels and (start_doy is None or end_doy is None):
        return None
    if require_labels and start_doy is not None and end_doy is not None and start_doy > end_doy:
        return None
    if len(observations) < min_points:
        return None

    x = observations_to_tensor(observations, feature_names=feature_names)
    return TileCropRecord(
        year=year,
        tile=tile,
        crop=crop,
        x=x,
        start_doy=start_doy,
        end_doy=end_doy,
        source_workbook=str(source_workbook),
        feature_set=feature_set,
        num_observations=len(observations),
    )


def observations_to_tensor(
    observations: list[tuple[int, dict[str, float]]],
    *,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    feature_means = np.full((len(feature_names), NUM_BINS), np.nan, dtype=np.float32)
    valid_fraction = np.zeros(NUM_BINS, dtype=np.float32)

    for bin_index in range(NUM_BINS):
        bin_values = {feature_name: [] for feature_name in feature_names}
        for doy, feature_values in observations:
            if doy_to_bin_index(doy) != bin_index:
                continue
            for feature_name in feature_names:
                value = feature_values.get(feature_name, np.nan)
                if np.isfinite(value):
                    bin_values[feature_name].append(float(value))

        observed_features = 0
        for feature_idx, feature_name in enumerate(feature_names):
            values = bin_values[feature_name]
            if values:
                feature_means[feature_idx, bin_index] = float(np.mean(values))
                observed_features += 1
        valid_fraction[bin_index] = observed_features / len(feature_names)

    feature_channels = []
    for feature_idx in range(len(feature_names)):
        feature_channels.append(
            interpolate_inside_observed_range(feature_means[feature_idx])
        )

    feature_channels.append(valid_fraction)
    return np.stack(feature_channels, axis=0).astype(np.float32)


def doy_to_bin_index(doy: int) -> int:
    return min((int(doy) - 1) // BIN_SIZE_DAYS, NUM_BINS - 1)


def interpolate_inside_observed_range(values: np.ndarray) -> np.ndarray:
    result = values.astype(np.float32).copy()
    valid_mask = np.isfinite(result)
    valid_idx = np.flatnonzero(valid_mask)

    if len(valid_idx) >= 2:
        interior = np.arange(valid_idx[0], valid_idx[-1] + 1, dtype=np.int64)
        result[interior] = np.interp(interior, valid_idx, result[valid_idx]).astype(np.float32)

    result[~np.isfinite(result)] = 0.0
    return result


def to_float(value: object) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def read_all_crops_workbook(path: str | Path) -> list[dict[str, object]]:
    path = Path(path)
    with zipfile.ZipFile(path, "r") as zf:
        shared_strings = load_shared_strings(zf)
        sheet_xml = zf.read("xl/worksheets/sheet1.xml")

    root = ET.fromstring(sheet_xml)
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows = root.findall(".//main:sheetData/main:row", ns)
    if not rows:
        return []

    parsed_rows = [parse_sheet_row(row, shared_strings, ns) for row in rows]
    header_row = parsed_rows[0]
    max_col = max(header_row.keys(), default=-1)
    headers = [
        str(header_row.get(col_idx, "")).strip()
        for col_idx in range(max_col + 1)
    ]

    records: list[dict[str, object]] = []
    for parsed_row in parsed_rows[1:]:
        record: dict[str, object] = {}
        for col_idx, header in enumerate(headers):
            if not header:
                continue
            record[header] = parsed_row.get(col_idx, "")
        records.append(record)
    return records


def load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    shared_strings_path = "xl/sharedStrings.xml"
    if shared_strings_path not in zf.namelist():
        return []

    root = ET.fromstring(zf.read(shared_strings_path))
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    values = []
    for si in root.findall("main:si", ns):
        text_parts = [node.text or "" for node in si.findall(".//main:t", ns)]
        values.append("".join(text_parts))
    return values


def parse_sheet_row(
    row: ET.Element,
    shared_strings: list[str],
    ns: dict[str, str],
) -> dict[int, object]:
    parsed: dict[int, object] = {}
    for cell in row.findall("main:c", ns):
        ref = cell.attrib.get("r", "")
        col_index = column_letters_to_index(re.match(r"([A-Z]+)", ref).group(1)) if ref else 0
        parsed[col_index] = parse_cell_value(cell, shared_strings, ns)
    return parsed


def parse_cell_value(
    cell: ET.Element,
    shared_strings: list[str],
    ns: dict[str, str],
) -> object:
    cell_type = cell.attrib.get("t")

    if cell_type == "inlineStr":
        text_parts = [node.text or "" for node in cell.findall(".//main:t", ns)]
        return "".join(text_parts)

    value_node = cell.find("main:v", ns)
    if value_node is None or value_node.text is None:
        return ""

    raw_value = value_node.text
    if cell_type == "s":
        return shared_strings[int(raw_value)]
    if cell_type == "b":
        return raw_value == "1"

    try:
        number = float(raw_value)
    except ValueError:
        return raw_value

    if number.is_integer():
        return int(number)
    return number


def column_letters_to_index(column_letters: str) -> int:
    result = 0
    for char in column_letters:
        result = result * 26 + (ord(char) - ord("A") + 1)
    return result - 1
