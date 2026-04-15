from dataclasses import dataclass
from typing import Literal, Optional

GROUPBYFIELD = Literal['patient_id', 'scan_id', 'slice_num', 'path', 'label', 'label_str']
class AlzheimersDatasetFilters:
    def __init__(self):
        self.results: list[AlzheimerDataSetAtom] = []

    def append(self, atom: AlzheimerDataSetAtom):
        self.results.append(atom)

    def groupby(self, by: GROUPBYFIELD | list[GROUPBYFIELD]) -> AlzheimersDatasetGroupBy:
        ...

    def __len__(self):
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, index: int):
        return self.results[index]


@dataclass
class AlzheimerDataSetAtom:
    patient_id: str
    scan_id: str
    slice_num: int
    path: str
    label: int
    label_str: Optional[str]


class AlzheimersDatasetGroupBy:
    def __init__(self, group_data: dict[str | tuple, list[AlzheimerDataSetAtom]], by: GROUPBYFIELD | list[GROUPBYFIELD]):
        self.grouped: dict[str | tuple, list[AlzheimerDataSetAtom]] = group_data
        self.by: GROUPBYFIELD | list[GROUPBYFIELD] = by


    def count(self) -> dict[str, int]:
        ...
    def sum(self, field: str, /) -> dict[str, int]:
        ...
    def avg(self, field: str, /) -> dict[str, int | float]:
        ...
    def first(self) -> dict[str, AlzheimerDataSetAtom]:
        ...
    def last(self) -> dict[str, AlzheimerDataSetAtom]:
        ...
    def min(self, field: str, /) -> dict[str, int | float]:
        ...
    def max(self, field: str, /) -> dict[str, int | float]:
        ...
