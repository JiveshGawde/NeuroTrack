from dataclasses import dataclass
from typing import Literal, Optional
from collections import defaultdict
from functools import reduce

GROUPBYFIELD = Literal['patient_id', 'scan_id', 'slice_num', 'path', 'label', 'label_str']


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
        return { k:len(value) for k, value in self.grouped.items() }
    def sum(self, field: str, /) -> dict[str, int]:
        return {k:reduce(lambda x,y: x + getattr(y, field), value, 0) for k, value in self.grouped.items()}
    def avg(self, field: str, /) -> dict[str, int | float]:
        return {k:reduce(lambda x,y: (x + getattr(y, field)), value, 0) / len(value) for k, value in self.grouped.items()}
    def first(self) -> dict[str, AlzheimerDataSetAtom]:
        return {k:value[0] for k,value in self.grouped.items()}
    def last(self) -> dict[str, AlzheimerDataSetAtom]:
        return {k:value[-1] for k,value in self.grouped.items()}
    def min(self, field: str, /) -> dict[str, int | float]:
        return {k: min(value,key= lambda x: getattr(x, field)) for k, value in self.grouped.items()}
    def max(self, field: str, /) -> dict[str, int | float]:
        return {k: max(value,key= lambda x: getattr(x, field)) for k, value in self.grouped.items()}


class AlzheimersDatasetFilters:
    def __init__(self):
        self.results: list[AlzheimerDataSetAtom] = []

    def append(self, atom: AlzheimerDataSetAtom):
        self.results.append(atom)

    def groupby(self, by: GROUPBYFIELD | list[GROUPBYFIELD]) -> AlzheimersDatasetGroupBy:
        group = defaultdict(list[AlzheimerDataSetAtom])

        for result in self.results:
            if isinstance(by, str):
                r = getattr(result, by)
                group[r].append(result)
            else:
                key = tuple(getattr(result, b) for b in by)
                group[key].append(result)

        return AlzheimersDatasetGroupBy(dict(group), by=by)
    def __len__(self):
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, index: int):
        return self.results[index]

