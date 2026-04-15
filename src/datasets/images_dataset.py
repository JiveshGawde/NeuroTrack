from collections import defaultdict
from typing import Literal, NotRequired, Optional, Tuple, TypedDict, Unpack
import torch
import re
from torchvision import transforms
import glob
import os
import cv2
from torch.utils.data import Dataset
from src.config import FULL_DATA_PATH
from src.types.AlzheimersDatasetTypes import AlzheimerDataSetAtom, AlzheimersDatasetFilters

MILD_DEMENTED_PATH: str = os.path.abspath(
    os.path.join(FULL_DATA_PATH, "MildDemented/MildDemented"))
MODERATE_DEMENTED_PATH: str = os.path.abspath(os.path.join(
    FULL_DATA_PATH, "ModerateDemented/ModerateDemented"))
NON_DEMENTED_PATH: str = os.path.abspath(
    os.path.join(FULL_DATA_PATH, "NonDemented (2)/NonDemented"))
VERYMILD_DEMENTED_PATH: str = os.path.abspath(os.path.join(
    FULL_DATA_PATH, "VeryMildDemented/VeryMildDemented"))

_FILENAME_RE = re.compile(
    r"OAS1_(?P<patient>\d+)_(?P<scan>MR\d+_\d+)\.nii_slice_(?P<slice>\d+)\.png"
)

LABEL_FOLDERS = [(0, NON_DEMENTED_PATH),
                 (1, VERYMILD_DEMENTED_PATH),
                 (2, MILD_DEMENTED_PATH),
                 (3, MODERATE_DEMENTED_PATH)]


class AlzheimersDatasetFilterKwargs(TypedDict):
    label: NotRequired[str | Literal['all', 'non',
                                     'demented', 'mild', 'very-mild', 'moderate']]
    patient_id: NotRequired[str]
    not_patient_ids: NotRequired[list[str]]
    not_slices: NotRequired[list[int]]
    slices: NotRequired[list[int]]
    patient_ids: NotRequired[list[str]]
    scan: NotRequired[str]
    scans: NotRequired[list[str]]
    not_scans: NotRequired[list[str]]
    slice_le: NotRequired[int]
    slice_ge: NotRequired[int]
    distinct_patients: NotRequired[bool]
    distinct_scans: NotRequired[bool]
    distinct_patients_strategy: NotRequired[str |
                                            Literal['first', 'middle', 'last']]


class AlzheimersDataset(Dataset):
    def __init__(self, n_images: int = 3, random_state: int = 0,
                 transform: Optional[transforms.Compose] = None):
        self.n_images = n_images
        self.random_state = random_state
        self.transform: Optional[transforms.Compose] = transform

        self.window = self.get_window(LABEL_FOLDERS, extension="png")

    def get_window(self, folders: list = LABEL_FOLDERS, extension: str = "png") -> list:

        group_images = defaultdict(list)
        window = []
        for label, folder in folders:
            for path in glob.glob(os.path.join(folder, f"*.{extension}")):
                parsed_data = self.parse_filename(path)
                if parsed_data is None:
                    continue
                patient, scan, slice = parsed_data
                group_images[(patient, scan, label)].append([slice, path])

        for (patient, scan, label), slices in group_images.items():
            slices.sort(key=lambda x: x[0])
            paths = [p for _, p in slices]

            for i in range(len(paths) - self.n_images + 1):
                window.append((paths[i: i + self.n_images], label))

        return window

    def get_unique_slices(self, folders: list = LABEL_FOLDERS, extension: str = "png") -> list:
        group_images = defaultdict(list)

        for label, folder in folders:
            for path in glob.glob(os.path.join(folder, f"*.{extension}")):
                parse_data = self.parse_filename(path)
                if parse_data is None:
                    continue

                patient, scan, slice = parse_data
                group_images[(patient, scan, label)].append((slice, path))

        result = []

        for (patient, scan, label), slices in group_images.items():
            slices.sort(key=lambda x: x[0])
            for slice_num, path in slices:
                result.append((patient, scan, slice_num, path, label))

        return result

    def __len__(self) -> int:
        return len(self.window)

    def __getitem__(self, index):
        paths, label = self.window[index]

        frames = []

        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if self.transform is not None:
                img = self.transform(img)

            img = transforms.ToTensor()(img[..., None])
            frames.append(img)

        return torch.stack(frames, dim=0), torch.tensor(label, dtype=torch.long)

    def parse_filename(self, path) -> Optional[Tuple[str, str, int]]:
        m = _FILENAME_RE.match(os.path.basename(path))
        if m is None:
            return None

        return m.group("patient"), m.group("scan"), int(m.group("slice"))

    def __asserts_filter(self, kwargs):
        assert not (
            'patient_id' in kwargs and 'patient_ids' in kwargs), "both patient_id and patient_ids cannot be selected, use either one"
        assert not (
            'scan' in kwargs and 'scans' in kwargs), "both scan and scans cannot be selected, use either one"
        assert not ('slices' in kwargs and ('slice_le' in kwargs or 'slice_ge' in kwargs)
                    ), "either use slices or use slice range (slice_le, slice_ge) not both"

        assert not (
            'patient_ids' in kwargs and 'not_patient_ids' in kwargs), "cannot use both patient_ids, and not_patient_ids, choose one"

        assert not (
            'patient_ids' in kwargs and kwargs.get('distinct_patients', False)
        ), "cannot use patient_ids with distinct_patients — patient_ids already restricts the patient set"

        assert not (
            'slices' in kwargs and 'not_slices' in kwargs), "cannot use both slices, and not_slices, choose one"
        assert not (
            'scans' in kwargs and 'not_scans' in kwargs), "cannot use both scans, and not_scans, choose one"

        assert not (
            'distinct_scans' in kwargs and not kwargs.get(
                'distinct_patients', False)
        ), "distinct_scans requires distinct_patients=True"
        assert not (
            'distinct_patients_strategy' in kwargs and not kwargs.get(
                'distinct_patients', False)
        ), "distinct_patients_strategy requires distinct_patients=True"

    def filter(self, extension: str = "png", **kwargs: Unpack[AlzheimersDatasetFilterKwargs]) -> AlzheimersDatasetFilters:
        """
        ACCEPTABLE ARGUMENTS:
        label: str
        slices: list[str]
        patient_id: str
        not_patient_ids: str
        patient_ids: list[str]
        slice_le: int
        slice_ge: int
        scan: str
        scans: list[str]
        not_scans: list[str]
        not_slices list[str]
        distinct_patients: bool
        distinct_scans: bool
        distinct_patients_strategy: str
        """

        self.__asserts_filter(kwargs)

        get_class: str = kwargs.get('label', 'all')
        filtered = self.get_unique_slices(
            get_match_label_folder(get_class), extension)

        use_distinct = kwargs.get('distinct_patients', False)
        has_slice_filter = any(k in kwargs for k in (
            'slices', 'slice_ge', 'slice_le', 'not_slices'))

        represent = {}
        if use_distinct and not has_slice_filter:
            strategy = kwargs.get('distinct_patients_strategy', 'middle')
            patients_slice_map = defaultdict(list)

            for patient, scan, slice_num, *_ in filtered:
                patients_slice_map[patient].append(slice_num)

            for patient, slice_nums in patients_slice_map.items():
                slice_nums.sort()
                if strategy == 'middle':
                    represent[patient] = slice_nums[len(slice_nums) // 2]
                elif strategy == 'first':
                    represent[patient] = slice_nums[0]
                else:
                    represent[patient] = slice_nums[-1]

        distinct_patients = set()

        result_filters = AlzheimersDatasetFilters()
        for patient, scan, slice_num, path, label in filtered:
            if 'not_patient_ids' in kwargs and patient in kwargs['not_patient_ids']:
                continue

            if 'not_slices' in kwargs and slice_num in kwargs['not_slices']:
                continue

            if 'not_scans' in kwargs and scan in kwargs['not_scans']:
                continue

            if 'patient_id' in kwargs and patient != kwargs['patient_id']:
                continue

            if 'patient_ids' in kwargs and patient not in kwargs['patient_ids']:
                continue

            if 'scan' in kwargs and scan != kwargs['scan']:
                continue

            if 'scans' in kwargs and scan not in kwargs['scans']:
                continue

            if 'slices' in kwargs and slice_num not in kwargs['slices']:
                continue

            if 'slice_le' in kwargs and slice_num > kwargs['slice_le']:
                continue

            if 'slice_ge' in kwargs and slice_num < kwargs['slice_ge']:
                continue

            if use_distinct:
                if not has_slice_filter:
                    if slice_num != represent[patient]:
                        continue
                track_key = patient if kwargs.get(
                    'distinct_scans', True) else (patient, scan)
                if track_key in distinct_patients:
                    continue
                distinct_patients.add(track_key)
            result_filters.append(AlzheimerDataSetAtom(patient_id=patient, scan_id=scan, slice_num=slice_num, path=path, label_str=get_match_label_str(label), label=label))

        return result_filters 


def get_match_label_folder(string: Literal['all', 'non', 'demented', 'mild', 'very-mild', 'moderate'] | str):
    match string.lower():
        case 'all':
            return LABEL_FOLDERS
        case 'non':
            return [(0, NON_DEMENTED_PATH)]
        case 'demented':
            return [(1, VERYMILD_DEMENTED_PATH),
                    (2, MILD_DEMENTED_PATH),
                    (3, MODERATE_DEMENTED_PATH)]
        case 'mild':
            return [(2, MILD_DEMENTED_PATH)]
        case 'very-mild':
            return [(3, VERYMILD_DEMENTED_PATH)]
        case 'moderate':
            return [(3, MODERATE_DEMENTED_PATH)]
        case _:
            return LABEL_FOLDERS


def get_match_label_str(number: int) -> Optional[str]:
    match number:
        case 0:
            return "non"
        case 1:
            return "very-mild"
        case 2:
            return "mild"
        case 3:
            return "moderate"

        case _:
            return None
