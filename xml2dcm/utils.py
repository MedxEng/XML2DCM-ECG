import os
import json

import pydicom

def get_all_files(path, ext=None, include=None, exclude=None, return_full_path=True):
    """
    Get all files in the directory
    :param path: str, directory path
    :param ext: str, file extension
    :param include: str, file name to include
    :param exclude: str, file name to exclude
    :param return_full_path: bool, return full path or not
    :return: list, file list
    """
    if ext is not None:
        return [
            os.path.join(path, x) if return_full_path else x
            for x in os.listdir(path)
            if x.endswith(ext) and (exclude is None or exclude not in x)
        ]
    else:
        return [x for x in os.listdir(path) if exclude is None or exclude not in x]


def get_all_files_recursive(path, ext=None, exclude=None):
    """
    Get all files in the directory recursively
    :param path: str, directory path
    :param ext: str, file extension
    :param exclude: str, file name to exclude
    :return: list, file list
    """
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if ext is not None:
                if file.endswith(ext) and (exclude is None or exclude not in file):
                    # if file.endswith(ext) and (exclude is None or file.split('.')[0] not in exclude):
                    file_list.append(os.path.join(root, file))
            else:
                if exclude is None or exclude not in file:
                    file_list.append(os.path.join(root, file))
    return file_list


def set_dcm_save_path(source_path, target_path, pattern=None, pattern_values=None, extension='dcm'):
    """
    Sets the DICOM save path based on the source path, target path, and optional pattern.
    If a pattern is provided, it formats the output file name using the pattern and pattern values.

    Input:
        source_path: str - The path to the source DICOM file.
        target_path: str - The directory where the DICOM file should be saved.
        pattern: str or None - Optional pattern for formatting the output file name.
        pattern_values: dict or None - Values to format the pattern with, if a pattern is provided.
        extension: str - The file extension for the output DICOM file (default is 'dcm').

    Output:
        str - The full path where the DICOM file will be saved, including the formatted file name.
    """
    if pattern is not None:
        assert pattern_values is not None, "pattern_values must be provided if pattern is specified"

        if 'seq' in pattern_values:
            pattern_values['seq'] = str(pattern_values['seq']).zfill(5)

        output_dcm_file_name = f"{pattern.format(**pattern_values)}.{extension}"
    else:
        output_dcm_file_name = f"{os.path.splitext(os.path.basename(source_path))[0]}.{extension}"

    return os.path.join(target_path, output_dcm_file_name)


def read_dicom(dicom_path):
    """
    Reads a DICOM file from the specified path and returns the DICOM dataset. This function
    is primarily used for accessing the underlying data within a DICOM file which includes
    metadata and potentially images or other medical information.

    Input:
        dicom_path: str - The file path to the DICOM file that is to be read.

    Output:
        dicom_data: FileDataset - The DICOM dataset object containing all the data stored in
                    the DICOM file. This includes patient information, imaging or waveform data,
                    and metadata such as study details and technical parameters.
    """
    dicom_data = pydicom.dcmread(dicom_path)

    return dicom_data


def save_mrn_map_table(target_dict, target_path=None) -> None:
    """
    Save the MRN mapping table to a CSV file for reference. This function is used to store
    the mapping of original MRN values to de-identified MRN values for future reference.
    Args:
        target_path(str): The path to the json file where the MRN mapping table will be saved.
        target_dict(dict): The dictionary containing the mapping of original MRN values to de-identified MRN values.

    Returns:
        None
    """

    if target_path is None:
        target_path = os.getcwd()

    target_file_name = os.path.join(target_path, 'mrn_mapping_table.json')

    if not os.path.exists(target_file_name):
        with open(target_file_name, 'w') as f:
            json.dump(target_dict, f)
            print(f'MRN mapping table saved to {target_file_name}')

    return None