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


def set_dcm_save_path(source_path, target_path, de_mrn=None, extension='dcm'):
    if de_mrn is not None:
        output_dcm_file_name = f"{de_mrn}_{os.path.basename(source_path).split('_')[2].split('.')[0]}.{extension}"
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