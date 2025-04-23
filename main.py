import os
import sys
import base64
import traceback
from dataclasses import replace

import xml.etree.ElementTree as ET

import numpy as np

from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, PYDICOM_IMPLEMENTATION_UID

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import get_all_files, set_dcm_save_path, read_dicom, save_mrn_map_table
from ecg_dcm_metadata import *


class XMLFile:
    """
    A class for parsing and handling ECG-related XML data.
    This class extracts patient, test, and ECG waveform information from an XML file.
    """

    def __init__(self, xml_file_path, new_mrn=None, converted_cnt=0):
        """
        Initializes the XMLFile instance.

        Args:
            xml_file_path (str): Path to the XML file.
            new_mrn (str, optional): If provided, raw data loading is disabled.
        """
        self.xml_file_path = xml_file_path
        self.new_mrn = new_mrn
        self.load_raw = False if new_mrn is not None else True
        self.converted_cnt = converted_cnt

        self.uid = UID()
        self.prefix = PreFix()

        self.ecg_data = ECGData()
        self.patient_data = PatientData()
        self.test_data = TestData()

        self.waveform_sequence_cycle = WaveformSequence()
        self.waveform_sequence = WaveformSequence()
        self.channel_definition_sequence_cycle = ChannelDefinitionSequence()
        self.channel_definition_sequence = ChannelDefinitionSequence()

        self.attributes = {}

        self.patient_info = None
        self.test_info = None
        self.diagnosis_info = None
        self.waveform_info = None
        self.waveform_cycle_info = None
        self.waveform_whole_info = None

        # warning: lead_data should be used after multiplied by digital_scale_factor
        self.lead_data = None  # whole ecg data in digitized form
        self.lead_names = None
        self.lead_orders = None
        self.lead_cycle_data = None  # ecg cycle data in digitized form
        self.digital_scale_factor = 0.0

        self.retrieve_xml_data()

    @staticmethod
    def safe_find_text(element, path):
        found = element.find(path)
        return found.text if found is not None else None

    def retrieve_waveform_info(self, target='whole'):
        """
        Extracts and processes ECG waveform information from the XML data.

        Args:
            target (str): Specifies whether to retrieve 'whole' ECG waveform data
                          or 'cycle' (segmented ECG cycles).

        This function reads waveform data for each ECG lead, decodes and converts it
        into numerical values, and stores the results. If the target is 'whole',
        it also computes derived leads (III, aVR, aVL, aVF) based on standard ECG
        derivations.
        """

        def compute_derived_leads(lead_dict):
            """
            Computes derived ECG leads (III, aVR, aVL, aVF) based on standard formulas.

            Args:
                lead_dict (dict): Dictionary containing raw lead waveform data.
                lead_dict (dict): Dictionary containing raw lead waveform data.

            Returns:
                dict: Dictionary containing both original and derived lead data.
            """
            for derived in self.ecg_data.derived_leads:
                if derived not in lead_data.keys():
                    if derived == 'III':
                        lead_data['III'] = lead_data['II'] - lead_data['I']
                    elif derived == 'aVR':
                        lead_data['aVR'] = -(lead_data['I'] + lead_data['II']) / 2
                    elif derived == 'aVL':
                        lead_data['aVL'] = lead_data['I'] - lead_data['II'] / 2
                    elif derived == 'aVF':
                        lead_data['aVF'] = lead_data['II'] - lead_data['I'] / 2

            # sort the lead_data by lead name
            lead_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

            sorted_lead_dict = {lead: lead_dict[lead] for lead in lead_order if lead in lead_dict}

            return sorted_lead_dict

        lead_data = {}

        info = self.waveform_whole_info if target == 'whole' else self.waveform_cycle_info
        num_waveform_channels = int(self.safe_find_text(info, 'NumberofLeads'))

        # SamplingFrequency = SampleBase×10^(SampleExponent)
        sample_base = int(self.safe_find_text(info, 'SampleBase'))
        sample_exponent = int(self.safe_find_text(info, 'SampleExponent'))
        sampling_frequency = sample_base * (10 ** sample_exponent)

        lead_cnt = 0
        for lead in info:
            if lead.tag != 'LeadData':
                continue

            sample_count = int(self.safe_find_text(lead, 'LeadSampleCountTotal'))
            duration = sample_count / sampling_frequency
            lead_sample_size = int(self.safe_find_text(lead, 'LeadSampleSize'))
            waveform_bits_allocated = lead_sample_size * 8

            high_pass_filter = str(float(self.safe_find_text(info, 'HighPassFilter')))
            low_pass_filter = str(float(self.safe_find_text(info, 'LowPassFilter')))

            if lead_cnt == 0:
                # Waveform Sequence/Channel Definition Sequence/Channel Sensitivity Correction Factor
                sensitivity_correction_factor = float(self.safe_find_text(lead, 'LeadAmplitudeUnitsPerBit'))
                # Waveform Sequence/Channel Definition Sequence/Channel Sensitivity Units Sequence
                amp_units = self.safe_find_text(lead, 'LeadAmplitudeUnits')
                if 'micro' in amp_units.lower():
                    code_value, code_meaning, scheme_designator = 'uV', 'microvolt', 'UCUM'
                elif 'milli' in amp_units.lower():
                    code_value, code_meaning, scheme_designator = 'mV', 'millivolt', 'UCUM'
                else:
                    raise ValueError(f'Invalid amplitude unit: {amp_units}')
                self.ecg_data = replace(self.ecg_data,
                                        sampling_frequency=sampling_frequency,
                                        sequence_length_in_seconds=duration,
                                        # amp_units_per_bit=amp_units_per_bit,
                                        num_waveform_channels=num_waveform_channels,
                                        num_waveform_samples=sample_count)

                multiplex_group_label = self.safe_find_text(lead, 'WaveformType')
                channel_sample_skew = self.safe_find_text(lead, 'LeadOffsetFirstSample')

                if target == 'cycle':
                    self.channel_definition_sequence_cycle.sensitivity_units_sequence = replace(
                        self.channel_definition_sequence_cycle.sensitivity_units_sequence,
                        code_value=code_value,
                        code_meaning=code_meaning,
                        scheme_designator=scheme_designator)
                    self.channel_definition_sequence_cycle = replace(self.channel_definition_sequence_cycle,
                                                                     lowpass_filter=low_pass_filter,
                                                                     highpass_filter=high_pass_filter,
                                                                     sensitivity_correction_factor=sensitivity_correction_factor,
                                                                     skew=channel_sample_skew)

                    self.waveform_sequence_cycle = replace(self.waveform_sequence_cycle,
                                                           originality='DERIVED',
                                                           num_channels=num_waveform_channels,
                                                           num_samples=sample_count,
                                                           sampling_frequency=sampling_frequency,
                                                           bits_allocated=waveform_bits_allocated,
                                                           multiplex_group_label=multiplex_group_label)
                else:
                    self.channel_definition_sequence.sensitivity_units_sequence = replace(
                        self.channel_definition_sequence.sensitivity_units_sequence,
                        code_value=code_value,
                        code_meaning=code_meaning,
                        scheme_designator=scheme_designator)
                    self.channel_definition_sequence = replace(self.channel_definition_sequence,
                                                               lowpass_filter=low_pass_filter,
                                                               highpass_filter=high_pass_filter,
                                                               sensitivity_correction_factor=sensitivity_correction_factor,
                                                               skew=channel_sample_skew)

                    self.waveform_sequence = replace(self.waveform_sequence,
                                                     originality='ORIGINAL',
                                                     num_channels=num_waveform_channels,
                                                     num_samples=sample_count,
                                                     sampling_frequency=sampling_frequency,
                                                     bits_allocated=waveform_bits_allocated,
                                                     multiplex_group_label=multiplex_group_label)

            lead_name = self.safe_find_text(lead, 'LeadID')
            if lead_name in self.ecg_data.expected_leads:
                waveform_data = base64.b64decode(self.safe_find_text(lead, 'WaveFormData'))
                lead_waveform = np.frombuffer(waveform_data, dtype='<i2', count=sample_count)
                lead_data[lead_name] = lead_waveform

            lead_cnt += 1

        if target == 'whole':
            self.lead_data = compute_derived_leads(lead_data)
            self.lead_names = self.lead_data.keys()
            self.lead_orders = np.arange(len(self.lead_names)) + 1
            self.ecg_data = replace(self.ecg_data, waveform_channel_count=len(self.lead_data.keys()))
            self.waveform_sequence = replace(self.waveform_sequence,
                                             num_channels=len(self.lead_data.keys()))
        else:
            self.lead_cycle_data = lead_data
            self.waveform_sequence_cycle = replace(self.waveform_sequence_cycle,
                                                   num_channels=len(self.lead_cycle_data.keys()))

    def retrieve_patient_info(self):
        """
        Extracts and formats patient information from the XML data.

        This function retrieves the patient's name, ID, age, and gender, formatting them
        according to standardized conventions. If a new MRN (Medical Record Number) is provided,
        it replaces the original patient ID.

        Updates:
            - `self.patient_data` with formatted patient details.
        """

        def format_age(age_str):
            """
            Formats the patient's age into a standardized format (e.g., '025Y' for 25 years old).

            Args:
                age_str (str or None): The age string extracted from XML.

            Returns:
                str: A zero-padded, three-digit age string followed by 'Y' (e.g., '025Y').
            """
            if age_str is None:
                reformat = '000Y'
            else:
                reformat = f'{int(age_str):03}Y'
            return reformat

        def format_gender(gender_str):
            """
            Converts patient gender into a standardized format.

            Args:
                gender_str (str or None): The gender string extracted from XML.

            Returns:
                str: 'M' for male, 'F' for female, 'O' for other/unknown.
            """
            if gender_str is None:
                reformat = 'O'
            elif gender_str.lower() == 'male':
                reformat = 'M'
            else:
                reformat = 'F'

            return reformat

        patient_first_name = self.safe_find_text(self.patient_info, 'PatientFirstName')
        patient_last_name = self.safe_find_text(self.patient_info, 'PatientLastName')
        patient_first_name = patient_first_name if patient_first_name is not None else ''
        patient_last_name = patient_last_name if patient_last_name is not None else ''
        patient_name = patient_last_name + '^' + patient_first_name
        patient_age = format_age(self.safe_find_text(self.patient_info, 'PatientAge'))
        patient_id = self.safe_find_text(self.patient_info, 'PatientID')
        patient_gender = format_gender(self.safe_find_text(self.patient_info, 'Gender'))

        if self.load_raw:
            self.patient_data = replace(self.patient_data,
                                        name=patient_name,
                                        id=patient_id,
                                        age=patient_age,
                                        sex=patient_gender)
        else:
            self.patient_data = replace(self.patient_data,
                                        id=self.new_mrn,
                                        age=patient_age,
                                        sex=patient_gender)

    def retrieve_test_info(self):
        """
        Extracts and formats test (acquisition) information from the XML data.

        This function retrieves test acquisition and study timestamps, formats them
        into standardized formats, and updates the `self.test_data` structure accordingly.

        Updates:
            - `self.test_data` with formatted test acquisition details.
        """

        def format_date(date_str):
            """
            Formats a date string from 'MM-DD-YYYY' to 'YYYYMMDD'.

            Args:
                date_str (str or None): The date string extracted from XML.

            Returns:
                str: Reformatted date string (YYYYMMDD) or '000000' if anonymized.
            """
            reformat = datetime.strptime(date_str, '%m-%d-%Y').strftime('%Y%m%d') if date_str is not None else None
            if not self.load_raw:
                # replace date information of reformat with zeros
                reformat = reformat[:6] + '01' if reformat is not None else '000000'

            return reformat

        def format_time(time_str):
            """
            Formats a time string by removing colons (e.g., '12:34:56' → '123456').

            Args:
                time_str (str or None): The time string extracted from XML.

            Returns:
                str: Reformatted time string (HHMMSS) or '000000' if anonymized.
            """
            reformat = time_str.replace(':', '') if time_str is not None else None
            if not self.load_raw:
                # replace reformat with zeros
                reformat = '000000' if reformat is not None else '000000'

            return reformat

        acquisition_date = format_date(self.safe_find_text(self.test_info, 'AcquisitionDate'))
        acquisition_time = format_time(self.safe_find_text(self.test_info, 'AcquisitionTime'))
        study_date = format_date(self.safe_find_text(self.test_info, 'EditDate'))
        study_time = format_time(self.safe_find_text(self.test_info, 'EditTime'))
        content_date = format_date(self.safe_find_text(self.test_info, 'EditDate'))
        content_time = format_time(self.safe_find_text(self.test_info, 'EditTime'))

        manufacture_model_name = self.safe_find_text(self.test_info, 'AcquisitionDevice')
        software_version = self.safe_find_text(self.test_info, 'AcquisitionSoftwareVersion')
        # institution_name = self.safe_find_text(self.test_info, 'SiteName')
        # station_name = self.safe_find_text(self.test_info, 'StationName')[:16]
        # current_patient_location = self.safe_find_text(self.test_info, 'CurrentPatientLocation')[:30]

        # referring_physician_name not found in xml
        # Study Instance UID (0020,000D) is generated by the function
        # Study ID (0020,0010) is generated by the function
        study_id = generate_uid()[-16:]

        self.test_data = replace(self.test_data,
                                 acquisition_date=acquisition_date,
                                 acquisition_time=acquisition_time,
                                 study_date=study_date,
                                 study_time=study_time,
                                 content_date=content_date,
                                 content_time=content_time,
                                 study_id=study_id,
                                 manufacture_model_name=manufacture_model_name,
                                 software_version=software_version)

    def retrieve_xml_data(self) -> None:
        """
        Parses the ECG XML file and extracts patient, test, diagnosis, and waveform information.
        """
        tree = ET.parse(self.xml_file_path)
        root = tree.getroot()

        # Extract patient-related information
        self.patient_info = root.find('.//PatientDemographics')
        self.retrieve_patient_info()

        # Extract waveform-related information
        self.waveform_info = root.findall('.//Waveform')

        # Check if at least one waveform segment exists
        if len(self.waveform_info) > 0:
            for info in self.waveform_info:
                waveform_type = self.safe_find_text(info, 'WaveformType')
                if waveform_type == 'Median':
                    # Cycle-level waveform
                    self.waveform_cycle_info = info
                else:
                    # Whole waveform
                    self.waveform_whole_info = info
                    # Retrieve high-pass and low-pass filter values from waveform data
                    high_pass_filter = str(float(self.safe_find_text(self.waveform_whole_info, 'HighPassFilter')))
                    low_pass_filter = str(float(self.safe_find_text(self.waveform_whole_info, 'LowPassFilter')))

                    # Update ECG data with extracted filter values
                    self.ecg_data = replace(self.ecg_data,
                                            lowpass_filter=low_pass_filter,
                                            highpass_filter=high_pass_filter)

        else:
            raise ValueError('No waveform data found in the XML file.')

        # Retrieve detailed waveform information for whole and cycle segments
        self.retrieve_waveform_info(target='whole')
        self.retrieve_waveform_info(target='cycle')

        # Extract test-related information
        self.test_info = root.find('.//TestDemographics')
        self.retrieve_test_info()

        # Extract diagnosis-related information
        self.diagnosis_info = root.find('.//Diagnosis')

        # Compile extracted attributes into a dictionary
        self.collate_attr_dict()

    def collate_attr_dict(self):
        """
        Organizes extracted attributes into a structured dictionary for further processing.
        """
        self.attributes = {
            "Modality": self.prefix.Modality,
            "SpecificCharacterSet": self.prefix.specific_character_set,

            "PatientName": self.patient_data.name,
            "PatientID": self.patient_data.id,
            "PatientSex": self.patient_data.sex,
            "PatientAge": self.patient_data.age,
            "PatientBirthDate": self.patient_data.birth_date,

            "AcquisitionDateTime": self.test_data.acquisition_date + self.test_data.acquisition_time,
            "StudyDate": self.test_data.study_date,
            "StudyTime": self.test_data.study_time,
            "StudyID": self.test_data.study_id,
            "AccessionNumber": self.test_data.accession_number,
            "ContentDate": self.test_data.content_date,
            "ContentTime": self.test_data.content_time,

            # Dummy Value for InstanceNumber and SeriesNumber
            "InstanceNumber": f"{self.converted_cnt:04d}",
            "SeriesNumber": f"{self.converted_cnt:04d}",

            "InstitutionName": self.test_data.institution_name,
            "StationName": self.test_data.station_name,
            # "CurrentPatientLocation": self.test_data.current_patient_location,

            "OperatorsName": self.test_data.operator_name,
            "NameOfPhysiciansReadingStudy": self.test_data.physician_name,
            "ReferringPhysicianName": self.test_data.referring_physician_name,

            "SOPClassUID": self.uid.twelve_lead_ecg_sop_class,
            "StudyInstanceUID": generate_uid(self.uid.study_class_uid),
            "SeriesInstanceUID": generate_uid(self.uid.series_class_uid),
            "SOPInstanceUID": generate_uid(self.uid.instance_class_uid),

            "Manufacturer": self.prefix.Manufacturer,
            "ManufacturerModelName": self.test_data.manufacture_model_name,
            "SoftwareVersions": self.test_data.software_version,

            "StudyDescription": self.prefix.StudyDescription
        }


def create_waveform_sequence_item(waveform_meta, lead_data, channel_def_seq):
    """
        Creates a single DICOM Waveform Sequence Item from the provided waveform metadata,
        lead data, and channel definition information.

        Parameters:
            waveform_meta: An object containing waveform metadata.
            lead_data: A dictionary where keys are lead names (e.g., 'I', 'II', 'V1', ...) and values are
                       lists or arrays of signal data.
            channel_def_seq: An object containing channel definition parameters shared across all leads,
                             such as sensitivity, filters, baseline, etc.

        Returns:
            A pydicom Dataset object representing a single item in the (5400, 0100) Waveform Sequence.
    """
    waveform_sequence_item = Dataset()
    waveform_sequence_item.WaveformOriginality = waveform_meta.originality
    waveform_sequence_item.NumberOfWaveformChannels = waveform_meta.num_channels
    waveform_sequence_item.NumberOfWaveformSamples = waveform_meta.num_samples
    waveform_sequence_item.SamplingFrequency = waveform_meta.sampling_frequency
    waveform_sequence_item.WaveformBitsAllocated = waveform_meta.bits_allocated
    waveform_sequence_item.WaveformSampleInterpretation = waveform_meta.sample_interpretation

    waveform_array = np.array(list(lead_data.values()), dtype=np.int16).T
    waveform_sequence_item.WaveformData = waveform_array.tobytes()

    waveform_sequence_item.ChannelDefinitionSequence = []
    for k, _ in lead_data.items():
        # *(003A, 0200) Channel Definition Sequence
        channel_def = Dataset()
        # **(003A, 0210) Channel Sensitivity
        channel_def.ChannelSensitivity = channel_def_seq.sensitivity
        # **(003A, 0215) Channel Sample Skew
        channel_def.ChannelSampleSkew = channel_def_seq.skew
        # **(003A, 021A) Waveform Bits Stored
        channel_def.WaveformBitsStored = channel_def_seq.bits_stored
        # **(3A, 0212) Channel Sensitivity Correction Factor
        channel_def.ChannelSensitivityCorrectionFactor = channel_def_seq.sensitivity_correction_factor
        # **(003A, 0220) Filter Low Frequency
        channel_def.FilterLowFrequency = channel_def_seq.lowpass_filter
        # **(003A,0221) Filter High Frequency
        channel_def.FilterHighFrequency = channel_def_seq.highpass_filter
        # **(003A,0213) Channel Baseline
        channel_def.ChannelBaseline = channel_def_seq.channel_baseline

        # **(003A, 0208) Channel Source Sequence
        channel_def.ChannelSourceSequence = [Dataset()]
        source = channel_def.ChannelSourceSequence[0]
        # ***(0008, 0100) Code Value
        source.CodeValue = k
        # ***(0008, 0102) Coding Scheme Designator
        source.CodingSchemeDesignator = channel_def_seq.source_sequence.scheme_designator
        # ***(0008, 0104) Code Meaning
        source.CodeMeaning = ' '.join(['Lead', k])

        # **(003A, 0211) Channel Sensitivity Units Sequence
        channel_def.ChannelSensitivityUnitsSequence = [Dataset()]
        unit = channel_def.ChannelSensitivityUnitsSequence[0]
        # ***(0008, 0100) Code Value
        unit.CodeValue = channel_def_seq.sensitivity_units_sequence.code_value
        # ***(0008, 0102) Coding Scheme Designator
        unit.CodeMeaning = channel_def_seq.sensitivity_units_sequence.code_meaning
        # ***(0008, 0102) Coding Scheme Designator
        unit.CodingSchemeDesignator = channel_def_seq.sensitivity_units_sequence.scheme_designator
        waveform_sequence_item.ChannelDefinitionSequence.append(channel_def)

    return waveform_sequence_item


def create_dicom_file(xml_file_path,
                      de_identified_mrn=None,
                      rid_index=None,
                      output_folder='ecg_dcm',
                      converted_cnt=0):
    """
    Converts an XML file containing ECG data into a DICOM file format. The function
    checks for the necessary XML elements, parses patient and test data, and uses this
    information to construct a DICOM file with ECG waveform data.

    Input:
        xml_file_path: str - The path to the XML file containing the ECG data.
        de_identified_mrn: str - The de-identified Medical Record Number (MRN) of the patient based on the Medical Record Number(MRN) of the patient.
        output_folder: str, optional - The directory where the generated DICOM file will be saved.
                                      Defaults to 'generated_dicom_from_xml'.

    Output:
        None - The DICOM file is written to the specified output folder. If the file already
               exists, the function will skip writing and print a message. If there is an error
               during processing, it will print an error message.
    """

    # Check if the output folder exists, create if not
    os.makedirs(output_folder, exist_ok=True)

    try:
        output_file_path = set_dcm_save_path(source_path=xml_file_path,
                                             target_path=output_folder,
                                             de_mrn=de_identified_mrn,
                                             rid_index=rid_index)

        # Skip if DICOM file already exists
        if os.path.exists(output_file_path):
            print(f'File {output_file_path} already exists. {xml_file_path} Skipping...')
            return

        xml_data = XMLFile(xml_file_path=xml_file_path,
                           new_mrn=de_identified_mrn,
                           converted_cnt=converted_cnt)

        # Create Meta data for Dicom file
        file_meta = Dataset()
        # https://dicom.nema.org/dicom/2000/draft/00_06dr.pdf (page 67)
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.9.1.1'  # Standard 12-Lead ECG
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        _, file_name = os.path.split(output_file_path)

        ds = FileDataset(file_name, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        # Set DICOM properties extracted from xml
        for attr, value in xml_data.attributes.items():
            if value is not None:
                setattr(ds, attr, value)

        # Concept Name Code Sequence (0040, A043)
        acq_context_seq = Sequence()

        # Lead System
        lead_system_item = Dataset()
        lead_system_item.ValueType = 'CODE'
        concept_name_code_seq = Dataset()
        concept_name_code_seq.CodeValue = '10:11345'
        concept_name_code_seq.CodingSchemeDesignator = 'MDC'
        concept_name_code_seq.CodeMeaning = 'Lead System'
        lead_system_item.ConceptNameCodeSequence = Sequence([concept_name_code_seq])

        # https://dicom.nema.org/medical/dicom/current/output/html/part16.html#sect_CID_3263
        concept_code_seq = Dataset()
        concept_code_seq.CodeValue = '10:11265'
        concept_code_seq.CodingSchemeDesignator = 'MDC'
        concept_code_seq.CodeMeaning = 'Standard 12-lead positions, electrodes placed individually'
        lead_system_item.ConceptCodeSequence = Sequence([concept_code_seq])

        acq_context_seq.append(lead_system_item)

        # Patient State
        patient_state_item = Dataset()
        patient_state_item.ValueType = 'CODE'
        concept_name_code_seq = Dataset()
        concept_name_code_seq.CodeValue = '109054'
        concept_name_code_seq.CodingSchemeDesignator = 'DCM'
        concept_name_code_seq.CodeMeaning = 'Patient State'
        patient_state_item.ConceptNameCodeSequence = Sequence([concept_name_code_seq])

        concept_code_seq = Dataset()
        concept_code_seq.CodeValue = '128975004'
        concept_code_seq.CodingSchemeDesignator = 'SCT'
        concept_code_seq.CodeMeaning = 'Resting state'
        patient_state_item.ConceptCodeSequence = Sequence([concept_code_seq])

        acq_context_seq.append(patient_state_item)

        # AcquisitionContextSequence
        ds.AcquisitionContextSequence = acq_context_seq

        # Create waveform sequence item for DICOM
        waveform_sequence = Sequence()
        # Whole waveform
        waveform_sequence.append(
            create_waveform_sequence_item(
                waveform_meta=xml_data.waveform_sequence,
                lead_data=xml_data.lead_data,
                channel_def_seq=xml_data.channel_definition_sequence
            )
        )

        # Cycle waveform
        waveform_sequence.append(
            create_waveform_sequence_item(
                waveform_meta=xml_data.waveform_sequence_cycle,
                lead_data=xml_data.lead_cycle_data,
                channel_def_seq=xml_data.channel_definition_sequence_cycle
            )
        )
        ds.WaveformSequence = waveform_sequence
        # ds.WaveformSequence = Sequence([waveform_sequence_item])

        # Save DICOM file
        ds.save_as(output_file_path, write_like_original=False)

        # Save raw xml data
        os.system(f'cp {xml_file_path} {output_folder}')

        return output_file_path

    except Exception as e:
        print(f'Error processing file {xml_file_path}: {e}')
        print(traceback.format_exc())

        return None


if __name__ == "__main__":
    import os
    from tqdm import tqdm

    # Example usage
    debug = True
    debug_n = 5

    root_xml_path = 'path_to_ecg_xml'
    mrn_map_table_save_path = os.getcwd()
    converted_dcm_save_path = 'ecg_dcm'

    ecg_xml_list = get_all_files(root_xml_path, 'xml')

    count = 0
    shifted_patient_dict = {}
    for ecg_xml in tqdm(ecg_xml_list):
        # add dict mrn as key, rid and examination date as value
        mrn = ecg_xml.split('/')[-1].split('_')[1]
        rid = ecg_xml.split('/')[-1].split('_')[0]
        examination_date = ecg_xml.split('/')[-1].split('_')[2].split('.')[0]

        # if mrn already exists, add values to existing key, else create new key
        if mrn in shifted_patient_dict.keys():
            existing_values = shifted_patient_dict[mrn]
            de_identified_mrn = existing_values[0][0]
            new_values = [de_identified_mrn, rid, examination_date]
            shifted_patient_dict[mrn].append([de_identified_mrn, rid, examination_date])
            rid_index = f'{len(shifted_patient_dict[mrn]):06d}'
        else:
            de_identified_mrn = f'{count:06d}'
            new_values = [de_identified_mrn, rid, examination_date]
            shifted_patient_dict[mrn] = [new_values]
            rid_index = f'{len(shifted_patient_dict[mrn]):06d}'

        dicom_path = create_dicom_file(ecg_xml, de_identified_mrn, rid_index, converted_dcm_save_path, count)

        if dicom_path is None:
            continue

        dicom_data = read_dicom(dicom_path)

        count += 1
        if debug:
            if count == debug_n:
                print(ecg_xml)
                save_mrn_map_table(target_dict=shifted_patient_dict)
                break

    save_mrn_map_table(target_dict=shifted_patient_dict)
    breakpoint()