from datetime import datetime
from dataclasses import dataclass, field
from pydicom.uid import generate_uid

@dataclass
class PreFix:
    DCM_UID: str = '1.2.840.10008.1.2.1'
    TWELVE_LEAD_ECG_SOP_CLASS_UID: str = '1.2.840.10008.5.1.4.1.1.9.1.1'
    GE_DCM_OID: str = '1.2.840.113619'
    Manufacturer: str = 'GE Healthcare'
    StudyDescription: str = '12-Lead ECG'
    Modality: str = 'ECG'
    specific_character_set: str = "ISO_IR 100"


@dataclass
class ECGData:
    sampling_rate: int = 500 # SampleBase
    sequence_length_in_seconds: int = 10
    waveform_length: int = sampling_rate * sequence_length_in_seconds
    waveform_channel_count: int = 0
    expected_leads: list = field(default_factory=lambda: ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
    derived_leads: list = field(default_factory=lambda: ['III', 'aVR', 'aVL', 'aVF'])
    lowpass_filter: str = 20.0 # LowPassFilter
    highpass_filter: str = 8.0 # HighPassFilter
    lead_time_offset: int = 0
    amp_units_per_bit: float = 4.88


@dataclass
class ChannelSourceSequence:
    code_value: str = 'I'
    scheme_designator: str = 'LOINC'
    code_meaning: str = 'Lead I'

@dataclass
class ChannelSensitivityUnitsSequence:
    code_value: str = 'uV'
    code_meaning: str = 'microvolt'
    scheme_designator: str = 'UCUM'

@dataclass
class ChannelDefinitionSequence:
    source_sequence: ChannelSourceSequence = ChannelSourceSequence()
    sensitivity_units_sequence: ChannelSensitivityUnitsSequence = ChannelSensitivityUnitsSequence()

    sensitivity: int = 10
    skew: int = "0"
    bits_stored: int = 16
    sensitivity_correction_factor: int = 4.88
    filter_low="20.0"
    filter_high="8.0"

@dataclass
class WaveformSequence:
    originality: str = 'ORIGINAL'
    num_channels: int = 12
    num_samples: int = 5000
    sample_rate: int = 500
    bits_allocated: int = 16
    sample_interpretation: str = 'SS'

@dataclass
class UID:
    ge_dcm_oid: str = '1.2.840.113619'
    twelve_lead_ecg_sop_class: str = '1.2.840.10008.5.1.4.1.1.9.1.1'

    current_date = datetime.now().strftime('%Y%m%d')

    study_class_uid = ge_dcm_oid + '.' + current_date + '.'
    series_class_uid = study_class_uid + '1.'
    instance_class_uid = series_class_uid + '1.'

    study_instance = generate_uid(study_class_uid)
    series_instance = generate_uid(series_class_uid)
    instance_instance = generate_uid(instance_class_uid)

@dataclass
class PatientData:
    id: str = 'Anonymized'
    name: str = 'Anonymized'
    age: str = '000Y'
    sex: str = 'M'
    birth_date: str = '00000000'

@dataclass
class TestData:
    datatype: str = 'RESTING'
    site: str = 'UNKNOWN'
    acquisition_date: str = '00000000'
    acquisition_time: str = '000000'
    study_date: str = '00000000'
    study_time: str = '000000'
    study_id: str = generate_uid()[-16:]

    # Tag: 	(0008,0050)
    accession_number: str = '0000000000000000'

    content_date: str = '00000000'
    content_time: str = '000000'
    # Tag: (0008, 1090)
    manufacture_model_name: str = 'UNKNOWN'
    # Tag: (0018, 1020)
    software_version: str = 'UNKNOWN'
    # Tag: (0008, 0080)
    institution_name: str = 'UNKNOWN'
    # Tag: (0008, 1010)
    station_name: str = 'UNKNOWN'
    # Tag: (0008, 0210)
    current_patient_location: str = 'UNKNOWN'

    # Tag: (0008, 1070)
    operator_name: str = 'UNKNOWN'
    # Tag: (0008, 1060)
    physician_name: str = 'UNKNOWN'
    # Tag: (0008, 0090)
    referring_physician_name: str = 'UNKNOWN'

@dataclass
class DiagnosisData:
    modality: str = 'ECG'


@dataclass
class DeIdentification:
    name: str = 'Anonymized'
    place: str = 'Anonymized'
    date: str = '000000'
    time: str = '000000'
