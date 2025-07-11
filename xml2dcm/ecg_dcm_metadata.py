from datetime import datetime
from dataclasses import dataclass, field
from pydicom.uid import generate_uid
from enum import Enum


ANONYMOUS = 'Anonymized'


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
    sampling_frequency: int = 500 # SampleBase
    sequence_length_in_seconds: int = 10
    num_waveform_samples: int = 0
    num_waveform_channels: int = 0
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
    source_sequence: ChannelSourceSequence = field(default_factory=ChannelSourceSequence)
    sensitivity_units_sequence: ChannelSensitivityUnitsSequence = field(default_factory=ChannelSensitivityUnitsSequence)

    sensitivity: int = 1
    skew: str = "0"
    bits_stored: int = 16
    sensitivity_correction_factor: int = 4.88
    lowpass_filter: str = 20.0  # LowPassFilter
    highpass_filter: str = 8.0  # HighPassFilter
    channel_baseline = "0.0"


@dataclass
class WaveformSequence:
    originality: str = 'ORIGINAL'
    num_channels: int = 12
    num_samples: int = 5000
    sampling_frequency: int = 500
    bits_allocated: int = 16
    sample_interpretation: str = 'SS'
    multiplex_group_label: str = 'whole'


@dataclass
class MeasurementUnitsCodeSequence:
    # (0040,08EA)
    code_value: str or None = None
    code_meaning: str or None = None
    scheme_designator: str or None = None


@dataclass
class ConceptNameCodeSequence:
    # (0040,A043)
    code_value: str or None = None
    code_meaning: str or None = None
    scheme_designator: str or None = None


@dataclass
class WaveformAnnotationSequence:
    measurement_units_code_sequence: MeasurementUnitsCodeSequence = field(default_factory=MeasurementUnitsCodeSequence)
    concept_name_code_sequence: ConceptNameCodeSequence = field(default_factory=ConceptNameCodeSequence)
    numeric_value: str = None
    referenced_waveform_channels = [1, 0]


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
    birth_date: str = '' # Type 2: if not provided, empty string
    race: str = 'Anonymized'


@dataclass
class TestData:
    datatype: str = 'RESTING'
    site: str = 'UNKNOWN'
    acquisition_date: str = '00000000'
    acquisition_time: str = '000000'
    study_date: str = '00000000'
    study_time: str = '000000'
    study_id: str = '0000000000000000'

    # Tag: 	(0008,0050)
    accession_number: str = '' # Type 2: if not provided, empty string

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
    # Tag: 	(0008, 1040)
    institutional_department_name: str = 'UNKNOWN'

    # Tag: (0008, 1070)
    operator_name: str = 'UNKNOWN'
    # Tag: (0008, 1060)
    physician_name: str = 'UNKNOWN'
    # Tag: (0008, 0090)
    referring_physician_name: str = 'UNKNOWN'


@dataclass
class DiagnosisData:
    diagnosis: str = ''


@dataclass
class DeIdentification:
    name: str = 'Anonymized'
    place: str = 'Anonymized'
    date: str = '000000'
    time: str = '000000'


class Measurement(Enum):
    # (code value, code_meaning, scheme_designator)
    VentricularRate = ('2:16016', 'Ventricular Heart Rate', 'MDC')
    AtrialRate = ('2:16020', 'Atrial Heart Rate', 'MDC')
    PRInterval = ('2:15872', 'PR interval global', 'MDC')
    QRSDuration = ('2:16156', 'QRS duration global', 'MDC')
    QTInterval = ('2:16160', 'QT interval global', 'MDC')
    QTCorrected = ('2:15876', 'QTc interval global', 'MDC')

    PAxis = ('8626-4', 'P wave Axis', 'LOINC')
    RAxis = ('9997-8', 'R wave axis', 'LOINC')
    TAxis = ('8638-9', 'T wave Axis', 'LOINC')

    POnset = ('18511-6', 'P wave onset', 'LOINC')
    POffset = ('18512-4', 'P wave offset', 'LOINC')
    TOffset = ('18515-7', 'T wave offset', 'LOINC')

    # QTcFrederica = ('QTcFrederica', 'milliseconds', 'ms', 'UCUM')
    # QRSCount = ('QRSCount', None, None, None)
    # QOnset = ('QOnset', None, None, None)
    # QOffset = ('QOffset', None, None, None)


class MeasurementUnit(Enum):
    # (xml attribute name, code value, code_meaning, scheme_designator)
    VentricularRate = ('VentricularRate', 'beats per minute', '/min', 'UCUM')
    AtrialRate = ('AtrialRate', 'beats per minute', '/min', 'UCUM')
    PRInterval = ('PRInterval', 'milliseconds', 'ms', 'UCUM')
    QRSDuration = ('QRSDuration', 'milliseconds', 'ms', 'UCUM')
    QTInterval = ('QTInterval', 'milliseconds', 'ms', 'UCUM')
    QTCorrected = ('QTCorrected', 'milliseconds', 'ms', 'UCUM')

    PAxis = ('PAxis', 'degrees', 'deg', 'UCUM')
    RAxis = ('RAxis', 'degrees', 'deg', 'UCUM')
    TAxis = ('TAxis', 'degrees', 'deg', 'UCUM')

    POnset = ('POnset', 'milliseconds', 'ms', 'UCUM')
    POffset = ('POffset', 'milliseconds', 'ms', 'UCUM')
    TOffset = ('TOffset', 'milliseconds', 'ms', 'UCUM')

    # QTcFrederica = ('QTcFrederica', 'milliseconds', 'ms', 'UCUM')
    # QRSCount = ('QRSCount', None, None, None)
    # QOnset = ('QOnset', None, None, None)
    # QOffset = ('QOffset', None, None, None)