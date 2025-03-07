# XML2DCM-ECG

## Description

This repository contains a Python-based solution for converting ECG data from XML format to the DICOM (Digital Imaging
and Communications in Medicine) format. The goal of this project is to facilitate the process of transforming clinical
ECG data stored in XML files into a standardized, widely-used DICOM format, which is essential for healthcare systems,
medical imaging devices, and Electronic Health Record (EHR) systems.

The solution handles the extraction and conversion of ECG information, including patient details, acquisition context,
and waveform data, from XML files into the DICOM format. The project also integrates essential metadata such as the type
of ECG lead (e.g., Lead I, Lead II), device information (e.g., acquisition device, software versions), and patient
demographics.

## Installation

1. Clone this repository to your local machine using the following command:
   ```
   bash
   
   git clone https://github.com/MedxEng/XML2DCM-ECG.git
   ```

2. Install the required Python packages using the following command:
   ```
    bash
   
    pip install -r requirements.txt
    ```
   
