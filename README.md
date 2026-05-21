# Propicie: Automated Physical Fitness Assessment

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-v0.8+-green.svg)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An innovative system for the automated evaluation of physical fitness tests from the Fullerton Battery, designed to support active aging. This project leverages computer vision with a Kinect V2 sensor and Google's MediaPipe Holistic library to provide accurate, real-time measurements.

## About the Project

This project, developed as part of the PROPICIE IPBEJA & IFSC collaboration and contributing to the CAPACITA initiative, aims to automate the physical fitness assessment of older adults. By automating the Fullerton Functional Fitness Test Battery, we can collect objective data on flexibility and strength, which is crucial for monitoring physical decline and promoting personalized active aging programs.

The system focuses on two main assessments:
* **Sit-and-Reach Test** (`Sitting and reaching the feet with the hands`): measures the flexibility of the lower limbs.
* **Back-Scratch Test** (`Reaching the hands behind the back`): measures the flexibility of the upper limbs (shoulders).

The core of the project is a Python application that uses a Kinect V2 sensor to capture the user's movements and the MediaPipe Holistic framework to perform real-time detection of body landmarks. This approach allows for the accurate calculation of body angles for posture validation and of the key distances for scoring the tests.

### Key Findings
The research conducted by Artem Bukhantsev and further developed by me concluded that:
* The MediaPipe implementation demonstrated superior accuracy for the Sit-and-Reach test, with a Mean Absolute Error (MAE) of approximately 2.25 cm.
* This approach was significantly more accurate than a native PyKinect implementation, which presented an MAE of 8.65 cm, due to challenges such as virtual skeleton instability ("jittering").
* The Back-Scratch test proved to be a challenge for computer vision due to limb occlusion and the user's orientation with their back to the camera.

## Features

* **Real-Time Assessment**: automated analysis of the "Sit-and-Reach" and "Back-Scratch" exercises.
* **High-Precision Tracking**: uses MediaPipe Holistic for robust, real-time tracking of 33 pose landmarks, in addition to detailed hand landmarks.
* **Posture Validation**: calculates joint angles (knee, hip, elbow) to ensure the user is performing the exercise correctly before taking the measurement.
* **User Registration**: a simple interface to register the participant's data (age, height, weight, gender) before starting the tests.
* **Data Logging**: automatically saves the test results, including the calculated distance, the actual distance (for validation), and the measurement error, into Excel files (`.xlsx`) for later analysis.
* **Statistical Analysis**: includes Python scripts to analyze the collected data and compute key statistics on the measurement error.
* **Real-Time Feedback**: provides on-screen visualizations of the skeleton, key metrics, and instructions to guide the user.

## How It Works

The system follows a clear workflow for each assessment:
1.  **User Registration**: the user enters their demographic data.
2.  **Video Capture**: a Kinect V2 captures the user's video feed.
3.  **Landmark Detection**: the video is processed frame by frame. MediaPipe Holistic detects the user's body, hand, and face landmarks.
4.  **Calibration and Posture Verification**:
    * For the Sit-and-Reach test, the system validates the posture by checking whether the knee, hip, and elbow angles are within predefined limits (for example, the knee must be extended). Once the user maintains a valid calibration pose, the foot position is locked in as a reference.
    * For the Back-Scratch test, the system waits for the user to hold a stable pose with their hands behind their back.
5.  **Distance Measurement**: the Euclidean distance between the key landmarks (for example, fingertips to the calibrated foot position, or one hand's fingertips to the other's) is calculated in pixels and converted to centimeters. An error correction factor, derived from empirical testing, is applied to improve accuracy.
6.  **Result Display and Logging**: the final calculated distance is displayed on the screen, and the complete results are saved to a log file and to an Excel spreadsheet for the user group.

## Technologies Used

* **Language**: Python 3.8+
* **Computer Vision**: OpenCV, MediaPipe Holistic
* **Hardware**: Microsoft Kinect for Windows v2
* **Kinect SDK Wrapper**: PyKinect2
* **Data Handling and Analysis**: Pandas, NumPy
* **Orchestration**: the scripts can be run directly with Python (`runner.py`) or through a C# .NET Runner (`CsRunner/`).

## Setup and Installation

To run this project, follow the steps below.

### Prerequisites
* A computer running Windows 10/11 (required for the Kinect SDK).
* A Microsoft Kinect v2 sensor with its corresponding power adapter and USB 3.0 cable.
* A free USB 3.0 port.
* Python 3.8 (the Anaconda distribution is recommended).

### Installation Steps

1.  **Install the Kinect for Windows SDK 2.0**:
    * Download and install the SDK from Microsoft's official site: [Kinect for Windows SDK 2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44561).
    * Connect your Kinect sensor to the PC via USB 3.0 and to a power source. Verify that it is recognized in the Device Manager.

2.  **Set Up the Python Environment**:
    * It is highly recommended to use a virtual environment. With Anaconda, you can create one with:
        ```bash
        conda create -n propicie_env python=3.8
        conda activate propicie_env
        ```

3.  **Install the Required Libraries**:
    * Install the main dependencies using pip:
        ```bash
        pip install opencv-python mediapipe pandas numpy openpyxl
        ```

4.  **Install PyKinect2**:
    * `PyKinect2` requires a manual installation. Clone the official repository and run the setup script.
        ```bash
        git clone [https://github.com/Kinect/PyKinect2.git](https://github.com/Kinect/PyKinect2.git)
        cd PyKinect2
        python setup.py install
        ```
    * If you run into issues, you may need to install `comtypes`.

## Usage

Once setup is complete, you can run the assessments.

### Running the Full Test Suite
You can run the Sit-and-Reach and Back-Scratch tests sequentially using the provided runner script.

```bash
python runner.py
```

### Running Individual Tests
You can also run each test script individually:

* For the Sit-and-Reach Test:
    ```bash
    python ./Sit-and-Reach/sit_and_reach_holistic_2.py
    ```
* For the Back-Scratch Test:
    ```bash
    python ./Back-Scratch/back_scratch.py
    ```

### The Process
1.  When a script is started, a window will appear prompting for the user's information (Age, Height, Weight, Gender). Fill in the fields and press `Enter`.
2.  Next, a window will prompt for the actual measured distance. This is used for validation and error calculation. Enter the value and press `Enter`.
3.  The main application window will open, showing the Kinect camera feed with the MediaPipe skeleton overlay.
4.  Follow the on-screen instructions to position yourself correctly.
5.  The system will automatically detect when you are in the correct posture, hold the pose, and then compute the result.
6.  The result will be displayed, and you will be prompted to continue (`c`) or quit (`q`).

## Project Structure

```
.
├── /analises/              # Scripts and results for the statistical analysis of the data.
├── /Back-Scratch/          # Contains the Python script for the Back-Scratch test.
├── /CsRunner/              # A C# .NET project to run the Python scripts.
├── /relatorios/            # Detailed progress and final reports.
├── /Sit-and-Reach/         # Contains Python scripts for the Sit-and-Reach test.
├── /tabelas_testes/        # Test data spreadsheets.
├── /tabelas_utentes/       # Spreadsheets with data collected from user testing.
├── .gitignore              # Specifies files to be ignored by Git.
├── runner.py               # A simple Python script to run all tests.
└── README.md               # This file.
```


## Acknowledgements

* This work is part of a research collaboration between the Polytechnic Institute of Beja (IPBeja) and the Federal Institute of Santa Catarina (IFSC).
* This project contributes to the broader CAPACITA project, which aims to develop digital tools to assess and improve the physical capabilities of the elderly population.
