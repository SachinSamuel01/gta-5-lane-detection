# Lane Detection for Autonomous Driving Cars in GTA 5 using Computer Vision

This repository contains the code for a project that simulates lane detection for autonomous driving cars in Grand Theft Auto 5 using computer vision techniques. The implementation relies on Python and the OpenCV library for image processing.

[![Demo Video](http://img.youtube.com/vi/pNX4P7aewsw/0.jpg)](https://youtu.be/pNX4P7aewsw "Demo Video")

## Overview

The project aims to showcase a computer vision-based approach to detect and follow lane lines within the game environment of GTA 5. The primary language used for this project is Python, a popular language for machine learning and computer vision tasks.

## Key Features

1. **Screen Region Grabbing**: The PyWin module is used to grab the region of the screen for processing, which is then used to determine the lanes.
2. **Image Processing**: The OpenCV-Python module is employed to process the captured screen region.
3. **Canny Edge Detection**: The Canny edge detection technique is used to identify edges in the captured image.
4. **HoughLinesP Function**: This function provides the coordinates of lines detected within the image.
5. **Custom Function for Lane Lines**: A user-made function is implemented to remove redundant lines and identify the best lines to represent lane lines.
6. **Drawing Lane Lines**: The detected lane lines are then drawn onto the original image using the line function.

## Getting Started

### Prerequisites

To run this project, you will need the following libraries installed:

- PyWin
- OpenCV-Python

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SachinSamuel01/gta-5-lane-detection.git
```

2. Navigate to the project folder:
```bash
cd gta-5-lane-detection
```

3. Open .sln file using Visual Studio, install dependencies and run the project



## Acknowledgements

- [Grand Theft Auto 5](https://www.rockstargames.com/V/) - Simulation environment
- [OpenCV](https://opencv.org/) - Open-source computer vision library
- [Python](https://www.python.org/) - Programming language
