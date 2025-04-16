# EEG Classifier Application
This README provides instructions for setting up and running the EEG Classifier application.

# Prerequisites
Before running the application, ensure you have the following installed:

Python 3.7 or higher: You can download it from python.org.
.venv: A virtual environment for Python packages.
Microsoft C++ Build Tools: Download from Visual Studio.
CMake: Download from cmake.org.

#Installation Steps
1. Set Up a Virtual Environment
Create a virtual environment for the project: 
[python -m venv venv]

2. Install Required Python Packages
Install the necessary packages using pip:
[pip install flask flask-cors llama-cpp-python]

3. Download the Model
Download the model from the provided Google Drive link and place it in the following directory:
#C:\Users\user\Documents\EEG_Project\model\
Make sure the model file is named unsloth.Q4_K_M.gguf.

4. Running the Application
To run the application, execute the following command in your terminal: 
[python localhost_server.py]

# Usage
You can send EEG data for classification through the /classify endpoint.
Access classification history via the /history endpoint.
Clear the history using the /clear_history endpoint.
Perform a health check with the /health endpoint.

# Troubleshooting
Ensure all paths are correctly set, especially for the model file.
Check console logs for any errors related to model loading or server issues.
