The main.py script analyzes video frames located in the specified directory and its subdirectories, identifies faces in these frames, and saves the cropped face images and analysis results in an "analyzed" directory structure.

**Input Arguments:**
The script takes one input argument, input_dir, which is the path to the directory containing the video frames to be analyzed.

**Loading Configuration:**
The script loads the configuration from the gpu.config.yaml file using OmegaConf.

**Initializing FaceAnalyzer:**
The script creates an instance of FaceAnalyzer based on the loaded configuration.

**Analyzing and Saving Results:**
The script recursively searches the input_dir directory and analyzes all video files within it and its subdirectories, excluding the analyzed directory.
For each video file:
  It performs the analysis using FaceAnalyzer.
  Extracts the data and converts it to a serializable format.
  Saves cropped face images that are larger than 40x40 pixels.
  Saves the full images with annotated faces and the analysis results in JSON format.

**Results Directory Structure:**
The analysis results are saved in a subdirectory named analyzed, which is created within the input_dir.
Each subdirectory within input_dir has a corresponding subdirectory in analyzed, containing a faces subdirectory (for cropped face images) and an analysis_results.json file (for analysis results).

**Example Usage:**
python main.py <input_dir>
