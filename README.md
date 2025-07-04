# Webcam Weapon Detection System

## Description

This project is a Python-based weapon detection system that uses your webcam to identify potential weapons in real-time. It leverages the YOLOv3 (You Only Look Once) object detection model and OpenCV for video processing. The system is configurable, allowing users to specify which categories of weapons to detect and adjust the detection confidence threshold.

## Features

*   **Real-time Webcam Detection**: Analyzes live feed from your webcam.
*   **YOLOv3 Model**: Utilizes the powerful YOLOv3 object detection algorithm.
*   **Configurable Weapon Categories**: Users can enable/disable predefined weapon categories (e.g., cutting weapons, blunt weapons, firearms, potential weapons, dangerous tools).
*   **Custom Categories**: Ability to define and detect custom object categories.
*   **Confidence Threshold**: Adjustable confidence level for detections.
*   **Detection Saving**: Option to save frames where weapons are detected.
*   **Detection Logging**: Logs detection events with timestamps and details.
*   **Custom Model Support**: Users can specify paths to their own YOLOv3 compatible model weights, config, and names files.
*   **Easy Setup**: A simple command to download necessary model files and set up directories.
*   **Command-Line Interface**: Manage and run the detection system via CLI arguments.

## Requirements

*   Python 3.x
*   OpenCV (`cv2`): `pip install opencv-python`
*   NumPy: `pip install numpy`

The YOLOv3 model files (`yolov3.weights`, `yolov3.cfg`) and class names (`coco.names`) will be automatically downloaded during the setup process if they are not already present in the `models/` and `data/` directories respectively.

## Setup

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install opencv-python numpy
    ```
    (If you have a `requirements.txt` file, you can use `pip install -r requirements.txt`)

3.  **Run the project setup script:**
    This command will create necessary directories (`models`, `data`, `logs`, `output`, `config`) and download the default YOLOv3 model files and COCO names file if they don't exist. It will also create a default configuration file.
    ```bash
    python weapon_detection.py --setup
    ```

## Usage

### Starting Webcam Detection

To start detecting weapons using your primary webcam:

```bash
python weapon_detection.py --webcam
```

Press the `ESC` key while the webcam window is active to stop the detection.

### Command-Line Arguments

The script offers several command-line arguments to customize its behavior:

*   `--setup`: Run the initial project setup (create directories, download models).
*   `--webcam`: Start webcam detection.
*   `--no-save`: Disable saving of frames when a weapon is detected. By default, saving is enabled.
*   `--show-categories`: Display the detected weapon's category along with its class name on the webcam feed.
*   `--confidence <float>`: Set the confidence threshold for detections (e.g., `0.6`). Value should be between 0.0 and 1.0. This updates the `config/weapon_detection_config.json` file.
*   `--custom-model <name>`: Use a custom YOLO model. Specify the base name of the model files (e.g., if you have `my_model.weights`, `my_model.cfg`, `my_model.names`, use `--custom-model my_model`). These files should be placed in the `models/` and `data/` directories accordingly.

#### Category Management:

*   `--list-categories`: List all available predefined weapon categories and the classes they include.
*   `--list-enabled`: List the weapon categories currently enabled in the configuration.
*   `--enable-category <category_name>`: Enable a specific weapon category (e.g., `--enable-category firearms`).
*   `--disable-category <category_name>`: Disable a specific weapon category.
*   `--add-custom-category <category_name>`: Define a new custom category name.
*   `--custom-classes <comma_separated_classes>`: Specify the object classes for the new custom category (e.g., `--custom-classes "lighter,matchbox"`). Use with `--add-custom-category`.

**Example of adding and enabling a custom category:**
```bash
python weapon_detection.py --add-custom-category "incendiaries" --custom-classes "lighter,matchbox"
# This automatically enables the 'incendiaries' category.
# You can verify with: python weapon_detection.py --list-enabled
```

## Configuration

The primary configuration for the weapon detection system is stored in `config/weapon_detection_config.json`. You can modify this file directly or use the command-line arguments to update certain settings.

```json
{
    "enabled_categories": [
        "cutting_weapons",
        "blunt_weapons",
        "potential_weapons",
        "firearms"
    ],
    "confidence_threshold": 0.5,
    "custom_classes": {
        "my_custom_category": ["object1", "object2"]
    }
}
```

*   **`enabled_categories`**: A list of strings specifying which weapon categories are active. These can be predefined categories or custom category names you've added.
*   **`confidence_threshold`**: A float between 0.0 and 1.0. Detections with confidence below this threshold will be ignored.
*   **`custom_classes`**: A dictionary where keys are your custom category names and values are lists of object class names belonging to that category.

## Output

*   **Detected Images**: If saving is enabled (default), frames where weapons are detected are saved as JPEG images in the `output/` directory. Filenames include a timestamp (e.g., `weapon_detected_YYYYMMDD_HHMMSS.jpg`).
*   **Detection Logs**: Details of each detection event are appended to `logs/detection_log.txt`. This includes the timestamp, detected classes, confidence scores, and the path to the saved image.

## Weapon Categories

The system comes with the following predefined weapon categories and their associated COCO classes:

*   **`cutting_weapons`**: `knife`, `scissors`
*   **`blunt_weapons`**: `baseball bat`, `sports ball` (note: "sports ball" can be a proxy for other blunt objects depending on context)
*   **`potential_weapons`**: `bottle`, `wine glass`, `cup`, `fork`, `backpack` (items that could potentially be used as weapons or conceal them)
*   **`firearms`**: `gun`, `pistol`, "rifle", "handgun", "firearm" (these are often custom trained or specific classes in some YOLO versions; the default COCO `gun` might not exist, but `handgun`, `rifle` etc. might be part of a custom model or specific COCO versions. The script maps common terms.)
    *   *Note: The default COCO dataset used by YOLOv3 might not explicitly have a generic "gun" class. Detection of specific firearms often requires models trained on datasets that include them. The system attempts to map common firearm-related terms.*
*   **`dangerous_tools`**: `hammer`, `drill`, `saw`, `axe` (tools that can be used as weapons)
    *   *Note: Some of these classes (e.g., "hammer", "drill", "saw", "axe") might not be present in the standard COCO dataset. For detecting these, a custom-trained model or a model trained on a more extensive dataset would be necessary.*

You can list these using `python weapon_detection.py --list-categories`.

## Troubleshooting

*   **Webcam Not Found/Cannot Open Webcam**:
    *   Ensure your webcam is properly connected and not being used by another application.
    *   If you have multiple webcams, the script defaults to camera index `0`. You might need to modify `cv2.VideoCapture(0)` in `weapon_detection.py` if your desired camera has a different index (e.g., `cv2.VideoCapture(1)`).
*   **Model Download Failure**:
    *   If the automatic download via `--setup` fails (e.g., due to network issues or broken links), the script will print the URLs. You can download these files manually:
        *   `yolov3.weights`: from `https://pjreddie.com/media/files/yolov3.weights` (place in `models/`)
        *   `yolov3.cfg`: from `https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg` (place in `models/`)
        *   `coco.names`: from `https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names` (place in `data/`)
*   **Low Detection Accuracy / False Positives / Missed Detections**:
    *   Adjust the `--confidence` threshold. A lower threshold increases sensitivity (more detections, potentially more false positives). A higher threshold reduces sensitivity (fewer false positives, potentially more missed detections).
    *   Lighting conditions, camera quality, and object occlusion can significantly impact detection performance.
    *   The default YOLOv3 model trained on COCO has limitations. For highly accurate detection of specific types of weapons or tools not well-represented in COCO, a custom-trained model is recommended.

## Contributing (Optional)

Contributions are welcome! If you have ideas for improvements or find bugs, please feel free to:
1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes.
4.  Submit a pull request.

---

This README provides a good starting point. Depending on the project's complexity and future development, more sections like "License", "Acknowledgements", or "Detailed Algorithm Explanation" could be added.
