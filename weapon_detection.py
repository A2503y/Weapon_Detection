import os
import cv2
import numpy as np
import time
import argparse
import urllib.request
import json
from datetime import datetime

# Define weapon categories and corresponding COCO classes
WEAPON_CATEGORIES = {
    "cutting_weapons": ["knife", "scissors"],
    "blunt_weapons": ["baseball bat", "sports ball"],
    "potential_weapons": ["bottle", "wine glass", "cup", "fork", "backpack"],
    "firearms": ["gun", "pistol", "rifle", "handgun", "firearm"],  # Added dedicated firearms category
    "dangerous_tools": ["hammer", "drill", "saw", "axe"]  # Moved gun to the firearms category
}

# Define config file structure for weapon types
DEFAULT_CONFIG = {
    "enabled_categories": ["cutting_weapons", "blunt_weapons", "potential_weapons", "firearms"],  # Added firearms to default enabled categories
    "confidence_threshold": 0.5,
    "custom_classes": {}
}

# Output directory path - UPDATED to Documents folder
OUTPUT_DIR = "output"  # <<<< MODIFIED LINE

def setup_project():
    """
    Create project directory structure and download required files
    """
    # Create project directories
    directories = ['models', 'data', 'logs', 'config']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # Download required files if they don't exist
    files = {
        'models/yolov3.weights': 'https://pjreddie.com/media/files/yolov3.weights',
        'models/yolov3.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
        'data/coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
    }

    for file_path, url in files.items():
        if not os.path.exists(file_path):
            print(f"Downloading {file_path}...")
            try:
                urllib.request.urlretrieve(url, file_path)
                print(f"Downloaded {file_path}")
            except Exception as e:
                print(f"Error downloading {file_path}: {e}")
                print(f"Please manually download from {url} and place in {file_path}")

    # Create default config file if it doesn't exist
    config_path = 'config/weapon_detection_config.json'
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
            print(f"Created default configuration file: {config_path}")

def load_config():
    """
    Load configuration from config file or create with defaults
    """
    config_path = 'config/weapon_detection_config.json'
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config

def update_config(key, value):
    """
    Update configuration file with new settings
    """
    config_path = 'config/weapon_detection_config.json'
    config = load_config()
    config[key] = value

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Updated configuration: {key} = {value}")

def get_weapon_classes(config):
    """
    Get the list of weapon classes based on enabled categories.
    Combines predefined and custom classes.
    """
    weapon_classes_set = set() # Use a set to automatically handle duplicates

    enabled_categories = config.get("enabled_categories", [])
    custom_classes_map = config.get("custom_classes", {})

    for category_name in enabled_categories:
        # Check predefined categories
        if category_name in WEAPON_CATEGORIES:
            weapon_classes_set.update(WEAPON_CATEGORIES[category_name])

        # Check custom categories
        if category_name in custom_classes_map:
            weapon_classes_set.update(custom_classes_map[category_name])

    return list(weapon_classes_set) # Convert back to list

def load_yolo(model_path=None, config_path=None, names_path=None):
    """
    Load YOLO model for object detection
    """
    model_path = model_path or "models/yolov3.weights"
    config_path = config_path or "models/yolov3.cfg"
    names_path = names_path or "data/coco.names"

    net = cv2.dnn.readNet(model_path, config_path)
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    try:
        # OpenCV versions >= 4.5.4
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        # OpenCV versions < 4.5.4
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, classes, output_layers

def save_detection(frame, detections):
    """
    Save frame with detection to output directory
    """
    if not detections:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_file = os.path.join(OUTPUT_DIR, f"weapon_detected_{timestamp}.jpg")

    # Save the frame with detection
    cv2.imwrite(image_file, frame)
    print(f"Detection saved to: {image_file}")

    # Log the detection details
    log_file = os.path.join('logs', "detection_log.txt")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] Detected: {', '.join(d['class'] for d in detections)}\n")
        f.write(f"  Image saved: {image_file}\n")
        for d in detections:
            f.write(f"  {d['class']}: Confidence {d['confidence']:.2f}\n")
        f.write("\n")

def webcam_detection(custom_model=None, enable_saving=True, display_categories=False):
    """
    Detect weapons using webcam
    """
    # Load configuration
    config = load_config()
    confidence_threshold = config.get("confidence_threshold", 0.5)

    # Load appropriate model
    if custom_model:
        model_path = f"models/{custom_model}.weights"
        config_path = f"models/{custom_model}.cfg"
        names_path = f"data/{custom_model}.names"
        net, classes, output_layers = load_yolo(model_path, config_path, names_path)
    else:
        net, classes, output_layers = load_yolo()

    # Get weapon classes based on enabled categories
    weapon_classes = get_weapon_classes(config)

    # Colors for visualization - use different colors for different categories
    np.random.seed(42)  # For consistent colors
    colors = {}
    for category, class_list in WEAPON_CATEGORIES.items():
        category_color = np.random.uniform(0, 255, size=3).tolist()
        for weapon_class in class_list:
            colors[weapon_class] = category_color

    # Handle custom classes
    for category, class_list in config.get("custom_classes", {}).items():
        category_color = np.random.uniform(0, 255, size=3).tolist()
        for weapon_class in class_list:
            colors[weapon_class] = category_color

    # Set up video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam. Please check your camera connection.")
        return

    print("Webcam detection started. Press ESC to exit.")

    # Variables for detection cooldown
    last_save_time = 0
    save_cooldown = 2  # seconds between saves

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Exiting...")
            break

        frame_count += 1
        # Skip frames for performance
        if frame_count % 3 != 0:  # Process every 3rd frame
            cv2.imshow("Weapon Detection", frame)
            if cv2.waitKey(1) == 27:  # ESC key
                break
            continue

        detections = process_frame(frame, net, classes, colors, output_layers, weapon_classes,
                                confidence_threshold, display_categories)

        # Save detection with cooldown
        current_time = time.time()
        if enable_saving and detections and (current_time - last_save_time) >= save_cooldown:
            save_detection(frame, detections)
            last_save_time = current_time

        cv2.imshow("Weapon Detection", frame)
        if cv2.waitKey(1) == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam detection stopped.")

def process_frame(frame, net, classes, colors, output_layers, weapon_classes, confidence_threshold=0.5, display_categories=False):
    """
    Process a single frame for weapon detection
    Returns a list of detections
    """
    height, width, _ = frame.shape

    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to display
    class_ids = []
    confidences = []
    boxes = []
    detected_classes = []

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Check if the detected object is in our weapon classes and meets confidence threshold
            if confidence > confidence_threshold and class_id < len(classes) and classes[class_id] in weapon_classes:
                # Object detected is potentially a weapon
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                detected_classes.append(classes[class_id])

    # Apply non-maximum suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    # Draw bounding boxes
    font = cv2.FONT_HERSHEY_SIMPLEX
    weapon_detected = False
    detections = []

    # Ensure indexes is iterable (for compatibility with different OpenCV versions)
    if isinstance(indexes, np.ndarray):
        indexes = indexes.flatten()
    elif isinstance(indexes, tuple) and len(indexes) > 0:
        indexes = indexes[0]
    else:
        indexes = []

    # Find weapon category for each detection
    categories_found = set()

    for i in range(len(boxes)):
        if i in indexes:
            weapon_detected = True
            x, y, w, h = boxes[i]
            class_name = classes[class_ids[i]]
            confidence = confidences[i]

            # Find which category this weapon belongs to
            category = "Unknown"
            for cat_name, cat_classes in WEAPON_CATEGORIES.items():
                if class_name in cat_classes:
                    category = cat_name
                    categories_found.add(category)
                    break

            # Get color for this class or use a default color
            color = colors.get(class_name, [0, 255, 255])

            # Draw the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Show category if requested
            label = f"{class_name}: {confidence:.2f}"
            if display_categories:
                label = f"{category} - {label}"

            cv2.putText(frame, label, (x, y - 10), font, 0.5, color, 2)

            # Add to detections list
            detections.append({
                'class': class_name,
                'category': category,
                'confidence': confidence,
                'box': [x, y, w, h]
            })

    # Add alert text for weapon detection
    if weapon_detected:
        alert_text = "WEAPON DETECTED!"
        # Add categories if showing categories
        if display_categories and categories_found:
            categories_str = ", ".join(categories_found)
            alert_text += f" ({categories_str})"

        cv2.putText(frame, alert_text, (10, 30), font, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No weapons detected", (10, 30), font, 1, (0, 255, 0), 2)

    return detections

def list_weapon_categories():
    """
    Print all available weapon categories and their classes
    """
    print("\nAvailable Weapon Categories:")
    for category, classes in WEAPON_CATEGORIES.items():
        print(f"  {category}:")
        for weapon_class in classes:
            print(f"    - {weapon_class}")
    print("\nNote: Some classes may not be detected by the default YOLO model.")

def manage_categories(add=None, remove=None, list_enabled=False):
    """
    Manage enabled weapon categories
    """
    config = load_config()
    enabled_categories = config.get("enabled_categories", [])

    if list_enabled:
        print("\nCurrently Enabled Categories:")
        for category in enabled_categories:
            print(f"  {category}")
            if category in WEAPON_CATEGORIES:
                for weapon_class in WEAPON_CATEGORIES[category]:
                    print(f"    - {weapon_class}")
        return

    if add:
        if add not in WEAPON_CATEGORIES and add not in config.get("custom_classes", {}):
            print(f"Category '{add}' not found. Use --list-categories to see available categories.")
            return

        if add not in enabled_categories:
            enabled_categories.append(add)
            update_config("enabled_categories", enabled_categories)
            print(f"Enabled category: {add}")
        else:
            print(f"Category '{add}' is already enabled.")

    if remove:
        if remove in enabled_categories:
            enabled_categories.remove(remove)
            update_config("enabled_categories", enabled_categories)
            print(f"Disabled category: {remove}")
        else:
            print(f"Category '{remove}' is not currently enabled.")

def add_custom_category(category_name, classes):
    """
    Add a custom category with specified classes
    """
    config = load_config()
    custom_classes = config.get("custom_classes", {})

    # Add or update custom category
    custom_classes[category_name] = classes.split(',')
    update_config("custom_classes", custom_classes)

    # Automatically enable the new category
    enabled_categories = config.get("enabled_categories", [])
    if category_name not in enabled_categories:
        enabled_categories.append(category_name)
        update_config("enabled_categories", enabled_categories)

    print(f"Added custom category '{category_name}' with classes: {classes}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webcam Weapon Detection System")

    # Main commands
    parser.add_argument('--setup', action='store_true', help='Setup project and download required files')
    parser.add_argument('--webcam', action='store_true', help='Start webcam detection')
    parser.add_argument('--no-save', action='store_true', help='Disable saving detected frames')
    parser.add_argument('--show-categories', action='store_true', help='Display weapon categories in detection')

    # Configuration options
    parser.add_argument('--confidence', type=float, help='Set confidence threshold (0.0-1.0)')
    parser.add_argument('--custom-model', help='Use custom model (specify name without extension)')

    # Category management
    category_group = parser.add_argument_group('Category Management')
    category_group.add_argument('--list-categories', action='store_true', help='List available weapon categories')
    category_group.add_argument('--list-enabled', action='store_true', help='List enabled weapon categories')
    category_group.add_argument('--enable-category', help='Enable a weapon category')
    category_group.add_argument('--disable-category', help='Disable a weapon category')
    category_group.add_argument('--add-custom-category', help='Add a custom category name')
    category_group.add_argument('--custom-classes', help='Classes for the custom category (comma separated)')

    args = parser.parse_args()

    # Setup project if requested
    if args.setup:
        setup_project()

    # Handle category management
    if args.list_categories:
        list_weapon_categories()

    if args.list_enabled or args.enable_category or args.disable_category:
        manage_categories(add=args.enable_category, remove=args.disable_category, list_enabled=args.list_enabled)

    if args.add_custom_category and args.custom_classes:
        add_custom_category(args.add_custom_category, args.custom_classes)

    # Update confidence threshold if specified
    if args.confidence is not None:
        if 0.0 <= args.confidence <= 1.0:
            update_config("confidence_threshold", args.confidence)
        else:
            print("Confidence threshold must be between 0.0 and 1.0")

    # Run webcam detection if requested
    if args.webcam:
        webcam_detection(
            custom_model=args.custom_model,
            enable_saving=not args.no_save,
            display_categories=args.show_categories
        )
    elif not (args.setup or args.list_categories or args.list_enabled or
              args.enable_category or args.disable_category or args.add_custom_category):
        print("Please run with --webcam to start detection")
        print("Or run with --setup to prepare the project")
        print("Use --help for more information")
