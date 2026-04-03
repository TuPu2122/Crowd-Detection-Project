from ultralytics import YOLO
import cv2
import os
import pandas as pd

# -------------------------------
# LOAD YOLO MODEL
# -------------------------------
model = YOLO("yolov8s.pt")   # if system is slow, use yolov8n.pt

# -------------------------------
# BASE FOLDER
# -------------------------------
base_folder = "training_images"

# -------------------------------
# ENCODING MAPS
# -------------------------------
day_type_map = {
    "normal_days": 0,
    "exam_special_days": 1,
    "event_days": 2
}

location_map = {
    "Gate1": 1,
    "Gate2": 2,
    "Gate3": 3,
    "Canteen": 4,
    "Open_Auditorium": 5,
    "Ground": 6
}

time_slot_map = {
    "Morning": 1,
    "Afternoon": 2,
    "Evening": 3
}

# -------------------------------
# DATA STORAGE
# -------------------------------
dataset = []

print("\nGenerating crowd dataset from folder structure...\n")

# -------------------------------
# WALK THROUGH FOLDER STRUCTURE
# -------------------------------
for day_folder in os.listdir(base_folder):
    day_folder_path = os.path.join(base_folder, day_folder)

    if not os.path.isdir(day_folder_path):
        continue

    if day_folder not in day_type_map:
        print(f"Skipping unknown day folder: {day_folder}")
        continue

    day_type = day_type_map[day_folder]

    # Location folders
    for location_folder in os.listdir(day_folder_path):
        location_path = os.path.join(day_folder_path, location_folder)

        if not os.path.isdir(location_path):
            continue

        if location_folder not in location_map:
            print(f"Skipping unknown location folder: {location_folder}")
            continue

        location_code = location_map[location_folder]

        # Time folders
        for time_folder in os.listdir(location_path):
            time_path = os.path.join(location_path, time_folder)

            if not os.path.isdir(time_path):
                continue

            if time_folder not in time_slot_map:
                print(f"Skipping unknown time folder: {time_folder}")
                continue

            time_slot = time_slot_map[time_folder]

            # Images
            for img_name in os.listdir(time_path):
                img_path = os.path.join(time_path, img_name)

                image = cv2.imread(img_path)
                if image is None:
                    print(f"Error loading image: {img_name}")
                    continue

                # -------------------------------
                # YOLO DETECTION
                # -------------------------------
                results = model(image, conf=0.25, imgsz=640)

                count = 0

                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])

                        if cls == 0:  # person class
                            count += 1

                # Save row
                dataset.append([
                    day_type,
                    location_code,
                    time_slot,
                    count,
                    img_name
                ])

                print(f"{img_name} | DayType: {day_folder} | Location: {location_folder} | Time: {time_folder} | Count: {count}")

                # OPTIONAL: Display image with count
                cv2.putText(image, f"Count: {count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow("YOLO Crowd Detection", image)
                cv2.waitKey(500)

cv2.destroyAllWindows()

# -------------------------------
# SAVE FINAL DATASET
# -------------------------------
df = pd.DataFrame(dataset, columns=[
    "day_type",
    "location",
    "time_slot",
    "count",
    "image_name"
])

df.to_csv("crowd_dataset.csv", index=False)

print("\nFinal dataset created successfully: crowd_dataset.csv")
print(df.head())