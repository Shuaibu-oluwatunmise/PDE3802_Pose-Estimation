import cv2
import os
from datetime import datetime
import time


def format_class_name(class_input: str) -> str:
    """
    Convert user input into a clean folder name.
    Example:
        "headset" → "Headset"
        "cards box" → "CardsBox"
    """
    parts = class_input.strip().split()
    return "".join(p.capitalize() for p in parts)


def create_directory(class_name):
    """Create directory for saving videos under data/raw_videos/<ClassName>"""
    dir_path = os.path.join("data", "raw_videos", class_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def get_video_filename(dir_path):
    """Generate unique filename with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"video_{timestamp}.avi"
    return os.path.join(dir_path, filename)


def record_video(class_name, duration=20):
    """Record a video clip for a given class name."""
    dir_path = create_directory(class_name)
    video_path = get_video_filename(dir_path)

    # Initialise camera
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False

    # Camera props
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # default FPS

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

    print(f"\nRecording for {duration} seconds…")
    print("Press 'q' to stop recording early.")

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        out.write(frame)

        elapsed = time.time() - start_time
        remaining = duration - elapsed

        # Display preview
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Recording: {remaining:.1f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Class: {class_name}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Recording", display_frame)

        # Stop if done or user pressed 'q'
        if elapsed >= duration or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video saved to: {video_path}")
    return True


def main():
    print("=" * 50)
    print("Generalised Video Data Collection Tool")
    print("=" * 50)
    print("\nEnter ANY object name to start recording.")
    print("Examples: headset, wallet, notebook, etc.")
    print("Type 'exit' to quit.")
    print("=" * 50)

    while True:
        class_input = input("\nEnter class name: ").strip()

        if class_input.lower() in ["exit", "quit", "q"]:
            print("Exiting data collection tool.")
            break

        if not class_input:
            print("Class name cannot be empty. Try again.")
            continue

        # Convert to clean folder name: "cards box" → "CardsBox"
        class_name = format_class_name(class_input)

        success = record_video(class_name)

        if success:
            print(f"✓ Successfully recorded video for class: {class_name}")
        else:
            print("Recording failed. Try again.")


if __name__ == "__main__":
    main()
