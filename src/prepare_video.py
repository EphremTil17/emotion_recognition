import cv2
import argparse
from pathlib import Path

def fix_video_orientation(input_path, output_path):
    """
    Fix video orientation to landscape mode.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    # Read first frame to get dimensions
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read video frame")

    height, width = frame.shape[:2]
    is_portrait = height > width

    if not is_portrait:
        print("Video is already in landscape orientation. No rotation needed.")
        return False

    # Calculate new dimensions
    new_width = height
    new_height = width

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        cap.get(cv2.CAP_PROP_FPS),
        (new_width, new_height)
    )

    # Reset video capture
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Process all frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate frame to landscape
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        out.write(frame)

        frame_count += 1
        if frame_count % 30 == 0:  # Update progress every 30 frames
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}%", end='\r')

    cap.release()
    out.release()
    print("\nVideo processing completed!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Fix video orientation for emotion recognition')
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('--output', help='Path to output video file (optional)')
    
    args = parser.parse_args()
    input_path = args.input_video
    
    if args.output:
        output_path = args.output
    else:
        # Create output path with '_landscape' suffix
        input_path_obj = Path(input_path)
        output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_landscape{input_path_obj.suffix}")
    
    try:
        print("Checking video orientation...")
        if fix_video_orientation(input_path, output_path):
            print(f"Video has been rotated and saved to: {output_path}")
        else:
            print("No orientation fix needed.")
    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()