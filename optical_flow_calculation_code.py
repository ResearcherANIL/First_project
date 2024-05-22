import cv2
import numpy as np
import math

def determine_direction(flow):
    """
    Determine direction and turning angle based on motion vectors.
    """
    mean_flow = np.mean(flow, axis=(0, 1))
    dx, dy = mean_flow

    threshold = 1
    angle = math.degrees(math.atan2(dy, dx))

    if abs(dx) > abs(dy):
        if dx > threshold:
            direction = "Left"
            if -90 < angle < 0:
                angle = 90 + angle
            elif 0 < angle < 90:
                angle = 180 - (90 - angle)
            
            if 0 < angle < 30:
                direction = 'small left'
            elif 30 < angle < 60:
                direction = 'medium left'
            elif 60 < angle:
                direction = 'large left'
        elif dx < -threshold:
            direction = "Right"
            if angle < -90:
                angle = 90 - (180 + angle)
            elif angle > 90:
                angle = 90 + (180 - angle)
            
            if 0 < angle < 30:
                direction = 'small right'
            elif 30 < angle < 60:
                direction = 'medium right'
            elif 60 < angle:
                direction = 'large right'
        else:
            direction = "Straight"
            angle = 0.0
    else:
        direction = "Straight"
        angle = 0.0

    return direction, angle

def process_video(input_path, output_path):
    """
    Process the input video to determine direction and angle, and save the result.
    """
    input_video = cv2.VideoCapture(input_path)
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    params = dict(pyr_scale=0.5, levels=5, winsize=20, iterations=10, poly_n=5, poly_sigma=1.5, flags=0)

    ret, prev_frame = input_video.read()
    if not ret:
        print("Error: Unable to read the video file.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = input_video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **params)

        direction, angle = determine_direction(flow)
        text = f"{direction}, Angle: {angle:.2f}"
        cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        output_video.write(frame)
        prev_gray = gray.copy()

    input_video.release()
    output_video.release()
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    input_path = 'Input_videos/file_01.mp4'
    output_path = 'Output_videos/file_01.mp4'
    process_video(input_path, output_path)
