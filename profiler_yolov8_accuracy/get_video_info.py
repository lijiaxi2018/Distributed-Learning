import cv2
import sys

def get_video_details(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video")

    fps = video.get(cv2.CAP_PROP_FPS)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = (width, height)

    video.release()

    print("FPS:", str(fps), "Resolution:", str(resolution))

if __name__ == "__main__":
     get_video_details(sys.argv[1])