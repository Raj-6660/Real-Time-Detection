import os
import torch
import cv2
import io
import uuid
import numpy as np
from PIL import Image
from ultralytics import YOLO
from flask import (
    Flask,
    url_for,
    request,
    redirect,
    send_file,
    render_template,
    send_from_directory,
)

# Init flask app
app = Flask(__name__)

# Define constants
FONT_SCALE = 2
TEXT_THICKNESS = 1
MIN_CONFIDENCE = 0.5
BOUNDING_BOX_THICKNESS = 2
FONT_NAME = cv2.FONT_HERSHEY_PLAIN
os.environ["YOLO_VERBOSE"] = "False"
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


class Detection:
    def __init__(self, model_name: str, confidence: float = MIN_CONFIDENCE):
        self.model = YOLO(model_name)
        self.conf = confidence

    def predict(self, img):
        with torch.no_grad():
            results = self.model(img, conf=self.conf,
                                 stream=True, verbose=False)
        return results

    def predict_and_detect(self, img):
        results = self.predict(img)
        for result in results:
            for box in result.boxes:
                x, y = (int(box.xyxy[0][0]), int(box.xyxy[0][1]))
                prediction_text = f"{result.names[int(box.cls[0])]}"
                # Bounding box for detection
                cv2.rectangle(
                    img,
                    (x, y),
                    (int(box.xyxy[0][2]), int(box.xyxy[0][3])),
                    (255, 0, 0),
                    BOUNDING_BOX_THICKNESS,
                )
                # Get text size
                (text_w, text_h), _ = cv2.getTextSize(
                    prediction_text,
                    FONT_NAME,
                    FONT_SCALE,
                    TEXT_THICKNESS,
                )
                # Background box for prediction text
                cv2.rectangle(
                    img, (x, y), (x + text_w, y + text_h), (255, 255, 255), -1
                )
                # Prediction text
                cv2.putText(
                    img,
                    prediction_text,
                    (x, y + text_h + FONT_SCALE - 1),
                    FONT_NAME,
                    FONT_SCALE,
                    (255, 0, 0),
                    TEXT_THICKNESS,
                )
        return img, results

    def only_detect(self, image):
        result_img, _ = self.predict_and_detect(image)
        return result_img

# <Insert Detection class definition here>
DETECTION_MODELS = {
    "yolov10n": Detection("jameslahm/yolov10n"),
    "yolov10s": Detection("jameslahm/yolov10s"),
    "yolov10m": Detection("jameslahm/yolov10m"),
    "yolov10b": Detection("jameslahm/yolov10b"),
    "yolov10l": Detection("jameslahm/yolov10l"),
    "yolov10x": Detection("jameslahm/yolov10x"),
    "custom_n_30epoch": Detection("./weights/yolov10n_30epoch_best.pt"),
    "custom_b_15epoch": Detection("./weights/yolov10b_15epoch_best.pt"),
    "custom_b_20epoch": Detection("./weights/yolov10b_20epoch_best.pt"),
}


@app.route("/")
def index():
    """Serve the main page of the application"""
    filename = request.args.get("filename", False)
    if filename:
        return render_template(
            "index.html", filename=filename, models=DETECTION_MODELS.keys()
        )
    return render_template("index.html", models=DETECTION_MODELS.keys())


@app.route("/processed/<filename>")
def processed_video(filename):
    """
    Serve a processed video file from the upload folder.

    Args:
        filename (str): The name of the video file to be served.

    Returns:
        Response: The requested MP4 video file.
    """
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename,
        as_attachment=True,
        conditional=False,
        max_age=600,
    )


@app.route("/object-detection/", methods=["POST"])
def apply_detection():
    """
    Handle object detection on an uploaded image or video file.

    Returns:
        - Uploaded image: Processed image as a PNG file.
        - Uploaded video: Redirected to the main page with video as query parameter.
        - Uploaded unsupported type: A placeholder PNG file.
        - Invalid request: Error message.
    """
    if (
        "identify_media" in request.files
        and request.files["identify_media"].filename != ""
    ):
        max_age = int(float(request.args.get("max_age", 1)) // 1 + 1)
        file = request.files["identify_media"]

        file_type = detect_mimetype(file)

        if file_type == "image":
            buf = predict_image(file, DETECTION_MODELS)
            return send_file(buf, mimetype="image/png", max_age=max_age)

        elif file_type == "video":
            temp_video_name = predict_video(file, DETECTION_MODELS)
            return redirect(url_for("index", filename=f"post_{temp_video_name}.mp4"))

        else:
            buf = generate_invalid_file_image()
            return send_file(buf, mimetype="image/png", max_age=max_age)

    else:
        return "Invalid request"


def detect_mimetype(file):
    """
    Determines the type of a file based on its MIME type.

    Args:
        file: The uploaded file.

    Returns:
        str: A string representing the file type. It can be:
            - "image" if the file is an image,
            - "video" if the file is a video,
            - "unknown" in other cases
    """
    mime_type = file.mimetype
    if mime_type.startswith("image/"):
        file_type = "image"
    elif mime_type.startswith("video/"):
        file_type = "video"
    else:
        file_type = "unknown"
    return file_type


def predict_video(file, models):
    """
    Processes an uploaded video file by applying object detection and saves the processed video.

    Args:
        file: The uploaded video file.

    Returns:
        str: The unique name (UUID) of the processed video file, excluding the extension.
    """
    video_ext = file.filename.split(".")[-1]
    temp_video_name = f"{uuid.uuid4().hex}"
    vid_opts = {
        "path": {
            "pre": os.path.join(
                app.config["UPLOAD_FOLDER"], f"{temp_video_name}.{video_ext}"
            ),
            "post": os.path.join(
                app.config["UPLOAD_FOLDER"], f"post_{temp_video_name}.mp4"
            ),
        },
        "codec": {"mp4": "avc1"},
    }
    file.save(vid_opts["path"]["pre"])
    raw_video = cv2.VideoCapture(vid_opts["path"]["pre"])
    [video_fps, video_dim] = [
        raw_video.get(cv2.CAP_PROP_FPS),
        (
            int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    ]
    processed_vid_writer = cv2.VideoWriter(
        vid_opts["path"]["post"],
        cv2.VideoWriter_fourcc(*vid_opts["codec"]["mp4"]),
        video_fps,
        video_dim,
    )
    while True:
        ret, frame = raw_video.read()
        if not ret:
            break
        processed_frame = models.get(
            request.args.get(
                "model", "custom_n_30epoch"), "custom_n_30epoch"
        ).only_detect(frame)
        processed_vid_writer.write(processed_frame)
    raw_video.release()
    processed_vid_writer.release()
    os.remove(vid_opts["path"]["pre"])
    return temp_video_name


def predict_image(file, models):
    """
    Processes an uploaded image by applying object detection and returns the processed image in memory.

    Args:
        file: The uploaded image file.

    Returns:
        io.BytesIO: A bytes buffer containing the processed image in PNG format.
    """
    img = Image.open(io.BytesIO(file.read()))
    img = np.array(img)
    # img = cv2.resize(img, (512, 512))
    img = models.get(
        request.args.get(
            "model", "custom_n_30epoch"), "custom_n_30epoch"
    ).only_detect(img)
    output = Image.fromarray(img)
    buf = io.BytesIO()
    output.save(buf, format="PNG")
    buf.seek(0)
    return buf


def generate_invalid_file_image():
    """
    Generates a placeholder image with an "INVALID FILE" message.

    Returns:
        io.BytesIO: A bytes buffer containing the generated placeholder image in PNG format.
    """
    rgb_image = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            rgb_image[i, j] = np.random.randint(0, 256, 3)
            # rgb_image[i, j] = [i, j, (i + j) // 2]
    (text_w, text_h), _ = cv2.getTextSize(
        "INVALID FILE",
        FONT_NAME,
        FONT_SCALE,
        TEXT_THICKNESS,
    )
    (x, y) = (int((256 - text_w) / 2), int((256 - text_h) / 2))
    cv2.putText(
        rgb_image,
        "INVALID FILE",
        (x, y + text_h + FONT_SCALE - 1),
        FONT_NAME,
        FONT_SCALE,
        (0, 0, 0),
        2,  # TEXT_THICKNESS,
    )
    rgb_image = Image.fromarray(rgb_image)
    buf = io.BytesIO()
    rgb_image.save(buf, format="PNG")
    buf.seek(0)
    return buf


if __name__ == "__main__":
    with Image.open("./test_cat.jpg") as sample_cat:
        for name, model in DETECTION_MODELS.items():
            pred = model.predict(sample_cat)
            print(
                f"{name}: {[result.names[int(box.cls[0])] for result in pred for box in result.boxes]}"
            )
    app.run(host="0.0.0.0", port=8000)
