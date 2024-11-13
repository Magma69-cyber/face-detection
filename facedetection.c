#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>

using namespace cv;

// Function to detect faces in an image
void detectAndDisplay(Mat frame, CascadeClassifier face_cascade) {
    Mat frame_gray;
    // Convert the image to grayscale
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    // Equalize the histogram
    equalizeHist(frame_gray, frame_gray);

    // Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    // Draw rectangles around the faces
    for (size_t i = 0; i < faces.size(); i++) {
        Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
        ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 2, 8, 0);
    }

    // Show the result
    imshow("Face Detection", frame);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        return -1;
    }

    // Load the Haar Cascade file for face detection
    const char* face_cascade_name = "haarcascade_frontalface_default.xml";
    CascadeClassifier face_cascade;
    if (!face_cascade.load(face_cascade_name)) {
        printf("--(!)Error loading face cascade\n");
        return -1;
    }

    // Load the image
    Mat image = imread(argv[1], IMREAD_COLOR);
    if (image.empty()) {
        printf("Could not open or find the image\n");
        return -1;
    }

    // Perform face detection and display the result
    detectAndDisplay(image, face_cascade);
    waitKey(0);
    return 0;
}
//Compiling and Running:
To compile the program, use the following command (assuming g++ is installed):
g++ face_detection.cpp -o face_detection `pkg-config --cflags --libs opencv4`
//To run the program, provide the path to an image file as an argument:
./face_detection path/to/your/image.jpg
