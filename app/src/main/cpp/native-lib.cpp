#include <jni.h>
#include <vector>
#include <cstdio>
#include <android/log.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace std;
using namespace cv;

extern "C" {
    /**
     * Tracks small features in frames; outputs circles indicating the tracking.
     *
     * @param addrGray - Address to greyscale input frame processed to find the small features.
     * @param addrRgba - Address to RGBA frame where processed information will be placed (i.e destination frame)
     */
    JNIEXPORT void JNICALL Java_com_maliktillman_faceapp_CameraActivity_FindFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba) {
        // Grayscale input frame
        Mat& mGr  = *(Mat*)addrGray;

        // RGB destination frame
        Mat& mRgb = *(Mat*)addrRgba;

        // Detect small details
        Ptr<FeatureDetector> detector = FastFeatureDetector::create(30);
        vector<KeyPoint> v;
        detector->detect(mGr, v);

        // Draw circles around found details
        for(const auto & kp : v)
            circle(mRgb, Point(kp.pt.x, kp.pt.y), 6, Scalar(255,0,0,255));
    }

    /**
     * Detects faces and eyes using loaded HarrCascades.
     *
     * @param env
     * @param frameAddr - Address to current frame
     * @param face_file - String file location for face cascade
     * @param eyes_file - String file location for eye cascade
     */
    JNIEXPORT void JNICALL Java_com_maliktillman_faceapp_CameraActivity_DetectFace (JNIEnv *env, jobject, jlong frameAddr, jstring face_file, jstring eyes_file) {
        // Declare face and eye cascade classifier
        CascadeClassifier face_cascade, eyes_cascade;

        // Reference current matrix frame address
        Mat& frame = *(Mat*)frameAddr;

        // Get files for face and eye cascades
        const char* mFace = env -> GetStringUTFChars( face_file, NULL );
        const char* mEyes = env -> GetStringUTFChars( eyes_file, NULL );
        const cv::String face_cascade_file = ( mFace );
        const cv::String eyes_cascade_file =  ( mEyes );

        // Load face cascade; Checks if it fails
        if (!face_cascade.load(face_cascade_file)) {
            printf("Error - Face cascade failed to load\n");
            return;
        }

        // Load eye cascades; Check if it fails
        if (!eyes_cascade.load(eyes_cascade_file)) {
            printf("Error - Eye cascades failed to load\n");
            return;
        }

        // Log cascade load success
        printf("Success - Cascades loaded\n");

        // Convert frame to gray scale
        Mat gray, smallImg;
        cvtColor( frame, gray, COLOR_BGR2GRAY );

        // Resize the Grayscale Image
        resize( gray, smallImg, Size(), 1, 1, INTER_LINEAR );

        // Equalize Histogram
        equalizeHist( smallImg, smallImg );

        // Detect faces of different sizes using cascade classifier
        vector<Rect> faces;
        face_cascade.detectMultiScale( smallImg, faces, 1.1,
                                       3, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

        // Draw circles around the faces
        for (const auto& face : faces)
        {
            // Circle color
            Scalar color = Scalar(255, 0, 0);

            // Find aspect ratio of face
            double aspect_ratio = (double) face.width / face.height;

            // Draw circle around faces between .75 and 1.3 aspect ratio
            if( 0.75 < aspect_ratio && aspect_ratio < 1.3 ) {
                // Defining center of classified face
                Point center;
                center.x = cvRound(( face.x + face.width * 0.5 ));
                center.y = cvRound(( face.y + face.height * 0.5 ));

                // Radius of circle around classified face
                int radius = cvRound(( face.width + face.height ) * 0.25 * 2);

                // Draw circle
                circle( frame, center, radius, color, 3, 8, 0 );
            }

            else // Draw rectangle around classified face
                rectangle(frame, cvPoint(cvRound( face.x ), cvRound( face.y )),
                          cvPoint(cvRound(( face.x + face.width - 1 )),
                                   cvRound(( face.y + face.height - 1 ))), color, 3, 8, 0);

            // If eyes are classified, draw circles around them
            if( !eyes_cascade.empty() ) {
                // Define algorithms range of interest
                Mat smallImgROI = smallImg( face );

                // Detection of eyes int the input image
                vector<Rect> nestedObjects;
                eyes_cascade.detectMultiScale( smallImgROI, nestedObjects, 1.1, 2,
                                               0u|CASCADE_SCALE_IMAGE, Size(30, 30) );

                // Draw circles around eyes
                for (const auto& nr : nestedObjects)
                {
                    // Defining center of classified eye
                    Point center;
                    center.x = cvRound((face.x + nr.x + nr.width * 0.5));
                    center.y = cvRound((face.y + nr.y + nr.height * 0.5));

                    // Radius of circle around classified eye
                    int radius = cvRound((nr.width + nr.height)*0.25);

                    // Draw circle
                    circle( frame, center, radius, color, 3, 8, 0 );
                }
            }
        }
    }
}