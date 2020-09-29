package com.maliktillman.faceapp;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.hardware.Camera.Size;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.view.animation.AlphaAnimation;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;
import android.support.v4.app.ActivityCompat;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.List;

public class CameraActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {
    /**
     * Custom java camera view
     */
    private JavaCamera3View mOCvCameraView;

    /**
     * Native JNI method call to find arbitrary features in matrix frames
     *
     * @param matAddrGr Grayscale frame matrix address to be processed
     * @param matAddrRgba Address to place processed RGBA frame matrix
     */
    public native void FindFeatures(long matAddrGr, long matAddrRgba);

    /**
     * Native JNI method call to detect faces in matrix frames
     *
     * @param addrRgba RGBA matrix address
     * @param Face face cascade files
     * @param eyes eyes cascade files
     */
    public native void DetectFace(long addrRgba, String Face, String eyes);

    /**
     * Main frame matrix
     */
    private Mat mFrame;

    /**
     * Hold frame matrices
     */
    private Mat holdFrame1;
    private Mat holdFrame2;

    /**
     * Main intermediate matrix
     */
    private Mat mIntermediateMat;

    /**
     * Canny effect threshold
     */
    private TextView cannyThreshold;

    /**
     * SeekBar for canny effect
     */
    private SeekBar cannySlider;

    /**
     * Camera permission request identifier
     */
    private static final int REQUEST_CAMERA = 0;

    /**
     * Write to external storage permission request identifier
     */
    private static final int REQUEST_STORAGE_WRITE = 1;

    /**
     * For loading face cascades
     */
    private String face_file, eye_file;

    /**
     * Clear filter code
     */
    private final int FILTER_NONE = 0;

    /**
     * Black & White filter code
     */
    private final int FILTER_MONO = 1;

    /**
     * Canny filter code
     */
    private final int FILTER_CANNY = 2;

    /**
     * Detect features filter code todo: refactor to 'FILTER_DETECT'
     */
    private final int FILTER_FEATURES = 3;

    /**
     * Find faces filter code todo: refactor to 'FILTER_FACE'
     */
    private final int FILTER_DETECT = 4;

    /**
     *  Define last filter (filter with highest int value)
     */
    private final int MAX_VALUE = FILTER_DETECT;

    /**
     *  Initial filter
     */
    private int filter = FILTER_NONE;

    /**
     * Callback for loading OPENCV library
     */
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override public void onManagerConnected(int status) {
            // Load cascades and initialize OpenCV dependencies
            if (status == LoaderCallbackInterface.SUCCESS) {
                loadCascade();
                initializeOpenCVDependencies();
            }

            else
                super.onManagerConnected(status);
        }
    };

    /**
     * Load and convert raw face and eye cascade data to byte.
     */
    public void loadCascade(){
        try {
            // Load face cascade file from application raw resources
            InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt2);

            // Make cascade directory
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);

            // Create face cascade file and output stream
            File frontalFace = new File(cascadeDir, "haarcascade_frontalface_alt2.xml");
            FileOutputStream os = new FileOutputStream(frontalFace);

            // Convert face cascade raw resource file to bytes
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }

            // Get absolute path of face cascade byte file
            face_file = frontalFace.getAbsolutePath();

            // Close input and output streams
            is.close();
            os.close();

            // Load eye cascade from application raw resources
            is = getResources().openRawResource(R.raw.haarcascade_eye_tree_eyeglasses);

            // Make new byte file for eye cascade and make an output stream
            File cEyes = new File(cascadeDir, "haarcascade_eye_tree_glassess.xml");
            os = new FileOutputStream(cEyes);

            // Convert eye cascade raw resource file to bytes
            byte[] buffer2 = new byte[4096];
            int bytesRead2;
            while ((bytesRead2 = is.read(buffer2)) != -1) {
                os.write(buffer2, 0, bytesRead2);
            }

            // Get absolute path of eye cascade byte file
            eye_file = cEyes.getAbsolutePath();

            // Close input and output streams
            is.close();
            os.close();

            // Delete cascade directory
            cascadeDir.delete();
        } catch (Exception e) {
            e.printStackTrace();
            Log.e("ERROR", "Failed to load cascade. Exception thrown: " + e);
        }
    }

    /**
     * Enable camera view
     */
    private void initializeOpenCVDependencies() {
        mOCvCameraView.enableView();
    }

    /**
     * Checks if proper app permissions have been set.
     * If not, it will request the user to grant these permissions.
     */
    public void checkPermissions() {
        // Check current permission values
        int cameraPermissions = ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA);
        int writeStoragePermissions = ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE);

        // Request CAMERA permission if not yet granted
        if(cameraPermissions != PackageManager.PERMISSION_GRANTED)
            ActivityCompat.requestPermissions(CameraActivity.this, new String[]{Manifest.permission.CAMERA},
                    REQUEST_CAMERA);

        // Request EXTERNAL WRITE permission if not yet granted
        if(writeStoragePermissions != PackageManager.PERMISSION_GRANTED)
            ActivityCompat.requestPermissions(CameraActivity.this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    REQUEST_STORAGE_WRITE);
    }

    /**
     * Initialize view and all UI elements
     *
     * @param scaleX horizontal scale
     * @param scaleY vertical scale
     */
    public void initView(Float scaleX, Float scaleY){
        // Load view and make visible
        mOCvCameraView = (JavaCamera3View) findViewById(R.id.view);
        mOCvCameraView.setVisibility(SurfaceView.VISIBLE);

        // Use front camera
        mOCvCameraView.setCameraIndex(1);

        // Set output scale
        mOCvCameraView.setScaleX(scaleX);
        mOCvCameraView.setScaleY(scaleY);

        // Initialize filter selection click regions
        initRegions(0);

        // Initialize gif camera button
        initCameraButton();

        // Initialize canny effect slider
        initCannySeek();
    }

    /**
     * Initialize filter selection click regions
     *
     * @param Alpha opacity level of regions
     */
    public void initRegions(int Alpha){
        // Load left click region and bring to front
        ImageView left_region = (ImageView) findViewById(R.id.left_region);
        left_region.bringToFront();
        left_region.setImageAlpha(Alpha);

        // Setup left click listener to switch to the previous filter in order
        left_region.setClickable(true);
        left_region.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) {
                // Set last filter, if current filter is initial filter
                if(filter == FILTER_NONE) filter = MAX_VALUE;

                // Otherwise set to previous filter by value
                else filter--;

                // Show threshold slider when canny effect is active
                if(filter==FILTER_CANNY) {
                    cannySlider.setVisibility(View.VISIBLE);
                    cannyThreshold.setVisibility(View.VISIBLE);
                }

                // Hide threshold slider when canny effect is not active
                else {
                    cannySlider.setVisibility(View.INVISIBLE);
                    cannyThreshold.setVisibility(View.INVISIBLE);
                }
            }
        });

        // Load right click region and bring to front
        ImageView right_region = (ImageView) findViewById(R.id.right_region);
        right_region.bringToFront();
        right_region.setImageAlpha(Alpha);

        // Setup right click listener to switch to the next filter in order
        right_region.setClickable(true);
        right_region.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) {
                // Set initial filter, if current filter is last filter
                if(filter==MAX_VALUE) filter=FILTER_NONE;

                // Otherwise set to the next filter by value
                else filter++;

                // Show threshold slider when canny effect is active
                if(filter==FILTER_CANNY) {
                    cannySlider.setVisibility(View.VISIBLE);
                    cannyThreshold.setVisibility(View.VISIBLE);
                }

                // Hide threshold slider when canny effect is not active
                else {
                    cannySlider.setVisibility(View.INVISIBLE);
                    cannyThreshold.setVisibility(View.INVISIBLE);
                }
            }
        });
    }

    /**
     * Initialize gif camera button
     */
    public void initCameraButton(){
        // Load gif button and bring to front
        GifImageView gifBtn = (GifImageView) findViewById(R.id.camera_btn);
        gifBtn.bringToFront();

        // Take picture on click
        gifBtn.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { takePic(); }
        });
    }

    /**
     * Initialize canny effect slider/seek
     */
    public void initCannySeek(){
        // Set fade in animation
        final AlphaAnimation fade_in = new AlphaAnimation(1.0f, 0.0f);
        fade_in.setDuration(1000);

        // Load canny effect slider and text
        cannySlider = (SeekBar) findViewById(R.id.cannySlider);
        cannyThreshold = (TextView) findViewById(R.id.threshold);

        // Bring threshold and slider to front of touch context
        cannyThreshold.bringToFront();
        cannySlider.bringToFront();

        // Set text to canny effect slider value
        cannyThreshold.setText(Integer.toString(cannySlider.getProgress()));

        // Update canny threshold text on slider change
        cannySlider.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            /**
             * When canny effect slider value has changed, set canny effect threshold text to
             * new value.
             *
             * @param seekBar canny effect slider
             * @param progress new canny effect value
             * @param fromUser Whether or not change came from the user
             */
            @Override public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                cannyThreshold.setText(Integer.toString(progress));
            }

            /**
             * Shows threshold value when slider touch tracking has started
             *
             * @param seekBar Canny effect slider
             */
            @Override public void onStartTrackingTouch(SeekBar seekBar) {
                cannyThreshold.setAlpha(.5f);
            }

            /**
             * Hides threshold value when touch tracking has stopped
             *
             * @param seekBar Canny effect slider
             */
            @Override public void onStopTrackingTouch(SeekBar seekBar) {
                cannyThreshold.setAlpha(0f);
            }
        });
    }

    /**
     * Converts current camera frame into bitmap and send to system media gallery
     */
    public void takePic(){
        // Get bitmap information for current frame
        Bitmap btmap = Bitmap.createBitmap(mFrame.cols(), mFrame.rows(), Bitmap.Config.ARGB_8888);

        // Convert frame to bitmap
        Utils.matToBitmap(mFrame, btmap);

        // Resize bitmap
        Bitmap btmapResized = Bitmap.createScaledBitmap(btmap, mFrame.width(), mFrame.height(), false);

        // Release frame
        mFrame.release();

        // Send bitmap to media gallery
        MediaStore.Images.Media.insertImage(getContentResolver(), btmapResized, "Face App Image", null);

        // Notify user picture has been saved
        Toast.makeText(this, "Picture saved to Gallery", Toast.LENGTH_LONG).show();
    }

    /**
     * Set image resolution using custom camera api
     */
    public void setResolution(){
        List<Size> resolutions = mOCvCameraView.getResolutionList();
        Size resolution = resolutions.get(0);

        for(int i = 0; i<resolutions.size()-1;i++){
            resolution = resolutions.get(i);
        }

        Toast.makeText(this, "Resolution: " + resolution, Toast.LENGTH_LONG).show();

        mOCvCameraView.setResolution(resolution);
    }

    /**
     * Check application permission, initialize views/UI, and load native C++ library.
     *
     * @param savedInstanceState saved state of app
     */
    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        // Check necessary permissions are granted todo:move to splash
        checkPermissions();

        // Set flag to keep screen on
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        // Lock screen orientation to portrait
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);

        // Initialize View and UI Elements
        initView(1.6f, 2.8f);

        // Class is current listener
        mOCvCameraView.setCvCameraViewListener(this);

        // Load native JNI library
        System.loadLibrary("native-lib");
    }

    /**
     * Makes sure OpenCV is loaded on device
     */
    @Override public void onResume() {
        super.onResume();

        // Uses OpenCV manager to initialize OpenCV on device if library is not found
        if (!OpenCVLoader.initDebug())
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback);

        // OpenCV library found on device
        else
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
    }

    /**
     * Initialize frames on view start
     *
     * @param width the width of the frames that will be delivered
     * @param height the height of the frames that will be delivered
     */
    public void onCameraViewStarted(int width, int height) {
        mFrame = new Mat(height, width, CvType.CV_8UC4);
        holdFrame1 = new Mat(height, width, CvType.CV_8UC4);
        holdFrame2 = new Mat(height, width, CvType.CV_8UC4);
        mIntermediateMat = new Mat(height, width, CvType.CV_8UC4);
    }

    /**
     * On each camera frame we apply the filter and rotate the frame 90 degrees
     *
     * @param inputFrame Camera frame to be processed
     * @return processed frame
     */
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        switch (filter){
            case FILTER_NONE:
                mFrame = inputFrame.rgba();
                break;
            case FILTER_MONO:
                Imgproc.cvtColor(inputFrame.gray(), mFrame, Imgproc.COLOR_GRAY2RGBA, 4);
                break;
            case FILTER_CANNY:
                mFrame = inputFrame.rgba();
                Imgproc.Canny(inputFrame.gray(), mFrame,
                        cannySlider.getProgress(), cannySlider.getProgress()*2,
                        3, false);
                break;
            case FILTER_FEATURES:
                mFrame = inputFrame.rgba();
                mIntermediateMat = inputFrame.gray();
                FindFeatures(mIntermediateMat.getNativeObjAddr(), mFrame.getNativeObjAddr());
                break;
            case FILTER_DETECT:
                mFrame = inputFrame.rgba();
                DetectFace(mFrame.getNativeObjAddr(), face_file, eye_file);
        }

        // Rotate mRgba 90 degrees
        Core.transpose(mFrame, holdFrame2);
        Imgproc.resize(holdFrame2, holdFrame1, holdFrame1.size(), 0,0, 0);
        Core.flip(holdFrame1, mFrame, -1 );

        // Return processed frame to be output
        return mFrame;
    }

    /**
     * Clear frame when camera has been closed
     */
    public void onCameraViewStopped() {
        mFrame.release();
    }

    /**
     * Disable camera view when not in focus
     */
    @Override public void onPause() {
        super.onPause();

        if (mOCvCameraView != null) mOCvCameraView.disableView();
    }

    /**
     * Disable camera view when app is closed
     */
    @Override public void onDestroy() {
        super.onDestroy();

        if (mOCvCameraView != null) mOCvCameraView.disableView();
    }

    /**
     * Close app if camera request permissions are not granted
     *
     * @param requestCode -
     * @param permissions -
     * @param grantResults-
     */
    @Override public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        // Check if camera permission request was denied
        if(requestCode == REQUEST_CAMERA && grantResults[0] == PackageManager.PERMISSION_DENIED) {
            // Alert user
            Toast.makeText(CameraActivity.this, "Sorry, this app requires camera permissions to function",
                    Toast.LENGTH_LONG).show();

            // Close app
            finish();
        }
    }
}
