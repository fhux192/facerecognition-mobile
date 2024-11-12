package com.example.facialdetection;

import android.Manifest;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.Mat;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collections;
import java.util.List;

public class MainActivity extends CameraActivity {
    private static final String TAG = "YourActivity";
    private CameraBridgeViewBase mOpenCvCameraView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        // Initialize your camera view
        mOpenCvCameraView = findViewById(R.id.opencv_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(cvCameraViewListener2);

        // Initialize OpenCV
        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "Failed to initialize OpenCV");
            // Handle initialization error
        } else {
            Log.d(TAG, "OpenCV initialized successfully");
            mOpenCvCameraView.enableView();
        }
    }
    @Override
    protected List<?extends CameraBridgeViewBase>getCameraViewList(){
        return Collections.singletonList(mOpenCvCameraView);
    }
    private CameraBridgeViewBase.CvCameraViewListener2 cvCameraViewListener2= new CameraBridgeViewBase.CvCameraViewListener2(){
        @Override
        public void onCameraViewStarted(int width, int height) {

        }

        @Override
        public void onCameraViewStopped() {

        }

        @Override
        public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
            Mat input_rgba = inputFrame.rgba();
            Mat input_gray = inputFrame.gray();

            MatOfPoint corners = new MatOfPoint();
            Imgproc.goodFeaturesToTrack(input_gray, corners, 20, 0.01, 10, new Mat(), 3, false);
            Point[] cornersArr = corners.toArray();

            for (int i = 0; i < cornersArr.length; i++) {
                Imgproc.circle(input_rgba, cornersArr[i], 10, new Scalar(0, 255, 0), 2);
            }

            return input_rgba;
        }



    };

    @Override
    protected void onResume() {
        super.onResume();
        // Re-initialize OpenCV if needed
        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "Failed to re-initialize OpenCV");
        } else {
            Log.d(TAG, "OpenCV re-initialized successfully");
            if (mOpenCvCameraView != null)
                mOpenCvCameraView.enableView();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }
}