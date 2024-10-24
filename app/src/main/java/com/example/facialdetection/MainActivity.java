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

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    private ImageView imageView;
    private Uri imageUri;
    private CascadeClassifier faceDetector;

    static {
        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Initialization Failed!");
        } else {
            Log.d("OpenCV", "Initialization Successful!");
        }
    }

    private final ActivityResultLauncher<Intent> galleryActivityResultLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            new ActivityResultCallback<ActivityResult>() {
                @Override
                public void onActivityResult(ActivityResult result) {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                        Uri imageUri = result.getData().getData();
                        if (imageUri != null) {
                            try {
                                Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
                                Bitmap processedBitmap = detectFaces(bitmap);
                                imageView.setImageBitmap(processedBitmap);
                            } catch (IOException e) {
                                e.printStackTrace();
                            }
                        }
                    }
                }
            }
    );

    private final ActivityResultLauncher<Intent> cameraActivityResultLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            new ActivityResultCallback<ActivityResult>() {
                @Override
                public void onActivityResult(ActivityResult result) {
                    if (result.getResultCode() == RESULT_OK) {
                        try {
                            Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
                            Bitmap processedBitmap = detectFaces(bitmap);
                            imageView.setImageBitmap(processedBitmap);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                }
            }
    );

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        imageView = findViewById(R.id.imageView);
        Button galleryBtn = findViewById(R.id.button1);
        Button cameraBtn = findViewById(R.id.button2);

        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED ||
                checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_DENIED) {
            String[] permissions = {Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE};
            requestPermissions(permissions, 122);
        }

        // Khởi tạo CascadeClassifier
        initializeOpenCVDependencies();

        galleryBtn.setOnClickListener(view -> {
            Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            galleryActivityResultLauncher.launch(galleryIntent);
        });

        cameraBtn.setOnClickListener(view -> {
                openCamera();
        });
    }

    private void initializeOpenCVDependencies() {
        try {
            // Mở tập tin cascade từ assets
            InputStream is = getResources().getAssets().open("haarcascade_frontalface_default.xml");
            File cascadeDir = getDir("cascade", MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_default.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
            // Tạo CascadeClassifier
            faceDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            if (faceDetector.empty()) {
                Log.e("OpenCV", "Failed to load cascade classifier");
                faceDetector = null;
            } else {
                Log.i("OpenCV", "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());
            }
            // Xóa thư mục cascade
            cascadeDir.delete();
        } catch (IOException e) {
            e.printStackTrace();
            Log.e("OpenCV", "Error loading cascade", e);
        }
    }
    private void openCamera() {
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.TITLE, "New Picture");
        values.put(MediaStore.Images.Media.DESCRIPTION, "From the Camera");
        imageUri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);

        if (imageUri != null) {
            Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
            cameraActivityResultLauncher.launch(cameraIntent);
        }
    }
    private Bitmap detectFaces(Bitmap original) {
        if (faceDetector == null) {
            Log.e("OpenCV", "CascadeClassifier not initialized");
            return original;
        }

        Bitmap mutableBitmap = original.copy(Bitmap.Config.ARGB_8888, true);
        Mat mat = new Mat();
        Utils.bitmapToMat(mutableBitmap, mat);

        // Chuyển đổi sang ảnh xám
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist(mat, mat);

        // Phát hiện khuôn mặt
        MatOfRect faces = new MatOfRect();
        faceDetector.detectMultiScale(mat, faces, 1.1, 2, 2,
                new org.opencv.core.Size(100, 100), new org.opencv.core.Size());

        Rect[] facesArray = faces.toArray();
        Log.d("OpenCV", "Number of faces detected: " + facesArray.length);

        // Vẽ hình chữ nhật xung quanh khuôn mặt
        Canvas canvas = new Canvas(mutableBitmap);
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(5);

        for (Rect face : facesArray) {
            canvas.drawRect(face.x, face.y, face.x + face.width, face.y + face.height, paint);
        }
        return mutableBitmap;
    }
}
