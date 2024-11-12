package com.example.appcpp;

import android.Manifest;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.location.Location;
import android.net.Uri;
import android.os.Bundle;
import android.os.Looper;
import android.provider.MediaStore;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.Toast;
import android.app.AlertDialog;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.app.ActivityCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import com.firebase.ui.database.FirebaseRecyclerOptions;
import com.google.firebase.database.DatabaseException;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.ValueEventListener;
import com.google.android.gms.location.*;
import com.google.android.gms.tasks.OnSuccessListener;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MainActivityCPP extends CameraActivity {

    RecyclerView recyclerView;
    MainAdapter mainAdapter;

    private static final String TAG = "MainActivityCPP";
    private CameraBridgeViewBase mOpenCvCameraView;
    private static final int GALLERY_REQUEST_CODE = 1001;
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 2001;
    private static final int LOCATION_PERMISSION_REQUEST_CODE = 2003;

    Button galleryBtn;
    Button showFacesBtn;
    ImageView imageView;

    private List<FaceData> faceDataList = new ArrayList<>();
    private DatabaseReference faceDataRef;

    // Location variables
    private FusedLocationProviderClient fusedLocationClient;
    private volatile Location currentLocation;
    private LocationCallback locationCallback;

    static {
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "Cannot connect to OpenCV Manager");
        } else {
            System.loadLibrary("appcpp");
        }
    }

    public native void InitFaceDetector(String modelPath);
    public native int DetectFaces(long matAddrGray, long matAddrRgba, float[] largestFaceRect);
    public native void InitFaceRecognition(String modelPath);
    public native float[] ExtractFaceEmbedding(long matAddr);
    public native float CalculateSimilarity(float[] emb1, float[] emb2);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main_cpp);

        mOpenCvCameraView = findViewById(R.id.opencv_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(cvCameraViewListener2);

        imageView = findViewById(R.id.imageView);

        galleryBtn = findViewById(R.id.galleryBtn);
        galleryBtn.setOnClickListener(view -> openGallery());

        showFacesBtn = findViewById(R.id.showFacesBtn);
        showFacesBtn.setOnClickListener(view -> displayRegisteredFaces());

        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this);

        locationCallback = new LocationCallback() {
            @Override
            public void onLocationResult(LocationResult locationResult) {
                if (locationResult == null) {
                    return;
                }
                for (Location location : locationResult.getLocations()) {
                    currentLocation = location;
                }
            }
        };

        requestPermissions();

        // Firebase Database reference
        faceDataRef = FirebaseDatabase.getInstance().getReference("faceDataList");

        // Load face data from Firebase
        loadFaceDataList();

        recyclerView = findViewById(R.id.rv);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));

        FirebaseRecyclerOptions<MainModel> options = new FirebaseRecyclerOptions.Builder<MainModel>()
                .setQuery(faceDataRef, MainModel.class)
                .build();

        mainAdapter = new MainAdapter(options);
        recyclerView.setAdapter(mainAdapter);
    }

    private void requestPermissions() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            requestCameraPermission();
        } else {
            initFaceDetectionAndRecognition();
            mOpenCvCameraView.enableView();
        }

        // Request location permission
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
                != PackageManager.PERMISSION_GRANTED) {
            requestLocationPermission();
        } else {
            startLocationUpdates();
        }
    }

    private void requestCameraPermission() {
        ActivityCompat.requestPermissions(this,
                new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
    }

    private void requestLocationPermission() {
        ActivityCompat.requestPermissions(this,
                new String[]{Manifest.permission.ACCESS_FINE_LOCATION}, LOCATION_PERMISSION_REQUEST_CODE);
    }

    private void initFaceDetectionAndRecognition() {
        // Face Detection
        try {
            InputStream inputStream = getAssets().open("face_detection_yunet_2023mar.onnx");
            FileUtil fileUtil = new FileUtil();
            java.io.File detectionModelFile = fileUtil.createTempFile(this, inputStream, "face_detection_yunet_2023mar.onnx");
            InitFaceDetector(detectionModelFile.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
        }


        // Face Recognition
        try {
            InputStream inputStream = getAssets().open("face_recognition_sface_2021dec.onnx");
            FileUtil fileUtil = new FileUtil();
            java.io.File recognitionModelFile = fileUtil.createTempFile(this, inputStream, "face_recognition_sface_2021dec.onnx");
            InitFaceRecognition(recognitionModelFile.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void openGallery() {
        // Use Storage Access Framework to pick an image without needing storage permissions
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        startActivityForResult(intent, GALLERY_REQUEST_CODE);
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    private CameraBridgeViewBase.CvCameraViewListener2 cvCameraViewListener2 =
            new CameraBridgeViewBase.CvCameraViewListener2() {

                @Override
                public void onCameraViewStarted(int width, int height) {
                }

                @Override
                public void onCameraViewStopped() {
                }

                @Override
                public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
                    long startTime = System.nanoTime();

                    Mat inputRgba = inputFrame.rgba();
                    Mat inputGray = inputFrame.gray();

                    float[] largestFaceRect = new float[4];

                    int numFaces = DetectFaces(inputGray.getNativeObjAddr(), inputRgba.getNativeObjAddr(), largestFaceRect);

                    float[] cameraFrameEmbedding = ExtractFaceEmbedding(inputRgba.getNativeObjAddr());

                    if (cameraFrameEmbedding != null && !faceDataList.isEmpty()) {
                        String matchedName = null;
                        float highestSimilarity = 0.0f;

                        for (FaceData faceData : faceDataList) {
                            float similarity = CalculateSimilarity(faceData.embedding, cameraFrameEmbedding);
                            if (similarity > 0.5 && similarity > highestSimilarity) {
                                highestSimilarity = similarity;
                                matchedName = faceData.name;
                            }
                        }

                        float x = largestFaceRect[0];
                        float y = largestFaceRect[1];

                        if (matchedName != null) {
                            Imgproc.putText(inputRgba, matchedName,
                                    new Point(x, y - 10),
                                    Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(100, 255, 100), 2);
                        } else {
                            Imgproc.putText(inputRgba, "Unknown",
                                    new Point(x, y - 10),
                                    Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 0, 255), 2);
                        }
                    }

                    long endTime = System.nanoTime();
                    long processingTimeMs = (endTime - startTime) / 1_000_000;

                    Imgproc.putText(inputRgba, "Processing Time: " + processingTimeMs + " ms",
                            new Point(10, 30),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 255), 2);

                    if (currentLocation != null) {
                        String gpsText = String.format("Lat: %.5f, Lon: %.5f",
                                currentLocation.getLatitude(), currentLocation.getLongitude());

                        Imgproc.putText(inputRgba, gpsText,
                                new Point(10, 65),
                                Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 255, 255), 2);
                    } else {
                        Imgproc.putText(inputRgba, "Location unavailable",
                                new Point(10, 65),
                                Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 0, 0), 2);
                    }

                    Log.d(TAG, "Processing time per frame: " + processingTimeMs + " ms");

                    return inputRgba;
                }
            };

    @Override
    protected void onResume() {
        super.onResume();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.enableView();

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
                == PackageManager.PERMISSION_GRANTED) {
            startLocationUpdates();
        }
        mainAdapter.startListening();
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();

        fusedLocationClient.removeLocationUpdates(locationCallback);
        mainAdapter.stopListening();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        // Handle image selection from gallery
        if (requestCode == GALLERY_REQUEST_CODE) {
            if (resultCode == RESULT_OK && data != null) {
                Uri imageUri = data.getData();
                if (imageUri != null) {
                    getContentResolver().takePersistableUriPermission(imageUri,
                            Intent.FLAG_GRANT_READ_URI_PERMISSION);

                    try {
                        Mat imgMat = uriToMat(imageUri);
                        float[] newEmbedding = ExtractFaceEmbedding(imgMat.getNativeObjAddr());

                        if (newEmbedding == null) {
                            Log.e(TAG, "No face detected in the selected image.");
                            Toast.makeText(this, "No face detected in the selected image.", Toast.LENGTH_SHORT).show();
                        } else {
                            Log.d(TAG, "Face embedding extracted from gallery image.");

                            // Check for duplicates
                            String matchedName = null;
                            float highestSimilarity = 0.0f;
                            for (FaceData faceData : faceDataList) {
                                float similarity = CalculateSimilarity(faceData.embedding, newEmbedding);
                                if (similarity > 0.8 && similarity > highestSimilarity) {
                                    highestSimilarity = similarity;
                                    matchedName = faceData.name;
                                }
                            }

                            if (matchedName != null) {
                                // Face already exists
                                handleDuplicateFace(matchedName, newEmbedding);
                            } else {
                                // No duplicate, proceed to add new face
                                promptForName(newEmbedding);
                            }
                        }

                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                } else {
                    Log.e(TAG, "Image URI is null");
                }
            } else {
                Log.e(TAG, "No image selected or action canceled");
            }
        }
    }

    private void handleDuplicateFace(String matchedName, float[] newEmbedding) {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Face Already Exists");
        builder.setMessage("A face matching this one is already registered as \"" + matchedName + "\". What would you like to do?");

        builder.setPositiveButton("Update Entry", (dialog, which) -> {
            // Update the existing entry
            updateFaceData(matchedName, newEmbedding);
            Toast.makeText(this, "Face data updated.", Toast.LENGTH_SHORT).show();
        });

        builder.setNegativeButton("Cancel", (dialog, which) -> {
            dialog.cancel();
            Toast.makeText(this, "Operation canceled.", Toast.LENGTH_SHORT).show();
        });

        builder.show();
    }

    private void updateFaceData(String name, float[] newEmbedding) {
        for (FaceData faceData : faceDataList) {
            if (faceData.name.equals(name)) {
                faceData.embedding = newEmbedding;
                saveFaceDataToFirebase(faceData);
                break;
            }
        }
    }

    private void promptForName(float[] embedding) {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Enter Name");

        final EditText input = new EditText(this);
        builder.setView(input);

        builder.setPositiveButton("OK", (dialog, which) -> {
            String name = input.getText().toString().trim();
            if (!name.isEmpty()) {
                // Check if name already exists
                boolean nameExists = false;
                for (FaceData faceData : faceDataList) {
                    if (faceData.name.equalsIgnoreCase(name)) {
                        nameExists = true;
                        break;
                    }
                }

                if (nameExists) {
                    AlertDialog.Builder nameExistsDialog = new AlertDialog.Builder(this);
                    nameExistsDialog.setTitle("Name Already Exists");
                    nameExistsDialog.setMessage("An entry with this name already exists. Do you want to update it?");
                    nameExistsDialog.setPositiveButton("Update", (dialog1, which1) -> {
                        updateFaceData(name, embedding);
                        Toast.makeText(this, "Face data updated.", Toast.LENGTH_SHORT).show();
                    });
                    nameExistsDialog.setNegativeButton("Cancel", (dialog1, which1) -> dialog1.cancel());
                    nameExistsDialog.show();
                } else {
                    // Save new face data
                    FaceData faceData = new FaceData(name, embedding);
                    faceDataList.add(faceData);
                    saveFaceDataToFirebase(faceData);
                    Toast.makeText(this, "Face data saved.", Toast.LENGTH_SHORT).show();
                }
            } else {
                Toast.makeText(this, "Name cannot be empty.", Toast.LENGTH_SHORT).show();
            }
        });
        builder.setNegativeButton("Cancel", (dialog, which) -> dialog.cancel());

        builder.show();
    }

    private Mat uriToMat(Uri uri) throws IOException {
        InputStream in = getContentResolver().openInputStream(uri);
        if (in == null) {
            throw new IOException("Unable to open input stream from URI");
        }
        Bitmap bitmap = BitmapFactory.decodeStream(in);
        in.close();
        if (bitmap == null) {
            throw new IOException("Unable to decode bitmap from URI");
        }
        Mat mat = new Mat();
        Bitmap bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Utils.bitmapToMat(bmp32, mat);
        return mat;
    }

    private void saveFaceDataToFirebase(FaceData faceData) {
        // Convert embedding array to a List<Float>
        List<Float> embeddingList = new ArrayList<>();
        for (float val : faceData.embedding) {
            embeddingList.add(val);
        }

        // Create a MainModel object
        MainModel mainModel = new MainModel(faceData.name, embeddingList);

        // Save to Firebase
        faceDataRef.child(faceData.name).setValue(mainModel);
    }

    private void loadFaceDataList() {
        faceDataRef.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                faceDataList.clear();
                for (DataSnapshot dataSnapshot : snapshot.getChildren()) {
                    try {
                        MainModel mainModel = dataSnapshot.getValue(MainModel.class);
                        if (mainModel != null) {
                            if (mainModel.getEmbedding() != null && mainModel.getEmbedding() instanceof List) {
                                List<Float> embeddingList = mainModel.getEmbedding();
                                float[] embeddingArray = new float[embeddingList.size()];
                                for (int i = 0; i < embeddingList.size(); i++) {
                                    embeddingArray[i] = embeddingList.get(i);
                                }
                                faceDataList.add(new FaceData(mainModel.getName(), embeddingArray));
                            } else {
                                Log.e(TAG, "Embedding is not a list as expected.");
                            }
                        }
                    } catch (DatabaseException e) {
                        Log.e(TAG, "DatabaseException: " + e.getMessage());
                    }
                }
                Log.d(TAG, "Face data list loaded from Firebase.");
                mainAdapter.notifyDataSetChanged();
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {
                Log.e(TAG, "Failed to load face data from Firebase.", error.toException());
            }
        });
    }

    private void startLocationUpdates() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
                != PackageManager.PERMISSION_GRANTED) {
            requestLocationPermission();
        } else {
            LocationRequest locationRequest = LocationRequest.create();
            locationRequest.setInterval(10000); // 10 seconds
            locationRequest.setFastestInterval(1000); // 1 second
            locationRequest.setPriority(LocationRequest.PRIORITY_HIGH_ACCURACY);

            fusedLocationClient.requestLocationUpdates(locationRequest,
                    locationCallback,
                    Looper.getMainLooper());
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                initFaceDetectionAndRecognition();
                mOpenCvCameraView.enableView();
            } else {
                Toast.makeText(this, "Camera permission is required for face detection.", Toast.LENGTH_LONG).show();
                finish();
            }
        } else if (requestCode == LOCATION_PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startLocationUpdates();
            } else {
                Toast.makeText(this, "Location permission is required to display GPS coordinates.", Toast.LENGTH_LONG).show();
            }
        }
    }

    private void displayRegisteredFaces() {
        if (faceDataList.isEmpty()) {
            Toast.makeText(this, "No faces registered.", Toast.LENGTH_SHORT).show();
            return;
        }

        // Use a ListView to display the faces
        ListView listView = new ListView(this);

        // Prepare data for the adapter
        List<String> names = new ArrayList<>();
        for (FaceData faceData : faceDataList) {
            names.add(faceData.name);
        }

        // Create an ArrayAdapter
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, names);
        listView.setAdapter(adapter);

        // Set item click listener to show embeddings
        listView.setOnItemClickListener((parent, view, position, id) -> {
            FaceData selectedFace = faceDataList.get(position);

            // Show embedding details
            AlertDialog.Builder embeddingDialog = new AlertDialog.Builder(this);
            embeddingDialog.setTitle("Embedding for " + selectedFace.name);

            StringBuilder sb = new StringBuilder();
            sb.append("[");
            for (int i = 0; i < selectedFace.embedding.length; i++) {
                sb.append(String.format("%.4f", selectedFace.embedding[i]));
                if (i < selectedFace.embedding.length - 1) {
                    sb.append(", ");
                }
            }
            sb.append("]");

            String embeddingString = sb.toString();

            embeddingDialog.setMessage(embeddingString);

            // Add a "Copy Embedding" button
            embeddingDialog.setNeutralButton("Copy Embedding", (dialogInterface, i) -> {
                ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
                ClipData clip = ClipData.newPlainText("", embeddingString);
                clipboard.setPrimaryClip(clip);
                Toast.makeText(this, "Embedding copied to clipboard.", Toast.LENGTH_SHORT).show();
            });

            embeddingDialog.setPositiveButton("Close", (dialog, which) -> dialog.dismiss());
            embeddingDialog.show();
        });

        // Display the ListView in a dialog
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Registered Faces");
        builder.setView(listView);
        builder.setNegativeButton("Close", (dialog, which) -> dialog.dismiss());
        builder.show();
    }
}
