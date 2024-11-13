// native-lib.cpp

#include <jni.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect/face.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <android/log.h>

#define LOG_TAG "native-lib"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

using namespace cv;
using namespace std;

extern "C" {

cv::Ptr<cv::FaceDetectorYN> face_detector;
cv::Ptr<cv::FaceRecognizerSF> face_recognition;

struct FaceData {
    std::string name;
    std::vector<float> embedding;
};

std::vector<FaceData> faceDataList;

JNIEXPORT void JNICALL Java_com_example_appcpp_MainActivityCPP_InitFaceDetector(JNIEnv* jniEnv, jobject, jstring jFilePath) {
    const char* jnamestr = jniEnv->GetStringUTFChars(jFilePath, NULL);
    std::string filePath(jnamestr);

    face_detector = cv::FaceDetectorYN::create(
            filePath,
            "",
            cv::Size(320, 320),
            0.5f,
            0.3f,
            5000,
            cv::dnn::DNN_BACKEND_DEFAULT,
            cv::dnn::DNN_TARGET_CPU
    );

    jniEnv->ReleaseStringUTFChars(jFilePath, jnamestr);
}

JNIEXPORT void JNICALL Java_com_example_appcpp_MainActivityCPP_InitFaceRecognition(JNIEnv* jniEnv, jobject, jstring jFilePath) {
    const char* jnamestr = jniEnv->GetStringUTFChars(jFilePath, NULL);
    std::string filePath(jnamestr);

    face_recognition = cv::FaceRecognizerSF::create(
            filePath,
            "",
            0,
            0
    );

    jniEnv->ReleaseStringUTFChars(jFilePath, jnamestr);
}

JNIEXPORT jint JNICALL Java_com_example_appcpp_MainActivityCPP_DetectFaces(JNIEnv* env, jobject, jlong addGray, jlong addrRGBA, jfloatArray largestFaceRectArray) {
    Mat* mGray = (Mat*) addGray;
    Mat* mRGBA = (Mat*) addrRGBA;

    if (face_detector.empty()) {
        return 0;
    }

    Mat mBGR;
    cv::cvtColor(*mRGBA, mBGR, cv::COLOR_RGBA2BGR);

    face_detector->setInputSize(mBGR.size());

    cv::Mat faces;
    face_detector->detect(mBGR, faces);

    int numFaces = faces.rows;

    if (faces.empty()) {
        return 0;
    }

    // Find the largest face
    int largestFaceIndex = 0;
    float maxArea = 0.0f;

    for (int i = 0; i < faces.rows; ++i) {
        float* data = faces.ptr<float>(i);
        float x = data[0];
        float y = data[1];
        float w = data[2];
        float h = data[3];
        float area = w * h;

        cv::Scalar rectColor = cv::Scalar(255, 0, 0); // Red color for other faces
        if (area > maxArea) {
            maxArea = area;
            largestFaceIndex = i;
            rectColor = cv::Scalar(0, 255, 0); // Green color for the largest face
        }

        cv::rectangle(*mRGBA, cv::Rect(cv::Point(x, y), cv::Size(w, h)), rectColor, 2);

        // Optionally, draw landmarks
//        for (int j = 0; j < 5; ++j) {
//            int landmark_x = static_cast<int>(faces.at<float>(i, 2 * j + 4));
//            int landmark_y = static_cast<int>(faces.at<float>(i, 2 * j + 5));
//            cv::circle(*mRGBA, cv::Point(landmark_x, landmark_y), 2, rectColor, 2);
//        }

    }

    // Get the coordinates of the largest face
    float* largestFaceData = faces.ptr<float>(largestFaceIndex);
    float x = largestFaceData[0];
    float y = largestFaceData[1];
    float w = largestFaceData[2];
    float h = largestFaceData[3];

    // Fill the largestFaceRectArray with x, y, w, h
    jfloat tempArray[4];
    tempArray[0] = x;
    tempArray[1] = y;
    tempArray[2] = w;
    tempArray[3] = h;

    env->SetFloatArrayRegion(largestFaceRectArray, 0, 4, tempArray);

    return numFaces;
}

JNIEXPORT jfloatArray JNICALL Java_com_example_appcpp_MainActivityCPP_ExtractFaceEmbedding(JNIEnv* env, jobject, jlong addrMat) {
    if (face_recognition.empty() || face_detector.empty()) {
        return nullptr;
    }

    Mat& img = *(Mat*)addrMat;

    Mat imgBGR;
    cv::cvtColor(img, imgBGR, cv::COLOR_RGBA2BGR);

    face_detector->setInputSize(imgBGR.size());

    Mat faces;
    face_detector->detect(imgBGR, faces);

    if (faces.empty()) {
        return nullptr;
    }

    // Find the largest face
    int largestFaceIndex = 0;
    float maxArea = 0.0f;

    for (int i = 0; i < faces.rows; ++i) {
        float* data = faces.ptr<float>(i);
        float w = data[2];
        float h = data[3];
        float area = w * h;

        if (area > maxArea) {
            maxArea = area;
            largestFaceIndex = i;
        }
    }

    // Process only the largest face
    Mat alignedFace;
    face_recognition->alignCrop(imgBGR, faces.row(largestFaceIndex), alignedFace);

    Mat embedding;
    face_recognition->feature(alignedFace, embedding);

    // Convert embedding to jfloatArray
    jsize len = embedding.cols;
    jfloatArray result = env->NewFloatArray(len);
    env->SetFloatArrayRegion(result, 0, len, (const jfloat*)embedding.ptr<float>());

    return result;
}

JNIEXPORT jfloat JNICALL Java_com_example_appcpp_MainActivityCPP_CalculateSimilarity(JNIEnv* env, jobject, jfloatArray emb1, jfloatArray emb2) {
    jsize len1 = env->GetArrayLength(emb1);
    jsize len2 = env->GetArrayLength(emb2);

    if (len1 != len2) {
        return -1.0f; // Indicate error
    }

    jfloat* emb1_data = env->GetFloatArrayElements(emb1, nullptr);
    jfloat* emb2_data = env->GetFloatArrayElements(emb2, nullptr);

    Mat emb1_mat(1, len1, CV_32F, emb1_data);
    Mat emb2_mat(1, len2, CV_32F, emb2_data);

    float cosine_sim = face_recognition->match(emb1_mat, emb2_mat, cv::FaceRecognizerSF::DisType::FR_COSINE);

    env->ReleaseFloatArrayElements(emb1, emb1_data, 0);
    env->ReleaseFloatArrayElements(emb2, emb2_data, 0);

    return cosine_sim;
}

// New JNI functions to handle face data in C++

JNIEXPORT void JNICALL Java_com_example_appcpp_MainActivityCPP_LoadFaceDataList(JNIEnv* env, jobject, jstring jFilePath) {
    const char* filePathC = env->GetStringUTFChars(jFilePath, NULL);
    std::string filePath(filePathC);

    faceDataList.clear();

    std::ifstream infile(filePath);
    if (!infile.is_open()) {
        LOGD("Face data file not found: %s", filePath.c_str());
        env->ReleaseStringUTFChars(jFilePath, filePathC);
        return;
    }

    std::string line;
    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        FaceData faceData;
        ss >> faceData.name;
        float val;
        while (ss >> val) {
            faceData.embedding.push_back(val);
        }
        faceDataList.push_back(faceData);
    }

    infile.close();
    env->ReleaseStringUTFChars(jFilePath, filePathC);
    LOGD("Face data list loaded from %s", filePath.c_str());
}

JNIEXPORT void JNICALL Java_com_example_appcpp_MainActivityCPP_SaveFaceDataList(JNIEnv* env, jobject, jstring jFilePath) {
    const char* filePathC = env->GetStringUTFChars(jFilePath, NULL);
    std::string filePath(filePathC);

    std::ofstream outfile(filePath);
    if (!outfile.is_open()) {
        LOGD("Failed to open face data file: %s", filePath.c_str());
        env->ReleaseStringUTFChars(jFilePath, filePathC);
        return;
    }

    for (const auto& faceData : faceDataList) {
        outfile << faceData.name;
        for (const auto& val : faceData.embedding) {
            outfile << " " << val;
        }
        outfile << std::endl;
    }

    outfile.close();
    env->ReleaseStringUTFChars(jFilePath, filePathC);
    LOGD("Face data list saved to %s", filePath.c_str());
}

JNIEXPORT jstring JNICALL Java_com_example_appcpp_MainActivityCPP_CheckDuplicateFace(JNIEnv* env, jobject, jfloatArray jEmbedding, jfloat threshold) {
    jsize len = env->GetArrayLength(jEmbedding);
    jfloat* embedding_data = env->GetFloatArrayElements(jEmbedding, nullptr);
    Mat newEmbedding(1, len, CV_32F, embedding_data);

    std::string matchedName;
    float highestSimilarity = 0.0f;

    for (const auto& faceData : faceDataList) {
        Mat existingEmbedding(1, faceData.embedding.size(), CV_32F, (void*)faceData.embedding.data());
        float similarity = face_recognition->match(newEmbedding, existingEmbedding, cv::FaceRecognizerSF::DisType::FR_COSINE);
        if (similarity > threshold && similarity > highestSimilarity) {
            highestSimilarity = similarity;
            matchedName = faceData.name;
        }
    }

    env->ReleaseFloatArrayElements(jEmbedding, embedding_data, 0);

    if (!matchedName.empty()) {
        return env->NewStringUTF(matchedName.c_str());
    } else {
        return nullptr;
    }
}

JNIEXPORT void JNICALL Java_com_example_appcpp_MainActivityCPP_UpdateFaceData(JNIEnv* env, jobject, jstring jName, jfloatArray jEmbedding) {
    const char* nameC = env->GetStringUTFChars(jName, NULL);
    std::string name(nameC);

    jsize len = env->GetArrayLength(jEmbedding);
    jfloat* embedding_data = env->GetFloatArrayElements(jEmbedding, nullptr);

    bool found = false;
    for (auto& faceData : faceDataList) {
        if (faceData.name == name) {
            faceData.embedding.assign(embedding_data, embedding_data + len);
            found = true;
            break;
        }
    }

    if (!found) {
        FaceData newFaceData;
        newFaceData.name = name;
        newFaceData.embedding.assign(embedding_data, embedding_data + len);
        faceDataList.push_back(newFaceData);
    }

    env->ReleaseStringUTFChars(jName, nameC);
    env->ReleaseFloatArrayElements(jEmbedding, embedding_data, 0);
}

}
