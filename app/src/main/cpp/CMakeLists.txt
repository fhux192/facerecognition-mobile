# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name.
project("facialdetection")

# Creates and names a library.
add_library(${CMAKE_PROJECT_NAME} SHARED
        facialdetection.cpp)

# Set the path to your OpenCV Android SDK (Update this to your SDK location)
set(OpenCV_SDK /home/asus/Downloads/opencv-4.10.0-android-sdk/OpenCV-android-sdk/sdk/native)

# Include OpenCV headers
include_directories(${OpenCV_SDK}/jni/include)

# Add OpenCV static libraries for arm64-v8a ABI
if(${ANDROID_ABI} STREQUAL "armeabi-v7a")
    set(OpenCV_STATIC_LIBS
            ${OpenCV_SDK}/staticlibs/armeabi-v7a/libopencv_core.a
            ${OpenCV_SDK}/staticlibs/armeabi-v7a/libopencv_imgproc.a
            ${OpenCV_SDK}/staticlibs/armeabi-v7a/libopencv_objdetect.a
            # Add other required libraries here
    )
elseif(${ANDROID_ABI} STREQUAL "arm64-v8a")
    set(OpenCV_STATIC_LIBS
            ${OpenCV_SDK}/staticlibs/arm64-v8a/libopencv_core.a
            ${OpenCV_SDK}/staticlibs/arm64-v8a/libopencv_imgproc.a
            ${OpenCV_SDK}/staticlibs/arm64-v8a/libopencv_objdetect.a
            # Add other required libraries here
    )
endif()

# Add OpenCV third-party libraries (JPEG, PNG, TIFF, etc.)
if(${ANDROID_ABI} STREQUAL "armeabi-v7a")
    set(OpenCV_STATIC_LIBS
            ${OpenCV_SDK}/staticlibs/armeabi-v7a/libopencv_core.a
            ${OpenCV_SDK}/staticlibs/armeabi-v7a/libopencv_imgproc.a
            ${OpenCV_SDK}/staticlibs/armeabi-v7a/libopencv_objdetect.a
            ${OpenCV_SDK}/3rdparty/libs/armeabi-v7a/liblibjpeg-turbo.a
            ${OpenCV_SDK}/3rdparty/libs/armeabi-v7a/liblibpng.a
            ${OpenCV_SDK}/3rdparty/libs/armeabi-v7a/liblibtiff.a
            # Add any other 3rdparty libraries needed for your project here
    )
elseif(${ANDROID_ABI} STREQUAL "arm64-v8a")
    set(OpenCV_STATIC_LIBS
            ${OpenCV_SDK}/staticlibs/arm64-v8a/libopencv_core.a
            ${OpenCV_SDK}/staticlibs/arm64-v8a/libopencv_imgproc.a
            ${OpenCV_SDK}/staticlibs/arm64-v8a/libopencv_objdetect.a
            ${OpenCV_SDK}/3rdparty/libs/arm64-v8a/liblibjpeg-turbo.a
            ${OpenCV_SDK}/3rdparty/libs/arm64-v8a/liblibpng.a
            ${OpenCV_SDK}/3rdparty/libs/arm64-v8a/liblibtiff.a
            # Add any other 3rdparty libraries needed for your project here
    )
endif()


find_library( # Sets the name of the path variable.
        log-lib
        log )

# Find jnigraphics library (Android NDK graphics library)
find_library(jnigraphics-lib jnigraphics)

# Specifies libraries CMake should link to your target library.
target_link_libraries(${CMAKE_PROJECT_NAME}
        android
        ${OpenCV_STATIC_LIBS}
        ${OpenCV_3RDPARTY_LIBS}
        ${jnigraphics-lib}
        ${log-lib}
        log)
