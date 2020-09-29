cmake_minimum_required(VERSION 3.4.1)

set(EXECUTABLE_OUTPUT_PATH      "${CMAKE_CURRENT_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}")

add_library( # Sets the name of the library.
             native-lib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             src/main/cpp/native-lib.cpp )

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

include_directories(C:/Users/malik/Desktop/Libs_and_Jars/OpenCV/OpenCV-android-sdk/sdk/native/jni/include)
add_library( lib-opencv SHARED IMPORTED )
set_target_properties(lib-opencv PROPERTIES IMPORTED_LOCATION  ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libopencv_java3.so)

target_link_libraries( # Specifies the target library.
                       native-lib

                       lib-opencv

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib} )