TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
#CONFIG -= qt

# OpenCV paths for headers and libs
INCLUDEPATH += C:\opencv\build\include


LIBS += C:\opencv-build\bin\libopencv_core320.dll
LIBS += C:\opencv-build\bin\libopencv_imgcodecs320.dll
LIBS += C:\opencv-build\bin\libopencv_imgproc320.dll
LIBS += C:\opencv-build\bin\libopencv_highgui320.dll
LIBS += C:\opencv-build\bin\libopencv_ml320.dll
LIBS += C:\opencv-build\bin\libopencv_video320.dll
LIBS += C:\opencv-build\bin\libopencv_videoio320.dll
LIBS += C:\opencv-build\bin\libopencv_features2d320.dll
LIBS += C:\opencv-build\bin\libopencv_calib3d320.dll
LIBS += C:\opencv-build\bin\libopencv_objdetect320.dll
LIBS += C:\opencv-build\bin\libopencv_flann320.dll
# --- ENDOF --- OpenCV paths for headers and libs

SOURCES += main.cpp
