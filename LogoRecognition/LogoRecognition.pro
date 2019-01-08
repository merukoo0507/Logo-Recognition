TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

INCLUDEPATH += C:\QtBuild\release\include

LIBS += -LC:\QtBuild\release\x64\mingw\lib \
        -llibopencv_calib3d248      \
        -llibopencv_contrib248      \
        -llibopencv_core248         \
        -llibopencv_features2d248   \
        -llibopencv_flann248        \
        -llibopencv_gpu248          \
        -llibopencv_highgui248      \
        -llibopencv_imgproc248      \
        -llibopencv_legacy248       \
        -llibopencv_ml248           \
        -llibopencv_nonfree248      \
        -llibopencv_objdetect248    \
        -llibopencv_photo248        \
        -llibopencv_stitching248    \
        -llibopencv_video248        \
        -llibopencv_videostab248    \
