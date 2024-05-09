#include "ofxTFLiteUtils.h"

//--------------------------------------------------------------
ofxTFLiteUtils::ofxTFLiteUtils():needNomalize(false),needSwap(false),useBGR(false)
{
    startThread();
}

//--------------------------------------------------------------
ofxTFLiteUtils::~ofxTFLiteUtils(){
    originImageData.close();
    paddedImageData.close();
    waitForThread(true);
}

//--------------------------------------------------------------
void ofxTFLiteUtils::update(){
    vector<float> swapPixelsData;
    while(paddedImageData.tryReceive(swapPixelsData)){
        ofNotifyEvent(preProcessedData, swapPixelsData);
    }
}

//--------------------------------------------------------------
void ofxTFLiteUtils::preProcess(const ofPixels & pixels, int inputWidth, int inputHeight, bool needNomalize, bool needSwap, bool useBGR){
    inputImgWidth = inputWidth;
    inputImgHeight = inputHeight;
    this->needNomalize = needNomalize;
    this->needSwap = needSwap;
    this->useBGR = useBGR;
    originImageData.send(pixels);
}

//--------------------------------------------------------------
ofPixels ofxTFLiteUtils::getResizedPixel(const ofPixels &inputPixels){
    int inputWidth = inputPixels.getWidth();
    int inputHeight = inputPixels.getHeight();
    int targetWidth = inputImgWidth;
    int targetHeight = inputImgHeight;
    
    double ratio = min(static_cast<double>(targetWidth) / inputWidth,
                                static_cast<double>(targetHeight) / inputHeight);
    int resizedWidth = inputWidth * ratio;
    int resizedHeight = inputHeight * ratio;
    ofPixels pixels = inputPixels;
    pixels.setImageType(OF_IMAGE_COLOR);
    
    ofImage resizeImage;
    resizeImage.setUseTexture(false);
    resizeImage.setFromPixels(pixels);
    resizeImage.resize(resizedWidth, resizedHeight);
    
    ofPixels paddedPixels;
    paddedPixels.allocate(targetWidth, targetHeight, pixels.getPixelFormat());
    paddedPixels.setColor(ofColor(114, 114, 114));
    resizeImage.getPixels().pasteInto(paddedPixels, 0, 0);

    return paddedPixels;
}

//--------------------------------------------------------------
vector<float> ofxTFLiteUtils::pixelsToFloats(const ofPixels & pixels){
    ofPixels resizedPixels = getResizedPixel(pixels);
    
    vector<float>normalizePixel;
    float normal = 1.0;
    
    if(needNomalize) { normal = 255.0; }
    for(int i = 0; i < resizedPixels.size(); i++){
        normalizePixel.push_back((float)resizedPixels[i] / normal);
    }
    
    if(!needSwap) { return normalizePixel; }
    
    vector<float>swapPixels;
    int channerOrder[3] = {0, 1, 2};
    if(useBGR){
        channerOrder[0] = 2;
        channerOrder[1] = 1;
        channerOrder[2] = 0;
    }
    for(int j = 0; j < 3; j++){//rgb
        for(int i = 0; i < normalizePixel.size(); i+=3){
            int num = i + channerOrder[j];//0,1,2 or 2,1,0
            swapPixels.push_back(normalizePixel[num]);
        }
    }
    return swapPixels;
}

//--------------------------------------------------------------
void ofxTFLiteUtils::threadedFunction(){
    ofPixels pixels;
    while(originImageData.receive(pixels)){
        paddedImageData.send(pixelsToFloats(pixels));
    }
    
}
