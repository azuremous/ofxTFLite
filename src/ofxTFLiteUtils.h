/*-----------------------------------------------------------/
 ofxTFLiteUtils.h

github.com/azuremous
Created by Jung un Kim a.k.a azuremous on 4/15/24.
/----------------------------------------------------------*/
#pragma once

#include "ofMain.h"

class ofxTFLiteUtils : public ofThread{
private:
    bool needNomalize;
    bool needSwap;
    bool useBGR;
    
    int inputImgWidth;
    int inputImgHeight;
    
    ofThreadChannel<ofPixels> originImageData;
    ofThreadChannel<vector<float> > paddedImageData;
    
protected:
    ofPixels getResizedPixel(const ofPixels &inputPixels);
    
    vector<float>pixelsToFloats(const ofPixels & pixels);
    
    void threadedFunction();
    
public:
    ofEvent<vector<float> >preProcessedData;
    
    ofxTFLiteUtils();

    virtual ~ofxTFLiteUtils();
    
    void update();

    void preProcess(const ofPixels & pixels, int inputWidth, int inputHeight, bool needNomalize = false, bool needSwap = false, bool useBGR = false);
};
