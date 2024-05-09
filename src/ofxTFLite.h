/*-----------------------------------------------------------/
 ofxTFLite.h

github.com/azuremous
Created by Jung un Kim a.k.a azuremous on 4/15/24.
/----------------------------------------------------------*/

#pragma once

#include "ofxTFLiteUtils.h"
#include "TensorFlowLiteC.h"

class ofxTFLite {
private:
    ofxTFLiteUtils utils;
    
    TfLiteInterpreterOptions* options;
    TfLiteInterpreter* interpreter;
    TfLiteModel* model;
    TfLiteTensor* input_tensor;
    const TfLiteTensor* output_tensor;
    
    vector<int> inputDims;
    vector<int> outputDims;
    
    string moduleName;
protected:
    int GetTensorSize(const TfLiteTensor* tensor);
    
    void checkTensorInfo(const TfLiteTensor * tensor, vector<int> &dimInfo);
    
    void setTensor(vector<float> &inputData);
    
    void invoke();
    
    void getTensor();
    
    bool TFLiteStatus(TfLiteStatus status);
    
public:
    ofEvent<vector<float> >outputData;
    
    ofxTFLite();
    
    virtual ~ofxTFLite();
    
    bool setup(string path, int threadSize = 0);
    
    void update();
    
    void setImageToTensor(const ofPixels & pixels, int inputWidthID, int inputHeightID, bool needNomalize = false, bool needSwap = false, bool useBGR = false);
    
    void setDataToTensor(const vector<float> &inputData);
    
    vector<int> getOutputDims();
    vector<int> getInputDims();
    
    float * rawInputData();
    float * rawOutputData();
};
