
#include "ofxTFlite.h"

//--------------------------------------------------------------
ofxTFLite::ofxTFLite():moduleName("ofxTFLite")
{
    
}

//--------------------------------------------------------------
ofxTFLite::~ofxTFLite(){
    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);
}

//--------------------------------------------------------------

bool ofxTFLite::setup(string path, int threadSize){
    
    string modelRoute = ofToDataPath(path, true);
    ofLogVerbose(moduleName, "load model : " + modelRoute);
    model = TfLiteModelCreateFromFile(modelRoute.c_str());
    options = TfLiteInterpreterOptionsCreate();
    if(threadSize > 0){
        TfLiteInterpreterOptionsSetNumThreads(options, threadSize);
    }
    // Create the interpreter.
    interpreter = TfLiteInterpreterCreate(model, options);
    
    if(TFLiteStatus(TfLiteInterpreterAllocateTensors(interpreter))){
        input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
        ofLogVerbose(moduleName, "Input Tensor-----------------");
        checkTensorInfo(input_tensor, inputDims);
        output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
        ofLogVerbose(moduleName, "output Tensor-----------------");
        checkTensorInfo(output_tensor, outputDims);
        ofAddListener(utils.preProcessedData, this, &ofxTFLite::setTensor);
        return true;
    }
    return false;
}

//--------------------------------------------------------------
void ofxTFLite::update(){
    utils.update();
}

//--------------------------------------------------------------
void ofxTFLite::setImageToTensor(const ofPixels & pixels, int inputWidthID, int inputHeightID, bool needNomalize, bool needSwap, bool useBGR){
    
    utils.preProcess(pixels, inputDims[inputWidthID], inputDims[inputHeightID], needNomalize, needSwap, useBGR);
}

//--------------------------------------------------------------
void ofxTFLite::setDataToTensor(const vector<float> &inputData){
    if(!TFLiteStatus(TfLiteTensorCopyFromBuffer(input_tensor, inputData.data(), inputData.size() * sizeof(float)))){
        return;
    }
    invoke();
}

//--------------------------------------------------------------
vector<int> ofxTFLite::getInputDims() {
    return inputDims;
}

//--------------------------------------------------------------
vector<int> ofxTFLite::getOutputDims() {
    return outputDims;
}

//--------------------------------------------------------------
float * ofxTFLite::rawInputData() {
    return input_tensor->data.f;
}

//--------------------------------------------------------------
float * ofxTFLite::rawOutputData() {
    return output_tensor->data.f;
}

//--------------------------------------------------------------
int ofxTFLite::GetTensorSize(const TfLiteTensor* tensor){
    return TfLiteTensorByteSize(tensor) / sizeof(float);
}

//--------------------------------------------------------------
void ofxTFLite::checkTensorInfo(const TfLiteTensor * tensor, vector<int> &dimInfo){
    ofLogVerbose(moduleName, "Tensor Size : "+ofToString(GetTensorSize(tensor)));
    
    int numDims = TfLiteTensorNumDims(tensor);
    
    ofLogVerbose(moduleName, "number of dimensions : "+ofToString(numDims));
    
    string dimData = "";
    for(int i = 0; i < numDims; i++){
        int dim = TfLiteTensorDim(tensor, i);
        dimData.append(ofToString(dim));
        if(i != numDims -1) { dimData.append(", "); }
        dimInfo.push_back(dim);
    }
    
    ofLogVerbose(moduleName, "dimension : " + dimData);
    ofLogVerbose(moduleName, "the size of the underlying data in bytes : " + ofToString(TfLiteTensorByteSize(tensor)));
    
    TfLiteQuantization quantization = tensor->quantization;
    
    if(quantization.type == 1){
        TfLiteQuantizationParams param = TfLiteTensorQuantizationParams(tensor);
        ofLogVerbose(moduleName, "Quantized scale : " + ofToString(param.scale));
        ofLogVerbose(moduleName, "Quantized zero_point : " + ofToString(param.zero_point));
        
    }
}

//--------------------------------------------------------------
void ofxTFLite::setTensor(vector<float> &inputData){
    if(!TFLiteStatus(TfLiteTensorCopyFromBuffer(input_tensor, inputData.data(), inputData.size() * sizeof(float)))){
        return;
    }
    invoke();
}

//--------------------------------------------------------------
void ofxTFLite::invoke(){
    if(!TFLiteStatus(TfLiteInterpreterInvoke(interpreter))){
        return;
    }
    getTensor();
}

//--------------------------------------------------------------
void ofxTFLite::getTensor(){
    vector<float>output_data(GetTensorSize(output_tensor), 0);
    if(!TFLiteStatus(TfLiteTensorCopyToBuffer(output_tensor, output_data.data(), output_data.size() * sizeof(float)))){
        return;
    }
    ofNotifyEvent(outputData, output_data);
}

//--------------------------------------------------------------
bool ofxTFLite::TFLiteStatus(TfLiteStatus status){
    if(status == 0) return true;
    switch (status) {
        case 1:
            ofLogError(moduleName, "kTfLiteError");
            break;
            
        case 2:
            ofLogError(moduleName, "kTfLiteDelegateError");
            break;
            
        case 3:
            ofLogError(moduleName, "kTfLiteApplicationError");
            break;
            
        case 4:
            ofLogError(moduleName, "kTfLiteDelegateDataNotFound");
            break;
            
        case 5:
            ofLogError(moduleName, "kTfLiteDelegateDataWriteError");
            break;
            
        case 6:
            ofLogError(moduleName, "kTfLiteDelegateDataReadError");
            break;
            
        case 7:
            ofLogError(moduleName, "kTfLiteUnresolvedOps");
            break;
            
        case 8:
            ofLogError(moduleName, "kTfLiteCancelled");
            break;
            
        default:
            break;
    }
    return false;
}
