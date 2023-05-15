// Interface for Super Resolution NCNN model

#ifndef SR_NCNN_MODEL_H
#define SR_NCNN_MODEL_H

#include <string>

#include "net.h"

//abstract(or interface) class for super resolution ncnn model
class SRNCNNModel
{
public:
    virtual ~SRNCNNModel() = 0 {};
    virtual int load(const std::wstring& parampath, const std::wstring& modelpath) = 0;
    virtual int process(const ncnn::Mat& inimage, ncnn::Mat& outimage) const = 0;
    virtual int GetScale() = 0;
};

#endif // SR_NCNN_MODEL_H
