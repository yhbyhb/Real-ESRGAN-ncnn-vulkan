// realesrgan implemented with ncnn library

#ifndef REALESRGAN_H
#define REALESRGAN_H

#include <string>

#include "sr_ncnn_model.h"
// ncnn
#include "net.h"
#include "gpu.h"
#include "layer.h"

class RealESRGAN : SRNCNNModel
{
public:
    RealESRGAN(int gpuid, int tilesize, bool tta_mode = false);
    ~RealESRGAN() override;

#if _WIN32
    int load(const std::wstring& parampath, const std::wstring& modelpath) override;
#else
    int load(const std::string& parampath, const std::string& modelpath);
#endif

    int process(const ncnn::Mat& inimage, ncnn::Mat& outimage) const override;
    int get_scale() override;
    int get_tilesize() override;

private:
    ncnn::Net net;
    ncnn::Pipeline* realesrgan_preproc;
    ncnn::Pipeline* realesrgan_postproc;
    ncnn::Layer* bicubic_2x;
    ncnn::Layer* bicubic_3x;
    ncnn::Layer* bicubic_4x;

    int gpuid;
    bool tta_mode;
    // realesrgan parameters        
    int scale;
    int tilesize;
    int prepadding;
};

#endif // REALESRGAN_H
