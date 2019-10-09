/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2018 University of Maryland, College Park
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file multi_proposal_target.cc
 * \brief Proposal target layer
 * \author Zhe Wu, Bharat Singh
*/

#include "./multi_proposal_target-inl.h"
#include <set>
#include <math.h>
#include <unistd.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include "./operator_common.h"
#include "./mshadow_op.h"
#include <time.h>
#include <stdlib.h> 
//============================
// Bounding Box Transform Utils
//============================

#define NUM_THREADS_NMS 1024

namespace mxnet {
namespace op {
namespace utils {



// filter box by set confidence to zero
// * height or width < rpn_min_size
inline void FilterBox(float *dets,
                      int num_dets, float min_size) {
  #pragma omp parallel for num_threads(8)
  for (int i = 0; i < num_dets; ++i) {
    float iw = dets[5*i + 2] - dets[5*i] + 1.0f;
    float ih = dets[5*i + 3] - dets[5*i + 1] + 1.0f;
    if (iw < min_size || ih < min_size) {
      dets[5*i+0] -= min_size / 2;
      dets[5*i+1] -= min_size / 2;
      dets[5*i+2] += min_size / 2;
      dets[5*i+3] += min_size / 2;
      dets[5*i+4] = -1.0f;
    }
  }
}


inline void _MakeAnchor(float w,
                        float h,
                        float x_ctr,
                        float y_ctr,
                        std::vector<float> *out_anchors) {
  out_anchors->push_back(x_ctr - 0.5f * (w - 1.0f));
  out_anchors->push_back(y_ctr - 0.5f * (h - 1.0f));
  out_anchors->push_back(x_ctr + 0.5f * (w - 1.0f));
  out_anchors->push_back(y_ctr + 0.5f * (h - 1.0f));
}

inline void _Transform(float scale,
                       float ratio,
                       const std::vector<float>& base_anchor,
                       std::vector<float>  *out_anchors) {
  float w = base_anchor[2] - base_anchor[0] + 1.0f;
  float h = base_anchor[3] - base_anchor[1] + 1.0f;
  float x_ctr = base_anchor[0] + 0.5 * (w - 1.0f);
  float y_ctr = base_anchor[1] + 0.5 * (h - 1.0f);
  float size = w * h;
  float size_ratios = std::floor(size / ratio);
  float new_w = std::floor(std::sqrt(size_ratios) + 0.5f) * scale;
  float new_h = std::floor((new_w / scale * ratio) + 0.5f) * scale;

  _MakeAnchor(new_w, new_h, x_ctr,
             y_ctr, out_anchors);
}

// out_anchors must have shape (n, 5), where n is ratios.size() * scales.size()
inline void GenerateAnchors(const std::vector<float>& base_anchor,
                            const nnvm::Tuple<float>& ratios,
                            const nnvm::Tuple<float>& scales,
                            std::vector<float> *out_anchors) {

  for (size_t j = 0; j < ratios.ndim(); ++j) {
    for (size_t k = 0; k < scales.ndim(); ++k) {
      _Transform(scales[k], ratios[j], base_anchor, out_anchors);
    }
  }
}

// greedily keep the max detections
__global__ void NonMaximumSuppressionAndTargetAssignment(float* idets,
                                  int post_nms_top_n,
                                  int num_images,
                                  int num_anchors,
                                  int width,
                                  int height,
                                  float* propsout,
                                  float* labels, 
                                  float* bbox_targets, 
                                  float* bbox_weights,
                                  float* gt_boxes,
                                  float* valid_ranges,
                                  float* ids,
                                  float* dets,
				  float* crowd_boxes,
				  float* label_weights) {
  int pre_nms_top_n = 6000;
  int i = blockIdx.x;
  int t = threadIdx.x;
  int chip_anchors = height*width*num_anchors;
  int multiplier = pre_nms_top_n;
  int num_threads = blockDim.x;
  int chip_index = i*chip_anchors;

  for (int j = t; j < pre_nms_top_n; j = j + num_threads) {
    dets[6*i*multiplier + 6*j] = idets[chip_index*6 + 6*(int)ids[chip_index + j]];
    dets[6*i*multiplier + 6*j+1] = idets[chip_index*6 + 6*(int)ids[chip_index + j]+1];
    dets[6*i*multiplier + 6*j+2] = idets[chip_index*6 + 6*(int)ids[chip_index + j]+2];
    dets[6*i*multiplier + 6*j+3] = idets[chip_index*6 + 6*(int)ids[chip_index + j]+3];
    dets[6*i*multiplier + 6*j+4] = idets[chip_index*6 + 6*(int)ids[chip_index + j]+4];
    dets[6*i*multiplier + 6*j+5] = idets[chip_index*6 + 6*(int)ids[chip_index + j]+5];
  }  
  __syncthreads();

  int vct = 0;
  __shared__ int keeps[300];
  chip_index = i*multiplier;


  for (int j = chip_index; j < chip_index + pre_nms_top_n && vct < post_nms_top_n; j++) {
    if (dets[6*j+4] == -1) {
      continue;
    }
    float ix1 = dets[6*j];
    float iy1 = dets[6*j+1];
    float ix2 = dets[6*j+2];
    float iy2 = dets[6*j+3];
    float iarea = dets[6*j+5];

    if (t == 0) {
      keeps[vct] = j;
    }

    vct = vct + 1;
    float xx1, xx2, yy1, yy2, w, h, inter, ovr;
    for (int pind = j + 1 + t; pind < chip_index + pre_nms_top_n; pind = pind + num_threads) {
      if (dets[6*pind + 4] == -1) {
        continue;
      } 
      xx1 = fmaxf(ix1, dets[6*pind]);
      yy1 = fmaxf(iy1, dets[6*pind + 1]);
      xx2 = fminf(ix2, dets[6*pind + 2]);
      yy2 = fminf(iy2, dets[6*pind + 3]);
      w = fmaxf(0.0f, xx2 - xx1 + 1.0f);
      h = fmaxf(0.0f, yy2 - yy1 + 1.0f);
      inter = w * h;
      ovr = inter / (iarea + dets[6*pind+5] - inter);
      if (ovr > 0.7) {
        dets[6*pind + 4] = -1;
      }
    }
    __syncthreads();
  }
  
  //set default values and assign gt boxes
  if (t < post_nms_top_n) {
    if (t < vct) {
      propsout[5*(i*post_nms_top_n + t)] = i;
      propsout[5*(i*post_nms_top_n + t) + 1] = dets[6*keeps[t]];
      propsout[5*(i*post_nms_top_n + t) + 2] = dets[6*keeps[t]+1];
      propsout[5*(i*post_nms_top_n + t) + 3] = dets[6*keeps[t]+2];
      propsout[5*(i*post_nms_top_n + t) + 4] = dets[6*keeps[t]+3];
    } else {
      propsout[5*(i*post_nms_top_n + t)] = i;
      propsout[5*(i*post_nms_top_n + t) + 1] = t % 100;
      propsout[5*(i*post_nms_top_n + t) + 2] = t % 100;
      propsout[5*(i*post_nms_top_n + t) + 3] = (t % 100) + 200;
      propsout[5*(i*post_nms_top_n + t) + 4] = (t % 100) + 200;
    }

    labels[i*post_nms_top_n + t] = 0;
    bbox_targets[4*(i*post_nms_top_n + t)] = 0;
    bbox_targets[4*(i*post_nms_top_n + t)+1] = 0;
    bbox_targets[4*(i*post_nms_top_n + t)+2] = 0;
    bbox_targets[4*(i*post_nms_top_n + t)+3] = 0;

    bbox_weights[4*(i*post_nms_top_n + t)] = 0;
    bbox_weights[4*(i*post_nms_top_n + t)+1] = 0;
    bbox_weights[4*(i*post_nms_top_n + t)+2] = 0;
    bbox_weights[4*(i*post_nms_top_n + t)+3] = 0;

    if (gt_boxes[5*(i*100 + t) + 4] != -1 && t < 100) {
      float x1 = gt_boxes[5*(i*100 + t)];
      float y1 = gt_boxes[5*(i*100 + t)+1];
      float x2 = gt_boxes[5*(i*100 + t)+2];
      float y2 = gt_boxes[5*(i*100 + t)+3];

      float area = (x2 - x1) * (y2 - y1);
      if (area < valid_ranges[2*i + 1]*valid_ranges[2*i + 1] && area >= valid_ranges[2*i]*valid_ranges[2*i]) {
        propsout[5*(i*post_nms_top_n + post_nms_top_n - t - 1) + 1] = x1;
        propsout[5*(i*post_nms_top_n + post_nms_top_n - t - 1) + 2] = y1;
        propsout[5*(i*post_nms_top_n + post_nms_top_n - t - 1) + 3] = x2;
        propsout[5*(i*post_nms_top_n + post_nms_top_n - t - 1) + 4] = y2;
      }
    }

  }
  __syncthreads();
  
  if (t < post_nms_top_n) {
    float x1 = propsout[5*(i*post_nms_top_n + t) + 1];
    float y1 = propsout[5*(i*post_nms_top_n + t) + 2];
    float x2 = propsout[5*(i*post_nms_top_n + t) + 3];
    float y2 = propsout[5*(i*post_nms_top_n + t) + 4];
    float xx1, xx2, yy1, yy2, w, h, a2;
    float a1 = (x2 - x1) * (y2 - y1);
    float maxovr = 0, inter, ovr;
    int maxid = 0;
    int j = 0;    

    while(crowd_boxes[5*(10*i + j) + 4] == 0 && j < 10) {
      xx1 = fmaxf(x1, crowd_boxes[5*(10*i + j)]);
      yy1 = fmaxf(y1, crowd_boxes[5*(10*i + j) + 1]);
      xx2 = fminf(x2, crowd_boxes[5*(10*i + j) + 2]);
      yy2 = fminf(y2, crowd_boxes[5*(10*i + j) + 3]);
      w = fmaxf(0.0f, xx2 - xx1 + 1.0f);
      h = fmaxf(0.0f, yy2 - yy1 + 1.0f);
      inter = w * h;
      ovr = inter / (a1 + 1);
      if (ovr > 0.9) {
      	labels[i*post_nms_top_n + t] = -1;
      	break;
      }      
      j = j + 1;
    }


    j = 0;
    while(gt_boxes[5*(100*i + j) + 4] != -1) {
      xx1 = fmaxf(x1, gt_boxes[5*(100*i + j)]);
      yy1 = fmaxf(y1, gt_boxes[5*(100*i + j) + 1]);
      xx2 = fminf(x2, gt_boxes[5*(100*i + j) + 2]);
      yy2 = fminf(y2, gt_boxes[5*(100*i + j) + 3]);
      w = fmaxf(0.0f, xx2 - xx1 + 1.0f);
      h = fmaxf(0.0f, yy2 - yy1 + 1.0f);
      a2 = (gt_boxes[5*(100*i + j) + 3] - gt_boxes[5*(100*i + j) + 1]) * (gt_boxes[5*(100*i + j) + 2] - gt_boxes[5*(100*i + j)]);
      inter = w * h;
      ovr = inter / (a1 + a2 - inter);
      if (ovr > maxovr) {
	maxovr = ovr;
	if (ovr > 0.5){
            maxid = j;
	}
      }
      j = j + 1;
    }
        
    
    float sigma1 = 0.25;
    float sigma2 = 50.0;
    float sigma3 = 20.0;

    if (maxovr < 0.5) {
      label_weights[i*post_nms_top_n + t] = sigma1 + (1-sigma1) * expf(-sigma2*expf(-sigma3*maxovr));
    } else {
      label_weights[i*post_nms_top_n + t] = 1;
    }

    if (maxovr >= 0.5) {
      labels[i*post_nms_top_n + t] = gt_boxes[500*i + 5*maxid + 4];
      
      bbox_weights[4*(i*post_nms_top_n + t)] = 1;
      bbox_weights[4*(i*post_nms_top_n + t)+1] = 1;
      bbox_weights[4*(i*post_nms_top_n + t)+2] = 1;
      bbox_weights[4*(i*post_nms_top_n + t)+3] = 1;

      float gx1 = gt_boxes[i*500 + maxid*5];
      float gy1 = gt_boxes[i*500 + maxid*5 + 1];
      float gx2 = gt_boxes[i*500 + maxid*5 + 2];
      float gy2 = gt_boxes[i*500 + maxid*5 + 3];

      float gw = gx2 - gx1 + 1;
      float gh = gy2 - gy1 + 1;
      float gcx = gx1 + gw*0.5;
      float gcy = gy1 + gh*0.5;

      float pw = x2 - x1 + 1;
      float ph = y2 - y1 + 1;
      float pcx = x1 + (pw-1)*0.5;
      float pcy = y1 + (ph-1)*0.5;

      bbox_targets[4*(i*post_nms_top_n + t)] = 10 * (gcx - pcx) / (pw + 1e-7);
      bbox_targets[4*(i*post_nms_top_n + t)+1] = 10 * (gcy - pcy) / (ph + 1e-7);
      bbox_targets[4*(i*post_nms_top_n + t)+2] = 5 * log(gw/(pw + 1e-7));
      bbox_targets[4*(i*post_nms_top_n + t)+3] = 5 * log(gh/(ph + 1e-7));
    }
  }
  __syncthreads();
}


__global__ void getProps(float* boxes,
                             float* deltas,
                             float* im_info,
                             float* anchorbuf,
                             float* scores,
                             float* valid_ranges,
                             int num_images,
                             int anchors,
                             int heights,
                             int widths,
                             int stride,
                             float*  scorebuf,
                             float* scoreids) {
  int num_anchors = anchors * heights * widths;
  int t = blockDim.x * blockIdx.x + threadIdx.x;

  if (t < num_images * num_anchors) {
    
    int b = t / num_anchors;
    int index = t % num_anchors;
    int a = index / (heights*widths);
    int mat = index % (heights*widths);
    int w = mat % widths; //width index
    int h = mat / widths; //height index
    boxes[6*t] = anchorbuf[4*a] + w * stride;
    boxes[6*t + 1] = anchorbuf[4*a+1] + h * stride;
    boxes[6*t + 2] = anchorbuf[4*a+2] + w * stride;
    boxes[6*t + 3] = anchorbuf[4*a+3] + h * stride;
    boxes[6*t + 4] = scores[b*num_anchors*2 + ((anchors + a)*heights + h)*widths + w];

    float width = boxes[6*t + 2] - boxes[6*t] + 1.0;
    float height = boxes[6*t + 3] - boxes[6*t + 1] + 1.0;
    float ctr_x = boxes[6*t + 0] + 0.5 * (width - 1.0);
    float ctr_y = boxes[6*t + 1] + 0.5 * (height - 1.0);
    float dx = deltas[b*num_anchors*4 + a*4*widths*heights + h*widths + w];
    float dy = deltas[b*num_anchors*4 + (a*4 + 1)*widths*heights + h*widths + w];
    float dw = deltas[b*num_anchors*4 + (a*4 + 2)*widths*heights + h*widths + w];
    float dh = deltas[b*num_anchors*4 + (a*4 + 3)*widths*heights + h*widths + w];
    float pred_ctr_x = dx * width + ctr_x;
    float pred_ctr_y = dy * height + ctr_y;
    float pred_w = exp(dw) * width;
    float pred_h = exp(dh) * height;
    float pred_x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
    float pred_y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
    float pred_x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
    float pred_y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);

    pred_x1 = fmaxf(fminf(pred_x1, im_info[3*b+1] - 1.0f), 0.0f);
    pred_y1 = fmaxf(fminf(pred_y1, im_info[3*b] - 1.0f), 0.0f);
    pred_x2 = fmaxf(fminf(pred_x2, im_info[3*b+1] - 1.0f), 0.0f);
    pred_y2 = fmaxf(fminf(pred_y2, im_info[3*b] - 1.0f), 0.0f);
    boxes[6*t] = pred_x1;
    boxes[6*t + 1] = pred_y1;
    boxes[6*t + 2] = pred_x2;
    boxes[6*t + 3] = pred_y2;
    
    int min_size = 3;
    if ((pred_y2 - pred_y1) < min_size && (pred_x2 - pred_x1) < min_size) {
      boxes[6*t] -= min_size/2;
      boxes[6*t + 1] -= min_size/2;
      boxes[6*t + 2] += min_size/2;
      boxes[6*t + 3] += min_size/2;
      boxes[6*t + 4] = -1;
    }
    float area = (boxes[6*t + 2] - boxes[6*t]) * (boxes[6*t + 3] - boxes[6*t + 1]);
    if (area >= valid_ranges[2*b+1] * valid_ranges[2*b+1] || area < valid_ranges[2*b]*valid_ranges[2*b]) {
      boxes[6*t + 4] = -1;  
    }
    boxes[6*t + 5] = area;
    scorebuf[t] = boxes[6*t + 4];
    scoreids[t] = index;
  }
}

}  // namespace utils


template<typename xpu>
class MultiProposalTargetGPUOp : public Operator{
 public:
  
  explicit MultiProposalTargetGPUOp(MultiProposalTargetParam param) {
    this->param_ = param;
    this->param_.workspace = (param_.workspace << 20) / sizeof(float);
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    CHECK_EQ(in_data.size(), 6);
    CHECK_EQ(out_data.size(), 5);
    
    using namespace mshadow;
    using namespace mshadow::expr;
    //clock_t t;
    //t = clock();
    Stream<gpu> *s = ctx.get_stream<gpu>();

    Tensor<gpu, 4> tscores = in_data[proposal::kClsProb].get<gpu, 4, real_t>(s);
    Tensor<gpu, 4> tbbox_deltas = in_data[proposal::kBBoxPred].get<gpu, 4, real_t>(s);
    Tensor<gpu, 2> tim_info = in_data[proposal::kImInfo].get<gpu, 2, real_t>(s);
    Tensor<gpu, 3> tgt_boxes = in_data[proposal::kGTBoxes].get<gpu, 3, real_t>(s);
    Tensor<gpu, 2> tvalid_ranges = in_data[proposal::kValidRanges].get<gpu, 2, real_t>(s);
    Tensor<gpu, 3> tcrowd_boxes = in_data[proposal::kCrowdBoxes].get<gpu, 3, real_t>(s);    


    Tensor<gpu, 2> rois = out_data[proposal::kRoIs].get<gpu, 2, real_t>(s);
    Tensor<gpu, 2> labels = out_data[proposal::kLabels].get<gpu, 2, real_t>(s);
    Tensor<gpu, 2> bbox_targets = out_data[proposal::kBboxTarget].get<gpu, 2, real_t>(s);
    Tensor<gpu, 2> bbox_weights = out_data[proposal::kBboxWeight].get<gpu, 2, real_t>(s);
    Tensor<gpu, 2> label_weights = out_data[proposal::kLabelWeight].get<gpu, 2, real_t>(s);

    int rpn_post_nms_top_n = param_.rpn_post_nms_top_n;
    int num_images = tbbox_deltas.size(0);
    int num_anchors = tbbox_deltas.size(1) / 4;
    int height = tbbox_deltas.size(2);
    int width = tbbox_deltas.size(3);
    int count_anchors = num_anchors*height*width;
    int total_anchors = count_anchors * num_images;

    int pre_nms_top_n = 6000;
    int bufsize = (total_anchors*8 + num_images * 6 * pre_nms_top_n + num_anchors*4)*sizeof(float);
    Tensor<gpu, 1> workspace = ctx.requested[proposal::kTempSpace].get_space_typed<gpu, 1, float>(Shape1(bufsize), s);

    float* propbuf = workspace.dptr_;
    float* scorebuf = workspace.dptr_ + total_anchors*6;
    float* idbuf = workspace.dptr_ + total_anchors*7;
    float* detbuf = workspace.dptr_ + total_anchors*8;
    float* anchorbuf = workspace.dptr_ + total_anchors*8 + num_images * 6 * pre_nms_top_n;
    

    std::vector<float> base_anchor(4);
    //usleep(20000000);
    base_anchor[0] = 0.0;
    base_anchor[1] = 0.0;
    base_anchor[2] = param_.feature_stride - 1.0;
    base_anchor[3] = param_.feature_stride - 1.0;

    std::vector<float> anchors;
    utils::GenerateAnchors(base_anchor,
                           param_.ratios,
                           param_.scales,
                           &anchors);
    unsigned int size = num_anchors*4*sizeof(float);
    cudaMemcpy(anchorbuf, &anchors[0], size, cudaMemcpyHostToDevice);

    //call cuda kernel
    int threadsPerBlock = NUM_THREADS_NMS; 
    int numblocks = (total_anchors/threadsPerBlock) + 1;
    utils::getProps<<<numblocks, threadsPerBlock>>>(propbuf, tbbox_deltas.dptr_, tim_info.dptr_, anchorbuf, tscores.dptr_,
                                                    tvalid_ranges.dptr_, num_images, num_anchors, height, width, param_.feature_stride, scorebuf, idbuf);
    std::vector <float> tmp(total_anchors);
    std::vector<float> ids(total_anchors);    

    cudaDeviceSynchronize();
    cudaMemcpy(&tmp[0], scorebuf, sizeof(float) * num_images * count_anchors, cudaMemcpyDeviceToHost);

    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < total_anchors; i++) {
      ids[i] = (float)(i % count_anchors);
    }
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < num_images; i++) {
      float basep = count_anchors*i;
      std::sort(ids.begin() + i*count_anchors, ids.begin() + (i+1)*count_anchors, 
          [&tmp, basep](float i1, float i2) {
            return tmp[(int)i1 + basep] > tmp[(int)i2 + basep];
          });
    }

    cudaMemcpy(idbuf, &ids[0], sizeof(float) * num_images * count_anchors, cudaMemcpyHostToDevice);

    utils::NonMaximumSuppressionAndTargetAssignment<<<num_images, threadsPerBlock>>>(propbuf, rpn_post_nms_top_n, num_images, num_anchors, width, height, 
                                                                  rois.dptr_, labels.dptr_, bbox_targets.dptr_, bbox_weights.dptr_, 
                                                                  tgt_boxes.dptr_, tvalid_ranges.dptr_, idbuf, detbuf, 
                                                                  tcrowd_boxes.dptr_, label_weights.dptr_);
    cudaDeviceSynchronize();
    cudaError_t error;
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
    }
    
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_grad.size(), 6);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> gscores = in_grad[proposal::kClsProb].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gbbox = in_grad[proposal::kBBoxPred].get<xpu, 4, real_t>(s);
    Tensor<xpu, 2> ginfo = in_grad[proposal::kImInfo].get<xpu, 2, real_t>(s);
    Tensor<xpu, 3> ggt_boxes = in_grad[proposal::kGTBoxes].get<xpu, 3, real_t>(s);
    Tensor<xpu, 2> gvalid_ranges = in_grad[proposal::kValidRanges].get<xpu, 2, real_t>(s);
    Tensor<xpu, 3> gcrowd_boxes = in_grad[proposal::kCrowdBoxes].get<xpu, 3, real_t>(s);    

    // can not assume the grad would be zero
    Assign(gscores, req[proposal::kClsProb], 0);
    Assign(gbbox, req[proposal::kBBoxPred], 0);
    Assign(ginfo, req[proposal::kImInfo], 0);
    Assign(ggt_boxes, req[proposal::kGTBoxes], 0);
    Assign(gcrowd_boxes, req[proposal::kCrowdBoxes], 0);    
    Assign(gvalid_ranges, req[proposal::kValidRanges], 0);
  }

 private:
  MultiProposalTargetParam param_;
};  // class MultiProposalOp

template<>
Operator *CreateOp<gpu>(MultiProposalTargetParam param) {
  return new MultiProposalTargetGPUOp<gpu>(param);
}


}  // namespace op
}  // namespace mxnet
