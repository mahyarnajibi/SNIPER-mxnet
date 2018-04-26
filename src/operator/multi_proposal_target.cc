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
 * \author Bharat Singh
*/

#include "./multi_proposal_target-inl.h"
#include <set>
#include <math.h>
#include <unistd.h>
#include <time.h>
//============================
// Bounding Box Transform Utils
//============================
namespace mxnet {
namespace op {
namespace utils {

inline void BBoxTransformInv(float* boxes,
                             const mshadow::Tensor<cpu, 4>& deltas,
                             float* im_info) {
  int num_images = deltas.size(0);
  int anchors = deltas.size(1) / 4;
  int heights = deltas.size(2);
  int widths = deltas.size(3);
  int num_anchors = anchors * heights * widths;
  //usleep(20000000);
  #pragma omp parallel for num_threads(8)
  for (int t = 0; t < num_images * num_anchors; ++t) {
    int b = t / num_anchors;
    int index = t % num_anchors;
    int a = index / (heights*widths);
    int mat = index % (heights*widths);
    int w = mat % widths; //width index
    int h = mat / widths; //height index
    float width = boxes[5*t + 2] - boxes[5*t] + 1.0;
    float height = boxes[5*t + 3] - boxes[5*t + 1] + 1.0;
    float ctr_x = boxes[5*t + 0] + 0.5 * (width - 1.0);
    float ctr_y = boxes[5*t + 1] + 0.5 * (height - 1.0);
    float dx = deltas[b][a*4 + 0][h][w];
    float dy = deltas[b][a*4 + 1][h][w];
    float dw = deltas[b][a*4 + 2][h][w];
    float dh = deltas[b][a*4 + 3][h][w];
    float pred_ctr_x = dx * width + ctr_x;
    float pred_ctr_y = dy * height + ctr_y;
    float pred_w = exp(dw) * width;
    float pred_h = exp(dh) * height;

    float pred_x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
    float pred_y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
    float pred_x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
    float pred_y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);

    pred_x1 = std::max(std::min(pred_x1, im_info[3*b+1] - 1.0f), 0.0f);
    pred_y1 = std::max(std::min(pred_y1, im_info[3*b] - 1.0f), 0.0f);
    pred_x2 = std::max(std::min(pred_x2, im_info[3*b+1] - 1.0f), 0.0f);
    pred_y2 = std::max(std::min(pred_y2, im_info[3*b] - 1.0f), 0.0f);

    boxes[5*t] = pred_x1;
    boxes[5*t + 1] = pred_y1;
    boxes[5*t + 2] = pred_x2;
    boxes[5*t + 3] = pred_y2;
  }
}

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

// greedily keep the max detections (already sorted)
inline void NonMaximumSuppression(float* dets,
                                  int post_nms_top_n,
                                  int num_images,
                                  int num_anchors,
                                  int width,
                                  int height,
                                  float* ranges,
                                  std::vector< std::vector<int> > & final_keep_images) {
  
  int total_anchors = num_images*num_anchors*width*height;
  int chip_anchors = num_anchors*width*height;
  
  float *area = new float[total_anchors];

  #pragma omp parallel for num_threads(8)
  for (int i = 0; i < total_anchors; ++i) {
    area[i] = (dets[5*i + 2] - dets[5*i + 0] + 1) * (dets[5*i + 3] - dets[5*i + 1] + 1);
    int imid = i/chip_anchors;
    if (area[i] > ranges[2*imid + 1]*ranges[2*imid + 1] || area[i] < ranges[2*imid]*ranges[2*imid]) {
      dets[5*i + 4] = -1;
    }
  }

  int max_nms = 6000;
  #pragma omp parallel for num_threads(8)
  for (int i = 0; i < num_images; i++) {
    std::vector <float> sortids(chip_anchors);
    for (int j = 0; j < chip_anchors; j++) {
      sortids[j] = j;
    }
    int chip_index = i*chip_anchors;
    std::sort(sortids.begin(), sortids.end(), 
        [&dets,chip_index](int i1, int i2) {
          return dets[5*(chip_index + i1) + 4] > dets[5*(chip_index + i2) + 4];
        });
    float *dbuf = new float[6*max_nms];

    //reorder for spatial locality in CPU, yo!
    for (int j = 0; j < max_nms; j++) {
      int index = i*chip_anchors + sortids[j];
      dbuf[6*j] = dets[5*index];
      dbuf[6*j+1] = dets[5*index+1];
      dbuf[6*j+2] = dets[5*index+2];
      dbuf[6*j+3] = dets[5*index+3];
      dbuf[6*j+4] = dets[5*index+4];
      dbuf[6*j+5] = area[index];
    }

    int vct = 0;
    for (int j = 0; j < max_nms && vct < post_nms_top_n; j++) {
      int index = i*chip_anchors + sortids[j];
      float ix1 = dbuf[6*j];
      float iy1 = dbuf[6*j+1];
      float ix2 = dbuf[6*j+2];
      float iy2 = dbuf[6*j+3];
      float iarea = dbuf[6*j+5];

      if (dbuf[6*j+4] == -1) {
        continue;
      }

      final_keep_images[i].push_back(index);
      vct = vct + 1;
      for (int pind = j + 1; pind < max_nms; pind++) {
        if (dbuf[6*pind + 4] == -1) {
          continue;
        } 
        float xx1 = std::max(ix1, dbuf[6*pind]);
        float yy1 = std::max(iy1, dbuf[6*pind + 1]);
        float xx2 = std::min(ix2, dbuf[6*pind + 2]);
        float yy2 = std::min(iy2, dbuf[6*pind + 3]);
        float w = std::max(0.0f, xx2 - xx1 + 1.0f);
        float h = std::max(0.0f, yy2 - yy1 + 1.0f);
        float inter = w * h;
        float ovr = inter / (iarea + dbuf[6*pind+5] - inter);
        if (ovr > 0.7) {
          dbuf[6*pind + 4] = -1;
        }
      }
    }
    delete [] dbuf;
  }
  delete [] area;
}

}  // namespace utils


template<typename xpu>
class MultiProposalTargetOp : public Operator{
 public:
  explicit MultiProposalTargetOp(MultiProposalTargetParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 5);
    CHECK_EQ(out_data.size(), 4);
    //clock_t t;
    //t = clock();
    //std::cout << "quack 1" << std::endl;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<cpu, 4> scores = in_data[proposal::kClsProb].get<cpu, 4, real_t>(s);
    Tensor<cpu, 4> tbbox_deltas = in_data[proposal::kBBoxPred].get<cpu, 4, real_t>(s);
    Tensor<cpu, 2> tim_info = in_data[proposal::kImInfo].get<cpu, 2, real_t>(s);
    Tensor<cpu, 3> gt_boxes = in_data[proposal::kGTBoxes].get<cpu, 3, real_t>(s);
    Tensor<cpu, 2> tvalid_ranges = in_data[proposal::kValidRanges].get<cpu, 2, real_t>(s);

    Tensor<cpu, 2> rois = out_data[proposal::kRoIs].get<cpu, 2, real_t>(s);
    Tensor<cpu, 2> labels = out_data[proposal::kLabels].get<cpu, 2, real_t>(s);
    Tensor<cpu, 2> bbox_targets = out_data[proposal::kBboxTarget].get<cpu, 2, real_t>(s);
    Tensor<cpu, 2> bbox_weights = out_data[proposal::kBboxWeight].get<cpu, 2, real_t>(s);
    int num_images = tbbox_deltas.size(0);
    int num_anchors = tbbox_deltas.size(1) / 4;
    int height = tbbox_deltas.size(2);
    int width = tbbox_deltas.size(3);
    //number of anchors per chip
    int count_anchors = num_anchors*height*width;
    //std::cout << "quack 2" << std::endl;
    //total number of anchors in a batch
    int total_anchors = count_anchors * num_images;
    
    float *proposals = new float[total_anchors*5];
    float *im_info = new float[num_images*3];
    float *valid_ranges = new float[num_images*2];

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

    //std::cout << "quack 3" << std::endl;
    #pragma omp parallel for num_threads(8)
    for (int t = 0; t < total_anchors; ++t) {
      int b = t / count_anchors;
      int index = t % count_anchors;
      int i = index / (height*width);
      int mat = t % (height*width);
      int k = mat % width; //width index
      int j = mat / width; //height index
      proposals[5*t] = anchors[4*i] + k * param_.feature_stride;
      proposals[5*t + 1] = anchors[4*i+1] + j * param_.feature_stride;
      proposals[5*t + 2] = anchors[4*i+2] + k * param_.feature_stride;
      proposals[5*t + 3] = anchors[4*i+3] + j * param_.feature_stride;
      proposals[5*t + 4] = scores[b][1][i*height + j][k];
    }

    //std::cout << "quack 4" << std::endl;
    //copy im_info
    for (int i = 0; i < num_images; i++) {
      im_info[i*3] = tim_info[i][0];
      im_info[i*3+1] = tim_info[i][1];
      im_info[i*3+2] = tim_info[i][2];
      valid_ranges[i*2] = tvalid_ranges[i][0];
      valid_ranges[i*2+1] = tvalid_ranges[i][1];
    }

    utils::BBoxTransformInv(proposals, tbbox_deltas, im_info);

    utils::FilterBox(proposals, total_anchors, 3);

    std::vector <std::vector<int> > keep_images(num_images);
    for (int i = 0; i < num_images; i++) {
      keep_images[i] = std::vector<int>(0);
    }
    //std::cout << "quack 5" << std::endl;
    int rpn_post_nms_top_n = param_.rpn_post_nms_top_n;
    utils::NonMaximumSuppression(proposals, rpn_post_nms_top_n, num_images, num_anchors, width, height, valid_ranges, keep_images);
    //std::cout << "quack 6" << std::endl;
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < num_images; i++) {
      int numpropsi = keep_images[i].size();
      for (int j = 0; j < numpropsi; j++) {
        int base = (i*rpn_post_nms_top_n + j);
        rois[base][0] = i;
        rois[base][1] = proposals[5*keep_images[i][j] + 0];
        rois[base][2] = proposals[5*keep_images[i][j] + 1];
        rois[base][3] = proposals[5*keep_images[i][j] + 2];
        rois[base][4] = proposals[5*keep_images[i][j] + 3];
      }

      for (int j = numpropsi; j < rpn_post_nms_top_n; j++) {
        int base = (i*rpn_post_nms_top_n + j);
        rois[base][0] = i;
        rois[base][1] = 0;
        rois[base][2] = 0;
        rois[base][3] = 100;
        rois[base][4] = 100; 
      }

    }

    //std::cout << "quack 7" << std::endl;
    std::vector <int> numgts_per_image(num_images);
    std::vector <int> sumgts_per_image(num_images);

    for (int i = 0; i < num_images; i++) {
      numgts_per_image[i] = 0;
      for (int j = 0; j < 100; j++) {
        if (gt_boxes[i][j][4] != -1) {
          numgts_per_image[i]++;
        }
      }
      if (i == 0) {
        sumgts_per_image[i] = numgts_per_image[i];
      } else {
        sumgts_per_image[i] = numgts_per_image[i] + sumgts_per_image[i-1];
      }
    }

    float xx1, yy1, xx2, yy2, w, h, inter, ovr, a2;
    //std::cout << "quack 8" << std::endl;
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < num_images; i++) {
      for (int j = 0; j < rpn_post_nms_top_n; j++) {
        int basepos = rpn_post_nms_top_n*i + j;
        labels[basepos][0] = 0;
        bbox_targets[basepos][0] = 1.0;
        bbox_targets[basepos][1] = 1.0;
        bbox_targets[basepos][2] = 1.0;
        bbox_targets[basepos][3] = 1.0;

        bbox_weights[basepos][0] = 0.0;
        bbox_weights[basepos][1] = 0.0;
        bbox_weights[basepos][2] = 0.0;
        bbox_weights[basepos][3] = 0.0;
      }
      int props_this_batch = rpn_post_nms_top_n;

      for (int k = props_this_batch - numgts_per_image[i], j = 0; k < props_this_batch; j++, k++) {
          float w = gt_boxes[i][j][2] - gt_boxes[i][j][0];
          float h = gt_boxes[i][j][3] - gt_boxes[i][j][1];
          float area = w*h;
          if (area >= valid_ranges[2*i]*valid_ranges[2*i] && area <= valid_ranges[2*i+1]*valid_ranges[2*i+1]) {
            rois[i*rpn_post_nms_top_n + k][1] = gt_boxes[i][j][0];
            rois[i*rpn_post_nms_top_n + k][2] = gt_boxes[i][j][1];
            rois[i*rpn_post_nms_top_n + k][3] = gt_boxes[i][j][2];
            rois[i*rpn_post_nms_top_n + k][4] = gt_boxes[i][j][3];
          }
        }
    }

    #pragma omp parallel for num_threads(8)
    for (int imid = 0; imid < num_images; imid++) {
      int tpct = 0;
      int num_gts_this_image = numgts_per_image[imid];
      //std::cout << "gtc " << num_gts_this_image << std::endl;
      int props_this_batch = rpn_post_nms_top_n;
      if (num_gts_this_image > 0) {
      	float *overlaps = new float[props_this_batch * num_gts_this_image];
        float *max_overlaps = new float[props_this_batch];
        for (int i = 0; i < props_this_batch; i++) {
          max_overlaps[i] = 0;
        }
        float *max_overlap_ids = new float[props_this_batch];
        std::set <int> positive_label_ids;
        for (int i = 0; i < props_this_batch; i++) {
          max_overlap_ids[i] = 0;
        }

        for (int i = props_this_batch; i < rpn_post_nms_top_n; i++) {
          labels[imid*rpn_post_nms_top_n + i][0] = -1;
        }
        //get overlaps, maximum overlaps and gt labels
        for (int i = 0; i < numgts_per_image[imid]; i++) {
          float x1 = gt_boxes[imid][i][0];
          float y1 = gt_boxes[imid][i][1];
          float x2 = gt_boxes[imid][i][2];
          float y2 = gt_boxes[imid][i][3];
          float pbase;
          float a1 = (x2 - x1) * (y2 - y1);
          for (int j = 0; j < props_this_batch; j++) {
            pbase = rpn_post_nms_top_n*imid + j;
            xx1 = std::max(x1, rois[pbase][1]);
            yy1 = std::max(y1, rois[pbase][2]);
            xx2 = std::min(x2, rois[pbase][3]);
            yy2 = std::min(y2, rois[pbase][4]);
            w = std::max(0.0f, xx2 - xx1 + 1.0f);
            h = std::max(0.0f, yy2 - yy1 + 1.0f);
            a2 = (rois[pbase][3] - rois[pbase][1]) * (rois[pbase][4] - rois[pbase][2]);
            inter = w * h;
            ovr = inter / (a1 + a2 - inter);
            overlaps[i*num_gts_this_image + j] = ovr;

            if (overlaps[i*num_gts_this_image + j] > max_overlaps[j] && overlaps[i*num_gts_this_image + j] > 0.5) {
              max_overlaps[j] = overlaps[i*num_gts_this_image + j];
              max_overlap_ids[j] = i;
              //set labels for positive proposals
              labels[imid*rpn_post_nms_top_n + j][0] = gt_boxes[imid][i][4];
              positive_label_ids.insert(j);
              tpct = tpct + 1;
            }
          }
        }
        //p is for proposal and g is for gt, cx is x center and w,h is width and height
        int pid, gtid;
        float gx1, gx2, gy1, gy2, px1, px2, py1, py2;
        float gcx, gcy, gw, gh, pcx, pcy, pw, ph;
        //generate bbox targets for the positive labels
        for (auto it = positive_label_ids.begin(); it !=positive_label_ids.end(); it++) {
          pid = *it;
          int baseid = (imid*rpn_post_nms_top_n + pid);
          bbox_weights[baseid][0] = 1;
          bbox_weights[baseid][1] = 1;
          bbox_weights[baseid][2] = 1;
          bbox_weights[baseid][3] = 1;

          gtid = max_overlap_ids[pid];

          gx1 = gt_boxes[imid][gtid][0];
          gy1 = gt_boxes[imid][gtid][1];
          gx2 = gt_boxes[imid][gtid][2];
          gy2 = gt_boxes[imid][gtid][3];

          gw = gx2 - gx1 + 1;
          gh = gy2 - gy1 + 1;
          gcx = gx1 + gw*0.5;
          gcy = gy1 + gh*0.5;

          px1 = rois[baseid][1];
          py1 = rois[baseid][2];
          px2 = rois[baseid][3];
          py2 = rois[baseid][4];

          pw = px2 - px1 + 1;
          ph = py2 - py1 + 1;
          pcx = px1 + (pw-1)*0.5;
          pcy = py1 + (ph-1)*0.5;

          bbox_targets[baseid][0] = param_.bbox_scale * 5 * (gcx - pcx) / (pw + 1e-7);
          bbox_targets[baseid][1] = param_.bbox_scale * 5 * (gcy - pcy) / (ph + 1e-7);
          bbox_targets[baseid][2] = param_.bbox_scale * 10 * log(gw/(pw + 1e-7));
          bbox_targets[baseid][3] = param_.bbox_scale * 10 * log(gh/(ph + 1e-7));
        }
        //std::cout << tpct << std::endl;
        delete [] max_overlap_ids;
        delete [] overlaps;
        delete [] max_overlaps;
      }      
    }
    //std::cout << "quack end" << std::endl;
    delete [] im_info;
    delete [] valid_ranges;
    delete [] proposals;
    //t = clock() - t;
    //printf ("It took me %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);
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
    CHECK_EQ(in_grad.size(), 5);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> gscores = in_grad[proposal::kClsProb].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gbbox = in_grad[proposal::kBBoxPred].get<xpu, 4, real_t>(s);
    Tensor<xpu, 2> ginfo = in_grad[proposal::kImInfo].get<xpu, 2, real_t>(s);
    Tensor<xpu, 3> ggt_boxes = in_grad[proposal::kGTBoxes].get<xpu, 3, real_t>(s);
    Tensor<xpu, 2> gvalid_ranges = in_grad[proposal::kValidRanges].get<xpu, 2, real_t>(s);

    // can not assume the grad would be zero
    Assign(gscores, req[proposal::kClsProb], 0);
    Assign(gbbox, req[proposal::kBBoxPred], 0);
    Assign(ginfo, req[proposal::kImInfo], 0);
    Assign(ggt_boxes, req[proposal::kGTBoxes], 0);
    Assign(gvalid_ranges, req[proposal::kValidRanges], 0);
  }

 private:
  MultiProposalTargetParam param_;
};  // class MultiProposalOp

template<>
Operator *CreateOp<cpu>(MultiProposalTargetParam param) {
  return new MultiProposalTargetOp<cpu>(param);
}

Operator* MultiProposalTargetProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(MultiProposalTargetParam);

MXNET_REGISTER_OP_PROPERTY(MultiProposalTarget, MultiProposalTargetProp)
.describe("Generate region proposals via RPN")
.add_argument("cls_prob", "NDArray-or-Symbol", "Score of how likely proposal is object.")
.add_argument("bbox_pred", "NDArray-or-Symbol", "BBox Predicted deltas from anchors for proposals")
.add_argument("im_info", "NDArray-or-Symbol", "Image size and scale.")
.add_argument("gt_boxes", "NDArray-or-Symbol", "Image size and scale.")
.add_argument("valid_ranges", "NDArray-or-Symbol", "Image size and scale.")
.add_arguments(MultiProposalTargetParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
