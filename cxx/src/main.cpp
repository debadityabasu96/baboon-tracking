#include <chrono>
#include <cstdlib>
#include <string>
#include <tuple>
#include <iostream>
#include <string.h>
#include <fstream>
#define ENABLE_BENCHMARKING
#include <fmt/format.h>

#define DEBUG_DRAW
#include <opencv2/core.hpp>
#ifdef DEBUG_DRAW
#include <opencv2/highgui.hpp>
#endif
#ifdef USE_CUDA
#include <opencv2/cudacodec.hpp>
#else
#include <opencv2/imgcodecs.hpp>
#endif
#include <opencv2/videoio.hpp>

#include "constant_velocity_kalman_filter.h"
#include "pipes.h"
// so the compiler doesnt complain
#define UNUSED(x) (void)(x)

#ifdef DEBUG_DRAW
void show(std::string window_name, cv::InputArray image) {
    static std::map<std::string, bool> has_shown;
    if (!has_shown[window_name])
      cv::namedWindow(window_name, cv::WINDOW_KEEPRATIO);
    cv::imshow(window_name, image);
    if (!has_shown[window_name]) {
      // Frame is too big to display on my screen
      cv::resizeWindow(window_name, image.cols() / 4.0, image.rows() / 4.0);
      has_shown[window_name] = true;
    }
}
#endif

template <typename frame> class pipeline {
private:
  using bt = baboon_tracking::pipes<frame>;

public:
  pipeline(std::shared_ptr<baboon_tracking::historical_frames_container<frame>>
               hist_frames, std::shared_ptr<baboon_tracking::keypoint_descriptor_container<frame>> kp_desc_container)
      : convert_bgr_to_gray{}, blur_gray{3}, compute_homography{0.27, 4.96,
                                                                0.13, 10001,
                                                                hist_frames,
                                                                kp_desc_container},
        transform_history_frames_and_masks{hist_frames},
        rescale_transformed_history_frames{48}, generate_weights{},
        generate_history_of_dissimilarity{}, intersect_frames{},
        union_intersected_frames{}, subtract_background{hist_frames},
        compute_moving_foreground{hist_frames}, apply_masks{},
        erode_dialate{5, 10}, detect_blobs{} {}

  auto process(std::uint64_t current_frame_num, frame &&bgr_frame) {
    auto start = std::chrono::steady_clock::now(); 
    auto drawing_frame = bgr_frame.clone();
    auto end = std::chrono::steady_clock::now(); 
    #ifdef ENABLE_BENCHMARKING
    #ifdef USE_CUDA
    fmt::print("Took {} us for drawing_frame\n",std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    #else
    fmt::print("Took {} ms for drawing_frame\n",std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    #endif
    #endif

    start = std::chrono::steady_clock::now(); 
    auto gray_frame = convert_bgr_to_gray.run(std::move(bgr_frame));
    end = std::chrono::steady_clock::now(); 
    
    #ifdef ENABLE_BENCHMARKING
    #ifdef USE_CUDA
    fmt::print("Took {} us for gray_frame\n",std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    #else
    fmt::print("Took {} ms for gray_frame\n",std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    #endif
    #endif


    start = std::chrono::steady_clock::now(); 
    auto blurred_frame = blur_gray.run(std::move(gray_frame));
    end = std::chrono::steady_clock::now(); 
    #ifdef ENABLE_BENCHMARKING
    #ifdef USE_CUDA
    fmt::print("Took {} us for blurred_frame\n",std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    #else
    fmt::print("Took {} ms for blurred_frame\n",std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    #endif
    #endif
    
    start = std::chrono::steady_clock::now(); 
    auto homographies =
        compute_homography.run(current_frame_num, std::move(blurred_frame));
    end = std::chrono::steady_clock::now(); 
    #ifdef ENABLE_BENCHMARKING
    #ifdef USE_CUDA
    fmt::print("Took {} us for compute_homography\n",std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    #else
    fmt::print("Took {} ms for compute_homography\n",std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    #endif
    #endif
  
    if (homographies.empty())
      return std::vector<cv::Rect>{};

  
    start = std::chrono::steady_clock::now(); 
    auto [transformed_history_frames, transformed_masks] =
        transform_history_frames_and_masks.run(current_frame_num,
                                               std::move(homographies));
    end = std::chrono::steady_clock::now(); 
    #ifdef ENABLE_BENCHMARKING
    #ifdef USE_CUDA
    fmt::print("Took {} us for history frames and masks\n",std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    #else
    fmt::print("Took {} ms for history frames and masks\n",std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    #endif
    #endif

    start = std::chrono::steady_clock::now(); 
    auto transformed_rescaled_history_frames =
        rescale_transformed_history_frames.run(transformed_history_frames);
    end = std::chrono::steady_clock::now(); 
    #ifdef ENABLE_BENCHMARKING
    #ifdef USE_CUDA
    fmt::print("Took {} us for rescale_transformed_history_frames\n",std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    #else
    fmt::print("Took {} ms for rescale_transformed_history_frames\n",std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    #endif
    #endif 
    
    // Note: seems to fail in weight generation... transformed_rescaled_frames
    // are ok, weights are not
    
    start = std::chrono::steady_clock::now(); 
    auto weights = generate_weights.run(transformed_rescaled_history_frames);
    end = std::chrono::steady_clock::now(); 
    #ifdef ENABLE_BENCHMARKING
    #ifdef USE_CUDA
    fmt::print("Took {} us for generating weights\n",std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    #else
    fmt::print("Took {} ms for generating weights\n",std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    #endif
    #endif
    start = std::chrono::steady_clock::now(); 
    auto history_of_dissimilarity = generate_history_of_dissimilarity.run(
        transformed_history_frames, transformed_rescaled_history_frames);
    end = std::chrono::steady_clock::now(); 
    #ifdef ENABLE_BENCHMARKING
    #ifdef USE_CUDA
    fmt::print("Took {} us for history of dissimilarity\n",std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    #else
    fmt::print("Took {} ms for history of dissimilarity\n",std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    #endif
	#endif

    start = std::chrono::steady_clock::now(); 
    auto intersected_frames =
        intersect_frames.run(std::move(transformed_history_frames),
                             std::move(transformed_rescaled_history_frames));
    end = std::chrono::steady_clock::now(); 
    #ifdef ENABLE_BENCHMARKING
    #ifdef USE_CUDA
    fmt::print("Took {} us for intersected_frames\n",std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    #else
    fmt::print("Took {} ms for intersected_frames\n",std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    #endif
    #endif
    
    start = std::chrono::steady_clock::now(); 
    auto union_of_all =
        union_intersected_frames.run(std::move(intersected_frames));
    end = std::chrono::steady_clock::now(); 
    #ifdef ENABLE_BENCHMARKING
    #ifdef USE_CUDA
    fmt::print("Took {} us for union_intersected_frames\n",std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    #else
    fmt::print("Took {} ms for union intersected_frames\n",std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    #endif
    #endif
    
    start = std::chrono::steady_clock::now(); 
    auto foreground = subtract_background.run(current_frame_num,
                                              std::move(union_of_all), weights);
    end = std::chrono::steady_clock::now(); 
    #ifdef ENABLE_BENCHMARKING
    #ifdef USE_CUDA
    fmt::print("Took {} us for subtract background\n",std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    #else
    fmt::print("Took {} ms for subtract background\n",std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    #endif
    #endif
    auto start_mf = std::chrono::steady_clock::now();

    auto moving_foreground = compute_moving_foreground.run(
        std::move(history_of_dissimilarity), std::move(foreground),
        std::move(weights));

    auto end_mf = std::chrono::steady_clock::now();
    fmt::print(
        "Moving foreground took {} ms\n",
        std::chrono::duration_cast<std::chrono::milliseconds>(end_mf - start_mf)
            .count());

    
    start = std::chrono::steady_clock::now(); 
    apply_masks.run(&moving_foreground, std::move(transformed_masks));
    end = std::chrono::steady_clock::now(); 
    #ifdef ENABLE_BENCHMARKING
    #ifdef USE_CUDA
    fmt::print("Took {} us for Applying masks\n",std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    #else
    fmt::print("Took {} ms for Applying masks\n",std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    #endif
    #endif
    
    start = std::chrono::steady_clock::now(); 
    erode_dialate.run(&moving_foreground);
    end = std::chrono::steady_clock::now(); 
    #ifdef ENABLE_BENCHMARKING
    #ifdef USE_CUDA
    fmt::print("Took {} us for Erode Dialtes\n",std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    #else
    fmt::print("Took {} ms for Erode Dialtes\n",std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    #endif
    #endif

    start = std::chrono::steady_clock::now(); 
    auto blobs = detect_blobs.run(std::move(moving_foreground));
    end = std::chrono::steady_clock::now(); 
    #ifdef ENABLE_BENCHMARKING
    #ifdef USE_CUDA
    fmt::print("Took {} us for Detecting Blobs\n",std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    #else
    fmt::print("Took {} ms for Detecting Blobs\n",std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    #endif
    #endif
    fmt::print("Frame {} done\n", current_frame_num);

    //auto bounding_box_color = cv::Scalar(255, 0, 0);
    //for (auto &&bounding_box : blobs) {
    //   cv::rectangle(drawing_frame, bounding_box, bounding_box_color, 4);
    //}

    //show("process", drawing_frame);
    //cv::waitKey(100);

    return blobs;
  }

private:
  typename bt::convert_bgr_to_gray convert_bgr_to_gray;
  typename bt::blur_gray blur_gray;

  typename bt::compute_homography compute_homography;
  typename bt::transform_history_frames_and_masks
      transform_history_frames_and_masks;
  typename bt::rescale_transformed_history_frames
      rescale_transformed_history_frames;
  typename bt::generate_weights generate_weights;
  typename bt::generate_history_of_dissimilarity
      generate_history_of_dissimilarity;
  typename bt::intersect_frames intersect_frames;
  typename bt::union_intersected_frames union_intersected_frames;
  typename bt::subtract_background subtract_background;
  typename bt::compute_moving_foreground compute_moving_foreground;
  typename bt::apply_masks apply_masks;
  typename bt::erode_dilate erode_dialate;
  typename bt::detect_blobs detect_blobs;
};

int main(int argc, char *argv[]) {
  constexpr unsigned int max_threads = 8; // TODO: put the thread pool back optionally
  char input_file_name[256] = "./input.mp4";

  if (argc == 2) {
    std::cout << "Using specified file name." << std::endl;
    strcpy(input_file_name, argv[1]);
  }

  std::ofstream myfile;
  myfile.open("../bounding_boxes.txt");
  myfile << "frame" << " " << "tl.x" << " " << "tl.y" << " "
         << "br.x" << " " << "br.y" << std::endl;

  // TODO: we're using USE_CUDA elsewhere to mean that CUDA headers are
  // available (an unfortunate kludge that happens because some builds of OpenCV
  // don't even come with the OpenCV CUDA headers even though those headers have
  // appropriate stubbed functions). This is not with the spirit of the meaning
  // of USE_CUDA (i.e. USE_CUDA should probably only be touched here, while
  // HAS_CUDA should maybe be used elsewhere?)
  cv::Mat frame_host;
#ifdef USE_CUDA
  cv::cuda::GpuMat frame;
#else
  cv::Mat frame;
#endif

  cv::VideoCapture vc{};
  if (!vc.open(input_file_name, cv::CAP_FFMPEG)) {
    throw std::runtime_error{fmt::format("Couldn't open {}", input_file_name)};
  }
  const auto fps = vc.get(cv::CAP_PROP_FPS);

  auto hist_frames = std::make_shared<
      baboon_tracking::historical_frames_container<decltype(frame)>>(
      9, max_threads);
  auto kp_desc_container = std::make_shared<
      baboon_tracking::keypoint_descriptor_container<decltype(frame)>>(9);
  pipeline pl{hist_frames, kp_desc_container};


  baboon_tracking::constant_velocity_kalman_filter<20> kf{
      {9, 9, 2, 2}, // Units are pixels, pixels, pixels/s, pixels/s respectively
      {2, 2, 5, 5}, // Units are all pixels
      30,           // Units are pixels
      1.0 / fps // s/frame
  };
  static constexpr int actual_num_baboons = 1;
  kf.set_x_hat(0, 2920);
  kf.set_x_hat(1, 1210);

  auto total_time_consumed = 0; //let's count frames in [9,25]

  
  std::ofstream file_handler;
  file_handler.open("output/baboons.csv");
  file_handler <<"x1,"<<"y1,"<<"x2,"<<"y2,"<<"frame"<<"\n";
  for (std::uint64_t i = 0; vc.read(frame_host) && !frame_host.empty(); i++) {
#ifdef USE_CUDA
    frame.upload(frame_host);
#else
    frame = frame_host;
#endif

#ifdef DEBUG_DRAW
    auto drawing_frame = cv::Mat{frame}.clone();
#endif

    auto start = std::chrono::steady_clock::now();
    auto bounding_boxes = pl.process(i, std::move(frame));
    auto end = std::chrono::steady_clock::now();
    auto time_taken = start -end;    
    // draw_regions(boudning_boxes, drawing_frame);


    frame = decltype(frame){};
     
    #ifdef ENABLE_BENCHMARKING
    fmt::print("Main Took {} ms \n",std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    #endif
    /*
    auto start_save_baboons = std::chrono::steady_clock::now();
    int count_of_bounding_boxes = 0;
    count_of_bounding_boxes = bounding_boxes.size();

    for(int j=0;j< count_of_bounding_boxes;j++)
    {
    auto rect = bounding_boxes[j];
    if((rect.size().width >=15) && (rect.size().height > = 15))
	    file_handler <<rect.tl().x <<"," <<rect.tl().y <<","<<rect.tl().x + rect.size().width<<","<<rect.tl().y + rect.size().height<<","<<i<<std::endl;
    }
    auto end_save_baboons = std::chrono::steady_clock::now();
    auto time_save_baboons = end_save_baboons - start_save_baboons;

    #ifdef ENABLE_BENCHMARKING
    fmt::print("Took {} ms for save_baboons \n",std::chrono::duration_cast<std::chrono::milliseconds>(end_save_baboons-start_save_baboons).count());
    #endif
    */
    if (!bounding_boxes.empty()) {
      auto x_hat = kf.run(actual_num_baboons, bounding_boxes);
      // macro so the compiler doesnt complain
      UNUSED(x_hat);
#ifdef DEBUG_DRAW
      static const auto bounding_box_color = cv::Scalar(255, 0, 0);
      //for (auto &&bounding_box : bounding_boxes) {
      //  cv::rectangle(drawing_frame, bounding_box, bounding_box_color, 4);
      //}

    int num_bboxes = bounding_boxes.size();
    std::cout << "all bbox: " << num_bboxes << std::endl;

    cv::putText(drawing_frame, "Frame: " + std::to_string(i), 
                    cv::Point(100, 100),
                    cv::FONT_HERSHEY_SIMPLEX,
                    4,
                    cv::Scalar(255, 0, 0),
                    8,
                    false
                    );

    auto start = std::chrono::steady_clock::now();

    int count = 0;
    if (num_bboxes > 0) {

        for (int j = 0; j < num_bboxes; j++){

          auto rect = bounding_boxes[j];
          //std::cout << rect.tl().x << " " << rect.tl().y << " " << rect.size().height 
          //           << " " << rect.size().width << std::endl;

        
          if ( (rect.size().width >= 14) && (rect.size().height >= 14)) {

              count++;
              //myfile << i << " "; //current_frame_num
              myfile << rect.tl().x << "," << rect.tl().y << "," << rect.tl().x + rect.size().width 
                     << "," << rect.tl().y + rect.size().height << ","<<i<< std::endl;

              cv::rectangle(drawing_frame, rect, bounding_box_color, 4);
              cv::putText(drawing_frame, 
                    std::to_string(j), 
                    cv::Point(rect.tl().x, rect.tl().y - 10),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1,
                    cv::Scalar(255, 0, 0),
                    8,
                    false
                    );
          }
       }
    }

    //cv::waitKey(10);

    auto end = std::chrono::steady_clock::now();

    if (i >= 9 && i <=25) {
      fmt::print(
        "Drawing regions took {} us\n",
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count());
      total_time_consumed += std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    }

    if (i > 25) {
        fmt::print("Total for drawing regions: {} us\n", total_time_consumed);

    }


    //std::cout << "large bbox count: " << count << std::endl;


    /*for (int j = 0; j < num_bboxes; j++){

      auto rect = bounding_boxes[j];
      cv::rectangle(drawing_frame, rect, bounding_box_color, 4);
      cv::putText(drawing_frame, 
                  std::to_string(j), 
                  cv::Point(rect.tl().x, rect.tl().y - 10),
                  cv::FONT_HERSHEY_SIMPLEX,
                  1,
                  cv::Scalar(255, 0, 0),
                  8,
                  false
                  );
      }
  */



      //for (int j = 0; j < kf.states_per_baboon * actual_num_baboons;
      //     j += kf.states_per_baboon) {
        //cv::circle(drawing_frame,
        //           {static_cast<int>(std::round(x_hat[j + 0])),
        //            static_cast<int>(std::round(x_hat[j + 1]))},
        //           3, {0, 255, 0}, 5);
        //fmt::print("kf estimate at ({}, {}) with velocity of ({}, {})\n",
        //           x_hat[j + 0], x_hat[j + 1], x_hat[j + 2], x_hat[j + 3]);
      //}
     

     #ifdef USE_CUDA
     std::cout << "Skipping GUI Display" << std::endl;
     #else
     show("Blobs on frame", drawing_frame);
     #endif

     cv::waitKey(100);
     if((cv::waitKey(25) & 0xFF) == 'q')
	     break;
#endif
    }
  


  int total_frames = vc.get(cv::CAP_PROP_FRAME_COUNT);
  float progress = (float(i)/float(total_frames))*100;

  if(i%100 == 0)
  {
	  std::cout<<"[";
	  for(int index=0;index < 100 ; index++)
	  {
		  if(index < progress)
			  std::cout << "=";
		  else if (index == int(progress))
			  std::cout <<".";
		  else
			  std::cout <<" ";
	}
	  std::cout <<"]"<<int(progress) <<"%"<<std::endl;
  }
  }
  fmt::print("finished\n");

  myfile.close();

  return EXIT_SUCCESS;
}
