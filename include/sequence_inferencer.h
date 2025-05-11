#pragma once
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <math.h>
#include <thread>
#include <chrono>
#include <windows.h>

#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "config.h"
#include "utils.h"
#include "data_structure.h"



class SequenceInferencer {
    public:
        SequenceInferencer(std::string& model_path, std::string& charset_path){   
            model_path_ = model_path;
            charset_path_ = charset_path;
            Init(model_path_, charset_path_);
        };

        void GetInputInfo();
        void GetOutputInfo();


        void PreProcess(std::string& image_path);
        void PreProcess(cv::Mat& image);
        void Inference();
        void PostProcess();
        std::pair<std::vector<int>, std::vector<char>> GetRes();

        void Release();
        

    private:
        Ort::SessionOptions options_;
        Ort::Session* session_;
        Ort::Env env_{nullptr};

        std::string image_path_;
        std::string model_path_;
        std::string charset_path_;
	    std::vector<char> charset_;  // charset of sequence model
        int charset_len_;
        int charset_len_with_blank_;  // "--hh-e-l-ll-oo--", "-" represents blank
        
	    cv::Mat image_;
        // Ort::Value input_tensor_;
        std::vector<Ort::Value> ort_outputs_;
        
        size_t numInputNodes_;  // usually, it is 1
        size_t numOutputNodes_;
        std::vector<std::string> input_node_names_;
	    std::vector<std::string> output_node_names_;
        std::vector<int> input_w_;  // net input (width)  # 事实上，通常仅仅只有1个输入node和1个输出node, 这里仅仅是为了接口通用
        std::vector<int> input_h_;  // net input (height)
        std::vector<int> output_class_num_;  // net output(class_num_)

        float x_factor_;
        float y_factor_;
        float scale_;
        int top_;  // border
        int bottom_;
        int left_;
        int right_;

        std::vector<int> predictions_;
        std::vector<float> scores_;
        std::vector<int> decoded_indices_;
        std::vector<char> decoded_chars_;

        void Init(std::string &model_path, std::string &charset_path){
            static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "default");  //holds the logging state 
            
            Ort::SessionOptions option;
            option.SetIntraOpNumThreads(1);
            option.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
            session_ = new Ort::Session(env, ConvertToWString(model_path).c_str(), option);

            std::ifstream file(charset_path);
            if (file.is_open()) {
                std::string line;
                while (std::getline(file, line)) {
                    charset_.push_back(line[0]);  // string ->[0]-> char
                }
                file.close();
            }
            charset_len_ = charset_.size();
            charset_len_with_blank_ = charset_len_ + 1;


        }

        //Ort::Env
        static Ort::Env CreateEnv(){
           
            return Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov11-onnx");
            
        }

        //Ort::SessionOptions
        static Ort::SessionOptions CreateSessionOptions(){
            Ort::SessionOptions options;
            options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
            
            return options;
        }

        //convert std::string to std::basic_string<ORTCHAR_T>
        static std::basic_string<ORTCHAR_T> ConvertToWString(std::string& model_path){
            
            return std::basic_string<ORTCHAR_T>(model_path.begin(), model_path.end());
        }

        static std::string return_image_path(std::string image_path){
            return image_path;
             
        }

        // TO DO, input and output count fix 1
        size_t GetSessionInputCount();
        size_t GetSessionOutputCount();

        cv::Mat pad_and_resize(cv::Mat image);    
        void SaveOrtValueAsImage(Ort::Value& value, const std::string& filename);

};
