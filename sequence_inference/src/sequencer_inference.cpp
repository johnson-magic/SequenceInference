
#include <cmath>
#include "sequencer_inference.h"

void SequenceInferencer::Init(std::string &model_path, std::string &charset_path)
{
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

void SequenceInferencer::GetInputInfo(){
	numInputNodes_ = GetSessionInputCount();
    for (int input_id = 0; input_id < numInputNodes_; input_id++) {
        Ort::AllocatorWithDefaultOptions allocator;  // 如何理解allocator的工作机制？ 它能够被复用吗？
        auto input_name = session_->GetInputNameAllocated(input_id, allocator);  // 通常，numInputNodes_只为1
        input_node_names_.push_back(input_name.get());  // char* -> string
	
        Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(input_id);
	
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        net_h_.push_back(input_dims[2]);
        net_w_.push_back(input_dims[3]);    
	}
}

size_t SequenceInferencer::GetSessionInputCount(){
    return session_->GetInputCount();
}

size_t SequenceInferencer::GetSessionOutputCount(){
    return session_->GetOutputCount();
}

void SequenceInferencer::GetOutputInfo(){
    // get the output information
	// 1. numOutputNodes_
	// 2. Output_node_names_
	// 3. Output_w_ and net_h_

	numOutputNodes_ = GetSessionOutputCount();
    
    for(int output_id = 0; output_id < numOutputNodes_; output_id++){
        Ort::AllocatorWithDefaultOptions allocator;
        auto out_name = session_->GetOutputNameAllocated(output_id, allocator);
        output_node_names_.push_back(out_name.get());

        Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(output_id);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        
        // output_h_.push_back(output_dims[1]);  // 注意：对于检测模型，模型的输出一般为(B, 8, anchor_point)
        // output_w_.push_back(output_dims[2]);
        class_num_.push_back(output_dims[1]);   
    } 
}

void SequenceInferencer::PreProcess(const std::string& image_path){
    //1. read
    //2. BGR2GRAY
    //3. RescaleToHeight
    //4. normalize
    image_path_ = image_path;
    std::ifstream test_open(image_path_);
    if(test_open.is_open()){
        image_ = cv::imread(image_path_);
        test_open.close();
        if (image_.empty()) {
	    	std::cerr << "Failed to read the image!" << std::endl;
            return;
        }
    }else{
        std::cerr << "Failed to read the image!" << std::endl;
        return;
    }
    cv::cvtColor(image_, image_, cv::COLOR_RGB2GRAY);  
    
    // RescaleToHeight
    int w_img = image_.cols;
    int h_img = image_.rows;
    
    int h_resized = 32;
    int w_resized = ceil(h_resized * 1.0 / h_img * w_img);
    if(w_resized % 16 != 0){
        w_resized = ceil(w_resized / 16) * 16;
    }
    cv::resize(image_, image_, cv::Size(w_resized, h_resized), 0, 0, cv::INTER_LINEAR); 
    image_.convertTo(image_, CV_32F);  // 在进行归一化之前，这个操作是必须的
    image_ = (image_ - 127.0) / 127.0;
    net_h_[0] = h_resized;
    net_w_[0] = w_resized;
}


void SequenceInferencer::PreProcess(cv::Mat image){
    //1. read
    //2. BGR2GRAY
    //3. RescaleToHeight
    //4. normalize
	image_ = image;
	if (image_.empty()) {
		std::cerr << "Failed to read the image!" << std::endl;
		return;
	}

    cv::cvtColor(image_, image_, cv::COLOR_RGB2GRAY);  
    
    // RescaleToHeight
    int w_img = image_.cols;
    int h_img = image_.rows;
    
    int h_resized = 32;
    int w_resized = ceil(h_resized * 1.0 / h_img * w_img);
    if(w_resized % 16 != 0){
        w_resized = ceil(w_resized / 16) * 16;
    }
    cv::resize(image_, image_, cv::Size(w_resized, h_resized), 0, 0, cv::INTER_LINEAR); 
    image_.convertTo(image_, CV_32F);  // 在进行归一化之前，这个操作是必须的
    image_ = (image_ - 127.0) / 127.0;
    
    net_h_[0] = h_resized;
    net_w_[0] = w_resized;
}

void SequenceInferencer::SaveOrtValueAsImage(Ort::Value& value, const std::string& filename) {
    // 确保值是张量
    if (!value.IsTensor()) {
        std::cerr << "Value is not a tensor." << std::endl;
        return;
    }

    Ort::TensorTypeAndShapeInfo info = value.GetTensorTypeAndShapeInfo();
    
    // 获取张量的维度
    std::vector<int64_t> shape = info.GetShape();
    int height = static_cast<int>(shape[2]);
    int width = static_cast<int>(shape[3]);
    
    // 检查是否为 RGB 图像，形状应为 {1, 3, height, width}
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3) {
        std::cerr << "Expected a 4D tensor with shape {1, 3, height, width}." << std::endl;
        return;
    }

    // 获取张量数据
    float* data = value.GetTensorMutableData<float>();

    // 将数据转为 OpenCV 的 cv::Mat 格式，注意通道顺序
    cv::Mat image(height, width, CV_32FC3, data);
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR); // 转换为 BGR 格式以便保存

    // 将数据类型转换为可保存的格式（如 8 位无符号整数）
    cv::Mat imageToSave;
    image.convertTo(imageToSave, CV_8UC3, 255.0); // 假设输入是范围在 [0, 1] 之间的浮点数

    // 保存图像
    if (!cv::imwrite(filename, imageToSave)) {
        std::cerr << "Failed to save image to " << filename << std::endl;
	}
}

void SequenceInferencer::Inference(){
	const std::array<const char*, 1> inputNames = { input_node_names_[0].c_str() };  // std::array用于fixed size array
	const std::array<const char*, 1> outNames = { output_node_names_[0].c_str() };
   
	// cv::Mat blob;
	// cv::dnn::blobFromImage(image_, blob, 1 / 255.0, cv::Size(net_w_[0], net_h_[0]), cv::Scalar(0, 0, 0), true, false);  // swapRB = true, crop = false,
    
	size_t tpixels = net_h_[0] * net_w_[0] * 3 * 1;
	std::array<int64_t, 4> input_shape_info{ 1, 1, net_h_[0], net_w_[0]};

    
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, image_.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
    assert(input_tensor.IsTensor());
	try {
		#ifdef CONFORMANCE_TEST
			SaveOrtValueToTextFile(input_tensor, "onnx_input.txt");
		#endif
		ort_outputs_ = session_->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor, inputNames.size(), outNames.data(), outNames.size());
		#ifdef CONFORMANCE_TEST
			for(int i=0; i<ort_outputs_.size(); i++){
				SaveOrtValueToTextFile(input_tensor, "onnx_output_" + std::to_string(i) + ".txt");
			}
		#endif
    }
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	}
}

void SequenceInferencer::PostProcess(){
	// output data
	float* pdata = ort_outputs_[0].GetTensorMutableData<float>();  // pdata的尺寸还需要从长计议 1, (w_input_[0] + 4)/4, 37;
    int L = (net_w_[0] + 4)/4;
    int C = charset_len_with_blank_;

    std::vector<int> predictions;
    decoded_indices_.clear();
    decoded_chars_.clear();
    for (int i = 0; i < L; ++i) {
        float* timestep_probs = pdata + i*C; // 第 i 个时间步的概率
        int max_idx = 0;
        float max_prob = timestep_probs[0];
        for (int j = 1; j < C; ++j) {
            if (timestep_probs[j] > max_prob) {
                max_prob = timestep_probs[j];
                max_idx = j;
            }
        }
        predictions.push_back(max_idx);
    }

    // 步骤2：合并重复字符并跳过空白符
    int prev_idx = -1;
    for (int idx : predictions) {
        if (idx == charset_len_) { // 36表示空白符
            prev_idx =-1; // 重置 prev_idx, 确认
            continue; // 跳过空白符
        }
        if (idx != prev_idx) { // 仅当字符变化时保留
            decoded_indices_.push_back(idx);
            decoded_chars_.push_back(charset_[idx]);
            prev_idx = idx;
        }
    }
            
	#ifdef CONFORMANCE_TEST
		SaveRotatedObjsToTextFile(remain_rotated_objects_, "remain_rotated_objects.txt");
	#endif

}
std::pair<std::vector<int>, std::vector<char>> SequenceInferencer::GetRes(){
    return std::make_pair(decoded_indices_, decoded_chars_);
}

void SequenceInferencer::Release(){
	session_->release();
}


