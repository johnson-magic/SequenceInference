
#include <cmath>
#include "sequence_inferencer.h"

void SequenceInferencer::GetInputInfo(){
    // get the input information
	// 1. numInputNodes_
	// 2. input_node_names_
	// 3. input_w_ and input_h_

	numInputNodes_ = GetSessionInputCount();
    
    for (int input_id = 0; input_id < numInputNodes_; input_id++) {
        Ort::AllocatorWithDefaultOptions allocator;  // 如何理解allocator的工作机制？ 它能够被复用吗？
        auto input_name = session_->GetInputNameAllocated(input_id, allocator);  // 通常，numInputNodes_只为1
        input_node_names_.push_back(input_name.get());  // char* -> string
	
        Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(input_id);
	
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_h_.push_back(input_dims[2]);
        input_w_.push_back(input_dims[3]);    
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
	// 3. Output_w_ and input_h_

	numOutputNodes_ = GetSessionOutputCount();
    
    for(int output_id = 0; output_id < numInputNodes_; output_id++){
        Ort::AllocatorWithDefaultOptions allocator;
        auto out_name = session_->GetOutputNameAllocated(output_id, allocator);
        output_node_names_.push_back(out_name.get());

        Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(output_id);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        
        // output_h_.push_back(output_dims[1]);  // 注意：对于检测模型，模型的输出一般为(B, 8, anchor_point)
        // output_w_.push_back(output_dims[2]);
        output_class_num_.push_back(output_dims[1]);   
    } 
}

void SequenceInferencer::PreProcess(){
    //1. read
    //2. BGR2GRAY
    //3. RescaleToHeight
    //4. normalize
	image_ = cv::imread(image_path_);
	if (image_.empty()) {
		std::cerr << "Failed to read the image!" << std::endl;
		return;
	}
    std::cout << "Type: " << image_.type() << std::endl;

	// ge_ = pad_and_resize(image_);
    cv::cvtColor(image_, image_, cv::COLOR_RGB2GRAY);  
    
    // RescaleToHeight
    int w_img = image_.cols;
    int h_img = image_.rows;
    
    int h_resized = 32;
    std::cout<<"000000000000000000000000000000000000000000000000"<<" "<<h_img<<" "<<w_img<<std::endl;
    int w_resized = ceil(h_resized * 1.0 / h_img * w_img);
    if(w_resized % 16 != 0){
        w_resized = ceil(w_resized / 16) * 16;
    }
    std::cout<<"11111111111111111111111111111111111111111111111111"<<" "<<w_resized<<" "<<h_resized<<std::endl;
    cv::resize(image_, image_, cv::Size(w_resized, h_resized), 0, 0, cv::INTER_LINEAR); 
    std::cout << "Type: " << image_.type() << std::endl;
    for (int r = 0; r < image_.rows; r++) {
        for (int c = 0; c < image_.cols; c++) {
            // 获取像素的 BGR 值
            // cv::Vec3b pixel = image_.at<cv::Vec3b>(r, c);
            
            // // 打印 B, G, R 通道的值（0-255）
            // std::cout << "Pixel(" << r << ", " << c << "): "
            //           << "B=" << (int)pixel[0] << ", "
            //           << "G=" << (int)pixel[1] << ", "
            //           << "R=" << (int)pixel[2] << std::endl;
            std::cout<<(int)image_.at<uchar>(r, c)<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl<<std::endl;
    image_.convertTo(image_, CV_32F);  // 在进行归一化之前，这个操作是必须的
    image_ = (image_ - 127.0) / 127.0;
    // for(int i=0; i<32; i++){
    //     for(int j=0; j<w_resized; j++){
    //         std::cout<<(float)image_.at<float>(i, j)<<" ";
    //     }
    //     std::cout<<std::endl;
    // }
    // std::cout<<std::endl;
    
    input_h_[0] = h_resized;
    input_w_[0] = w_resized;
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
    // } else {
    //     std::cout << "Image saved to " << filename << std::endl;
    // }
}

void SequenceInferencer::Inference(){
	const std::array<const char*, 1> inputNames = { input_node_names_[0].c_str() };  // std::array用于fixed size array
	const std::array<const char*, 1> outNames = { output_node_names_[0].c_str() };
   
	// cv::Mat blob;
	// cv::dnn::blobFromImage(image_, blob, 1 / 255.0, cv::Size(input_w_[0], input_h_[0]), cv::Scalar(0, 0, 0), true, false);  // swapRB = true, crop = false,
    
	size_t tpixels = input_h_[0] * input_w_[0] * 3 * 1;
	std::array<int64_t, 4> input_shape_info{ 1, 1, input_h_[0], input_w_[0]};

    
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
	float* pdata = ort_outputs_[0].GetTensorMutableData<float>();  // pdata的尺寸还需要从长计议 1, (w_input_[0] + 4)/4, 37
	// int prediction = 0;
	// float max_prob = *(pdata + prediction);
	// int begin_id = 1;
	// for(int i = begin_id; i<output_class_num_[0]; i++){
	// if(*(pdata + i) > max_prob){
	// max_prob = *(pdata + i);
	// prediction = i;
	// }
	// }
	// predictions_.push_back(prediction);
	// scores_.push_back(max_prob);
    for(int i=0; i< (input_w_[0] + 4)/4; i++){
        for(int j=0; j<37; j++){
            std::cout<<pdata[i*37 + j] <<" ";
        }
        std::cout<<std::endl;
    }
    int L = (input_w_[0] + 4)/4;
    int C = 37;

    std::vector<int> predictions;
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
    //std::vector<int> decoded_indices;
    int prev_idx = -1;
    for (int idx : predictions) {
        if (idx == 36) { // 36表示空白符
            prev_idx =-1; // 重置 prev_idx, 确认
            continue; // 跳过空白符
        }
        if (idx != prev_idx) { // 仅当字符变化时保留
            decoded_indices_.push_back(idx);
            prev_idx = idx;
        }
    }
    for(int i=0; i<decoded_indices_.size(); i++){
        std::cout<<decoded_indices_[i]<<" ";
    }
    std::cout<<std::endl;
            
	#ifdef CONFORMANCE_TEST
		SaveRotatedObjsToTextFile(remain_rotated_objects_, "remain_rotated_objects.txt");
	#endif

}

cv::Mat SequenceInferencer::pad_and_resize(const cv::Mat img){
	/*如果原始图片的宽和高有任意1个超过网络的输入宽、高，则先按照长宽比进行padding，再resize；
      否则，直接padding到网络的输入宽、高*/
    
    int w_img = img.cols;
    int h_img = img.rows;
    
    
    // 计算缩放比例
    scale_ = std::min(static_cast<double>(input_h_[0]) / h_img, static_cast<double>(input_w_[0]) / w_img);
    scale_ = std::min(static_cast<double>(scale_), 1.0);  //如果原始图片的宽和高都不超过网络的宽和高，则将scale设置为1，表示事实上不进行resize.
    // 计算新的宽度和高度
    int w_resized = static_cast<int>(std::round(scale_ * w_img));
    int h_resized = static_cast<int>(std::round(scale_ * h_img));
    // 缩放图像
    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(w_resized, h_resized), 0, 0, cv::INTER_LINEAR);

    // 计算边框宽度和高度
    int dw = input_w_[0] - w_resized;
    int dh = input_h_[0] - h_resized;

    // 分配边框宽度（上下左右）
    top_ = dh / 2;
    bottom_ = dh - top_;
    left_ = dw / 2;
    right_ = dw - left_;

    // 添加边框
    cv::Mat img_bordered;
    cv::copyMakeBorder(img_resized, img_bordered, top_, bottom_, left_, right_,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // 返回处理后的图像
    return img_bordered;
}

std::pair<std::vector<int>, std::vector<float>> SequenceInferencer::GetRes(){
    return std::make_pair(predictions_, scores_);
}

void SequenceInferencer::Release(){
	session_->release();
}


