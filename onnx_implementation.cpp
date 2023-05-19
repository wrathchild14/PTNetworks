#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

static std::vector<float> load_image(const std::string& filename, const int size_x = 400, const int size_y = 400)
{
	cv::Mat image = cv::imread(filename);
	if (image.empty())
	{
		std::cout << "No image found.";
	}

	// convert from BGR to RGB
	cvtColor(image, image, cv::COLOR_BGR2RGB);

	resize(image, image, cv::Size(size_x, size_y));

	// normalize from [0, 255] to [-1, 1]
	std::vector<float> output(size_x * size_y * 3);
	for (int row = 0; row < image.rows; ++row)
	{
		for (int col = 0; col < image.cols; ++col)
		{
			auto pixel = image.at<cv::Vec3b>(row, col);
			for (int ch = 0; ch < 3; ++ch)
			{
				output[ch * size_x * size_y + row * size_x + col] = static_cast<float>(pixel[ch]) / 127.5 - 1.0;
			}
		}
	}
	return output;
}

// libraries used Microsoft.ML.OnnxRuntime (nuget package) and opencv (locally)
int main()
{
	Ort::Env env;
	Ort::RunOptions run_options;
	Ort::Session session(nullptr);

	constexpr int64_t channels = 3;
	constexpr int64_t width = 400;
	constexpr int64_t height = 400;
	constexpr int64_t input_elements = channels * height * width;

	// note this works with loading a picture locally (will need to send data here)
	const std::string image_file = "C:\\Git\\DenoisingProject\\image.jpg";
	auto model_path = L"C:\\Git\\DenoisingProject\\model_unet.onnx";

	std::vector<float> image_vec = load_image(image_file, 400, 400);
	if (image_vec.empty())
	{
		std::cout << "Failed to load image: " << image_file << std::endl;
		return 1;
	}

	Ort::SessionOptions ort_session_options;

	OrtCUDAProviderOptions options;
	options.device_id = 0;
	//options.arena_extend_strategy = 0;
	//options.gpu_mem_limit = 2 * 1024 * 1024 * 1024;
	//options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
	//options.do_copy_in_default_stream = 1;

	OrtSessionOptionsAppendExecutionProvider_CUDA(ort_session_options, options.device_id);

	session = Ort::Session(env, model_path, ort_session_options);

	// if CPU
	// session = Ort::Session(env, modelPath, Ort::SessionOptions{ nullptr });

	// define shape
	const std::array<int64_t, 4> input_shape = {1, channels, height, width};
	const std::array<int64_t, 4> output_shape = {1, channels, height, width};

	// define array
	std::vector<float> input(input_elements);
	std::vector<float> output(input_elements); // what goes out to the gpu

	// define Tensor
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), input_shape.data(),
	                                                   input_shape.size());
	auto output_tensor = Ort::Value::CreateTensor<float>(memory_info, output.data(), output.size(), output_shape.data(),
	                                                    output_shape.size());

	// Resize input vector
	input.resize(image_vec.size());

	// Copy image data to input array
	std::copy(image_vec.begin(), image_vec.end(), input.begin());

	// define names
	Ort::AllocatorWithDefaultOptions ort_alloc;
	Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, ort_alloc);
	Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, ort_alloc);
	const std::array<const char*, 1> input_names = {input_name.get()};
	const std::array<const char*, 1> output_names = {output_name.get()};
	input_name.release();
	output_name.release();


	// run inference
	try
	{
		session.Run(run_options, input_names.data(), &input_tensor, 1, output_names.data(), &output_tensor, 1);
	}
	catch (Ort::Exception& e)
	{
		std::cout << e.what() << std::endl;
		return 1;
	}

	const std::string output_image_file = R"(C:\Git\DenoisingProject\output_image.jpg)";

	// normalize the output data from -1 to 1 to the range of 0 to 1
	std::vector<float> outputNormalized(output.size());
	for (size_t i = 0; i < output.size(); ++i)
	{
		outputNormalized[i] = (output[i] + 1.0) / 2.0;
	}

	// transpose (Chanel, Height, Width) to (Height, Width, Channel)
	cv::Mat outputImage(height, width, CV_32FC3);
	for (size_t row = 0; row < height; ++row)
	{
		for (size_t col = 0; col < width; ++col)
		{
			for (size_t ch = 0; ch < 3; ++ch)
			{
				outputImage.at<cv::Vec3f>(row, col)[ch] = outputNormalized[ch * height * width + row * width + col];
			}
		}
	}

	// from float to uint8
	cv::Mat output_image_uint8;
	outputImage.convertTo(output_image_uint8, CV_8UC3, 255.0);

	cvtColor(output_image_uint8, output_image_uint8, cv::COLOR_RGB2BGR);

	// save the output image
	imwrite(output_image_file, output_image_uint8);

	std::cout << "De-noised output saved as: " << output_image_file << std::endl;
}
