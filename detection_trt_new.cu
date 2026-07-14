// ObjDetection.cu
// ---------- Header Inclusion ----------
#include "ObjDetection.h"

// ---------- Logger for TensorRT messages ----------
class Logger : public ILogger
{
	void log(Severity severity, const char *msg) noexcept override
	{
		if (severity <= Severity::kWARNING)
		{
			cout << "[TensorRT] : " << msg << endl;
		}
	}
};

Logger logger;

__global__ void ConvToFP16_NormCHW(const float *ch0, const float *ch1, const float *ch2, __half *output, int32_t width, int32_t height)
{
	int32_t x, y, idx, imgArea;

	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
	{
		return;
	}

	idx = y * width + x;
	imgArea = width * height;
	output[0 * imgArea + idx] = __float2half(ch0[idx] / 255.0f);
	output[1 * imgArea + idx] = __float2half(ch1[idx] / 255.0f);
	output[2 * imgArea + idx] = __float2half(ch2[idx] / 255.0f);
}

__global__ void fp16_to_fp32(const __half *input, float *output, int32_t N)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		output[idx] = __half2float(input[idx]);
	}
}

// ---------- Constructor ----------
ObjDetection ::ObjDetection(void)
{
	Yolov5_Dim.height = 640;
	Yolov5_Dim.width = 640;

	Detections = 25200U;
	DetectionSize = 9U;
	ClassCnt = 4U;

	ConfThreshold = 0.60f;
	IOUThreshold = 0.4f;

	TargetDataArr.clear();

	ActiveArr = 0;
	StoreArr = 1;
	TemplateArr[0].clear();
	TemplateArr[1].clear();

	MaxTemplateCnt = 10U;
	TemplateFrameCnt = 5U;
	TempConfThreshold = 0.85f;

	if (LoadEngineFile() == true)
	{
		AIStatusFlag = true;
		cout << "Engine Loaded Successfully" << endl;
	}
	else
	{
		AIStatusFlag = false;
		cout << "Failed to load Engine file" << endl;
	}

	return;
}

// ---------- Destructor ----------
ObjDetection ::~ObjDetection(void)
{
	try
	{
		for (void *ptr : Bindings)
		{
			if (ptr)
			{
				cudaFree(ptr);
			}
		}

		cudaStreamDestroy(rawStream);
	}
	catch (Exception &e)
	{
	}

	return;
}

// ---------- Get Memory Size ----------
size_t ObjDetection ::getMemorySize(Dims dims, size_t ElementSize)
{
	size_t size = ElementSize;

	for (Int32_t i = 0; i < dims.nbDims; i++)
	{
		size *= dims.d[i];
	}

	return size;
}

// ---------- Load Engine File ----------
bool ObjDetection ::LoadEngineFile(void)
{
	Dims dims;
	size_t elementSize;

	try
	{
		setenv("CUDA_MODULE_LOADING", "LAZY", 1);

		ifstream EngineFile(ModelPath, ios::binary);
		if (!EngineFile)
		{
			cerr << "Failed to open engine file." << endl;
			return false;
		}

		EngineFile.seekg(0, ios::end);

		auto fsize = EngineFile.tellg();
		EngineFile.seekg(0, ios::beg);

		vector<char> engine_data(fsize);
		EngineFile.read(engine_data.data(), fsize);

		runtime.reset(createInferRuntime(logger));
		engine.reset(runtime->deserializeCudaEngine(engine_data.data(), fsize));
		context.reset(engine->createExecutionContext());

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

		Int32_t nbBindings = engine->getNbBindings();
		Bindings.resize(nbBindings, nullptr);
		BindingSizes.resize(nbBindings, 0);

		for (Int32_t i = 0; i < nbBindings; i++)
		{
			dims = engine->getBindingDimensions(i);
			elementSize = sizeof(__half);

			BindingSizes[i] = getMemorySize(dims, elementSize);
			cudaMalloc(&Bindings[i], BindingSizes[i]);
		}

#pragma GCC diagnostic pop

		rawStream = cuda::StreamAccessor::getStream(cvStream);
		return true;
	}
	catch (Exception &e)
	{
		return false;
	}

	return false;
}

// ---------- PreProcess ----------
void ObjDetection ::preProcess(Mat VideoFrame, __half *InputBuff)
{
	dim3 block(32, 32);
	cuda::GpuMat Frame, GpuCHW[3];
	dim3 grid((Yolov5_Dim.width + block.x - 1) / block.x, (Yolov5_Dim.height + block.y - 1) / block.y);

	Frame.upload(VideoFrame, cvStream);

	// Created padded output(resize by padding to 640x480 are within 640x640)
	cuda::GpuMat GpuFrame(Yolov5_Dim.width, Yolov5_Dim.height, Frame.type(), Scalar::all(0));
	Frame.copyTo(GpuFrame(Rect(0, 0, VideoDim.width, VideoDim.height)), cvStream);

	// Convert the float32 & normalize on GPU
	GpuFrame.convertTo(GpuFrame, CV_32FC3, cvStream);
	cuda::split(GpuFrame, GpuCHW, cvStream);

	// Launch custom FP16 kernel conversion & normalization kernel
	ConvToFP16_NormCHW<<<grid, block, 0, rawStream>>>(GpuCHW[0].ptr<float>(), GpuCHW[1].ptr<float>(), GpuCHW[2].ptr<float>(), static_cast<__half *>(InputBuff), Yolov5_Dim.width, Yolov5_Dim.height);

	return;
}

// ---------- PostProcess ----------
void ObjDetection ::postProcess(UInt16_t VideoWidth, UInt16_t VideoHeight)
{
	int32_t Total = DetectionSize * Detections;

	vector<Rect> Boxes;
	vector<Int32_t> Indices;
	vector<UInt8_t> ClassIds;
	vector<float> Confidances;

	float Confidence;
	float ClassScore[4];
	float ObjConf, ClassConf;
	UInt8_t ClassID = 0;

	float *FP32_OutBuff = nullptr;
	dim3 block(1024), grid((Total + 255) / 256);

	vector<float> Output(Total);

	Int16_t CentreX, CentreY, Width, Height;
	Int16_t X1, Y1, X2, Y2;

	struct _AIData_ Target;

	Boxes.clear();
	Indices.clear();
	ClassIds.clear();
	Confidances.clear();
	TargetDataArr.clear();

	// Allocate device memory for FP32
	cudaMalloc(&FP32_OutBuff, Total * sizeof(float));

	// Launch fp16->fp32 conversion
	fp16_to_fp32<<<grid, block, 0, rawStream>>>(static_cast<const __half *>(Bindings[1]), FP32_OutBuff, Total);

	cudaMemcpyAsync(Output.data(), FP32_OutBuff, Output.size() * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(FP32_OutBuff);

	cvStream.waitForCompletion();

	for (UInt32_t i = 0; i < Detections; i++)
	{
		ObjConf = Output[(i * DetectionSize) + ClassCnt];

		// Skip if Object Confidence is less than Threshold
		if (ObjConf < ConfThreshold || ObjConf > 1.0f || isnan(ObjConf))
		{
			continue;
		}

		for (UInt32_t j = 0; j < ClassCnt; j++)
		{
			ClassScore[j] = Output[(i * DetectionSize) + 5 + j];
		}

		ClassID = max_element(ClassScore, ClassScore + ClassCnt) - ClassScore;
		ClassConf = ClassScore[ClassID];
		Confidence = ObjConf * ClassConf;

		// Calculate Bounding Box Co-Ordinate & Size
		CentreX = Output[i * DetectionSize + 0];
		CentreY = Output[i * DetectionSize + 1];
		Width = Output[i * DetectionSize + 2];
		Height = Output[i * DetectionSize + 3];

		// Skip if Confidence is less than Threshold
		if (CentreX < 0 || CentreX > Yolov5_Dim.width || CentreY < 0 || CentreY > Yolov5_Dim.height || Width <= 0 || Height <= 0 || Width > Yolov5_Dim.width || Height > Yolov5_Dim.height || isnan(CentreX) || isnan(CentreY) || isnan(Width) || isnan(Height))
		{
			continue;
		}

		X1 = FindMax(0, (CentreX - (Width / 2)));
		Y1 = FindMax(0, (CentreY - (Height / 2)));

		X2 = FindMin(VideoWidth - 1, (CentreX + (Width / 2)));
		Y2 = FindMin(VideoHeight - 1, (CentreY + (Height / 2)));

		if (X2 >= X1 && Y2 >= Y1)
		{
			Boxes.emplace_back(Rect(X1, Y1, X2 - X1, Y2 - Y1));
			Confidances.emplace_back(Confidence);
			ClassIds.emplace_back(ClassID);
		}
	}

	dnn::NMSBoxes(Boxes, Confidances, ConfThreshold, IOUThreshold, Indices);

	for (UInt32_t idx : Indices)
	{
		Target.Type = 0;
		Target.ClassID = ClassIds[idx];
		Target.Confidance = Confidances[idx];
		Target.BoundingBox = Boxes[idx];
		Target.CenterPoint = Point(Boxes[idx].x + Boxes[idx].width / 2, Boxes[idx].y + Boxes[idx].height / 2);

		TargetDataArr.push_back(Target);
	}

	return;
}

// ---------- Template Matching ----------
void ObjDetection ::templateMatch(Mat &VideoFrame)
{
	UInt16_t FrameWidth = VideoFrame.cols;
	UInt16_t FrameHeight = VideoFrame.rows;

	UInt32_t DetectionCnt = 0U;

	const float ROIScale = 8.0f;
	const float OverLapScale = 8.0f;
	const Int16_t StaticPixelTolerance = 3;     // pixels of movement to consider "still moving"
	const UInt8_t StaticFrameThreshold = 4U;    // frames of near-zero movement before dropping

	Rect bbox;
	float Confidance;
	Point minLoc, maxLoc;
	double minVal, maxVal;
	Mat ResFrame, ROIFrame;

	struct _AIData_ Target;
	struct _Template_ Template;

	// Count how many AI detections we got this frame (for the edge case of zero detections)
	for (auto &Tgtit : TargetDataArr)
	{
		if (Tgtit.ClassID < 4)
		{
			DetectionCnt++;
		}
	}

	// Phase 1: Remove OverLapping Templates & Save New Templates from AI detections
	for (auto &Tgtit : TargetDataArr)
	{
		if (4 <= Tgtit.ClassID)
		{
			continue;
		}

		// Remove OverLapping Templates
		for (auto Tmpit = TemplateArr[ActiveArr].begin(); Tmpit != TemplateArr[ActiveArr].end();)
		{
			if (Tmpit->OverLapBox.contains(Tgtit.CenterPoint) || Tgtit.BoundingBox.contains(Tmpit->CenterPoint))
			{
				Tmpit->Template.release();
				Tmpit = TemplateArr[ActiveArr].erase(Tmpit);
			}
			else
			{
				Tmpit++;
			}
		}

		if (BirdPlaneDataFlag == false && (Tgtit.ClassID == 0 || Tgtit.ClassID == 3))
		{
			continue;
		}

		// Save New Templates from AI detections
		bbox = Tgtit.BoundingBox & Rect(0, 0, FrameWidth, FrameHeight);

		Template.FrameCnt = 0U;
		Template.StaticCnt = 0U;                    // NEW: initialize static counter
		Template.LastCenter = Tgtit.CenterPoint;    // NEW: seed the movement reference
		Template.ClassID = Tgtit.ClassID;
		Template.CenterPoint = Tgtit.CenterPoint;
		Template.BoundingBox = Tgtit.BoundingBox;

		Template.Template = VideoFrame(bbox).clone();

		Template.ROIBox.width = FindMin(FrameWidth, ((UInt16_t)(bbox.width * ROIScale)));
		Template.ROIBox.height = FindMin(FrameHeight, ((UInt16_t)(bbox.height * ROIScale)));
		Template.ROIBox.x = FindMax(0, (Tgtit.CenterPoint.x - (Template.ROIBox.width / 2)));
		Template.ROIBox.y = FindMax(0, (Tgtit.CenterPoint.y - (Template.ROIBox.height / 2)));
		Template.ROIBox = Template.ROIBox & Rect(0, 0, FrameWidth, FrameHeight);

		Template.OverLapBox.width = FindMin(FrameWidth, ((UInt16_t)(bbox.width * OverLapScale)));
		Template.OverLapBox.height = FindMin(FrameHeight, ((UInt16_t)(bbox.height * OverLapScale)));
		Template.OverLapBox.x = FindMax(0, (Tgtit.CenterPoint.x - (Template.OverLapBox.width / 2)));
		Template.OverLapBox.y = FindMax(0, (Tgtit.CenterPoint.y - (Template.OverLapBox.height / 2)));
		Template.OverLapBox = Template.OverLapBox & Rect(0, 0, FrameWidth, FrameHeight);

		if (TemplateArr[StoreArr].size() < MaxTemplateCnt)
		{
			TemplateArr[StoreArr].push_back(Template);
		}
	}

	// Phase 2: Template Matching - track every active template independently
	for (auto Tmpit = TemplateArr[ActiveArr].begin(); Tmpit != TemplateArr[ActiveArr].end();)
	{
		ResFrame.release();
		ROIFrame.release();

		ROIFrame = VideoFrame(Tmpit->ROIBox).clone();

		matchTemplate(ROIFrame, Tmpit->Template, ResFrame, TM_SQDIFF_NORMED);
		minMaxLoc(ResFrame, &minVal, &maxVal, &minLoc, &maxLoc);

		Confidance = 1.0f - static_cast<float>(minVal);

		// Quality check - drop if confidence too low
		if (minVal > TempConfThreshold)
		{
			Tmpit->Template.release();
			Tmpit = TemplateArr[ActiveArr].erase(Tmpit);
			continue;
		}

		bbox.x = Tmpit->ROIBox.x + minLoc.x;
		bbox.y = Tmpit->ROIBox.y + minLoc.y;
		bbox.width = Tmpit->Template.cols;
		bbox.height = Tmpit->Template.rows;
		bbox = bbox & Rect(0, 0, FrameWidth, FrameHeight);

		Point newCenter(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);

		// --- NEW: Static-target rejection ---
		// Compute how much the target moved since last frame
		double movement = norm(newCenter - Tmpit->LastCenter);

		if (movement < StaticPixelTolerance)
		{
			Tmpit->StaticCnt++;
		}
		else
		{
			Tmpit->StaticCnt = 0U;
		}

		// If stationary for too many consecutive frames, drop it
		if (Tmpit->StaticCnt >= StaticFrameThreshold)
		{
			Tmpit->Template.release();
			Tmpit = TemplateArr[ActiveArr].erase(Tmpit);
			continue;   // don't report or refresh
		}
		// --- END new section ---

		// Add Template Data Targets Data
		Target.Type = 1;
		Target.ClassID = Tmpit->ClassID;
		Target.Confidance = Confidance;
		Target.BoundingBox = bbox;
		Target.CenterPoint = newCenter;

		TargetDataArr.push_back(Target);

		// --- FIXED: Refresh EVERY valid template independently (multi-target fix) ---
		// Old bug: only refreshed if Tmpit->CenterPoint == TargetData.CenterPoint
		// Now: refresh whenever FrameCnt < (TemplateFrameCnt - 1), or on zero-detection singleton fallback
		if ((Tmpit->FrameCnt < (TemplateFrameCnt - 1)) ||
		    (DetectionCnt == 0U && TemplateArr[StoreArr].size() == 0U && TemplateArr[ActiveArr].size() == 1U))
		{
			Template.BoundingBox = bbox;
			Template.ClassID = Target.ClassID;
			Template.FrameCnt = Tmpit->FrameCnt + 1U;
			Template.StaticCnt = Tmpit->StaticCnt;    // carry forward the static counter
			Template.LastCenter = newCenter;           // update movement reference
			Template.CenterPoint = newCenter;

			Template.Template = VideoFrame(bbox).clone();

			Template.ROIBox.width = FindMin(FrameWidth, ((UInt16_t)(bbox.width * ROIScale)));
			Template.ROIBox.height = FindMin(FrameHeight, ((UInt16_t)(bbox.height * ROIScale)));
			Template.ROIBox.x = FindMax(0, (newCenter.x - (Template.ROIBox.width / 2)));
			Template.ROIBox.y = FindMax(0, (newCenter.y - (Template.ROIBox.height / 2)));
			Template.ROIBox = Template.ROIBox & Rect(0, 0, FrameWidth, FrameHeight);

			Template.OverLapBox.width = FindMin(FrameWidth, ((UInt16_t)(bbox.width * OverLapScale)));
			Template.OverLapBox.height = FindMin(FrameHeight, ((UInt16_t)(bbox.height * OverLapScale)));
			Template.OverLapBox.x = FindMax(0, (newCenter.x - (Template.OverLapBox.width / 2)));
			Template.OverLapBox.y = FindMax(0, (newCenter.y - (Template.OverLapBox.height / 2)));
			Template.OverLapBox = Template.OverLapBox & Rect(0, 0, FrameWidth, FrameHeight);

			if (TemplateArr[StoreArr].size() < MaxTemplateCnt)
			{
				TemplateArr[StoreArr].push_back(Template);
			}
		}
		// else: FrameCnt reached TemplateFrameCnt-1 → refresh skipped, template dies naturally

		// Delete Current Template
		Tmpit->Template.release();
		Tmpit = TemplateArr[ActiveArr].erase(Tmpit);
	}

	ResFrame.release();
	ROIFrame.release();

	// Switch Template Array
	ActiveArr = !ActiveArr;
	StoreArr = !StoreArr;

	return;
}

// ---------- Draw Boxes & Sort Using ROI ----------
void ObjDetection ::drawBoxes(Mat &VideoFrame)
{
	Int8_t thin = 1;
	string Label = "";
	Scalar white(255, 255, 255);

	UInt16_t FrameWidth = VideoFrame.cols;
	UInt16_t FrameHeight = VideoFrame.rows;
	Point FrameCentre(FrameWidth / 2, FrameHeight / 2);

	bool found = false;
	struct _AIData_ best;
	vector<struct _AIData_> Targets;
	double Dist, MinDist = numeric_limits<double>::max();

	// AOI Search
	Int16_t xExpStep = 10, yExpStep = 8;
	Rect AOI((ROI_Loc[0] - ROI_Dim[0] / 2), (ROI_Loc[1] - ROI_Dim[1] / 2), ROI_Dim[0], ROI_Dim[1]);
	AOI = AOI & Rect(0, 0, FrameWidth, FrameHeight);

	memset((void *)&TargetData, 0, sizeof(TargetData));

	// Draw Bounding Boxes
	for (auto &Target : TargetDataArr)
	{
		if (4 <= Target.ClassID)
		{
			continue;
		}

		rectangle(VideoFrame, Target.BoundingBox, white, thin);
		Label = ((Target.Type == 1) ? "TM: " : "AI: ") + ClassName[Target.ClassID] + " " + to_string((UInt32_t)(Target.Confidance * 100)) + "% (" + to_string(Target.CenterPoint.x - VideoFrame.cols / 2) + ", " + to_string(Target.CenterPoint.y - VideoFrame.rows / 2) + ")";
		putText(VideoFrame, Label, Target.BoundingBox.tl(), FONT_HERSHEY_SIMPLEX, 0.45, white, thin);

		cout << ((Target.Type == 1) ? "TM : " : "AI : ") << ClassName[Target.ClassID] << " ID:" << (UInt32_t)Target.ClassID << " Conf:" << (UInt32_t)(Target.Confidance * 100) << "% Ny:" << (Target.CenterPoint.x - VideoFrame.cols / 2) << " Nz:" << (Target.CenterPoint.y - VideoFrame.rows / 2) << " W:" << Target.BoundingBox.width << " H:" << Target.BoundingBox.height << endl
			 << flush;
	}

	// Sort Targets using ROI
	while (found == false)
	{
		Targets.clear();

		if (AOI.x < 0)
		{
			AOI.x = 0;
		}

		if (AOI.y < 0)
		{
			AOI.y = 0;
		}

		if (FrameWidth < (AOI.x + AOI.width))
		{
			AOI.width = FrameWidth - AOI.x;
		}

		if (FrameHeight < (AOI.y + AOI.height))
		{
			AOI.height = FrameHeight - AOI.y;
		}

		AOI = AOI & Rect(0, 0, FrameWidth, FrameHeight);

		for (auto &Target : TargetDataArr)
		{
			if (4 <= Target.ClassID || (BirdPlaneDataFlag == false && (Target.ClassID == 0 || Target.ClassID == 3)))
			{
				continue;
			}

			if (AOI.contains(Target.CenterPoint))
			{
				Targets.push_back(Target);
			}
		}

		if (!Targets.empty())
		{
			found = true;
			MinDist = 1e9;

			for (auto &Target : Targets)
			{
				Dist = norm(Target.CenterPoint - FrameCentre);
				if (Dist < MinDist)
				{
					MinDist = Dist;
					best = Target;
				}
			}

			TargetData.ClassID = best.ClassID;
			TargetData.Confidance = best.Confidance;
			TargetData.CenterPoint = best.CenterPoint;
			TargetData.BoundingBox = best.BoundingBox;
			TargetData.ValidData = 1;

			break;
		}

		if (AOI.width >= FrameWidth && AOI.height >= FrameHeight)
		{
			TargetData.ValidData = 0;
			break;
		}

		AOI.x -= xExpStep;
		AOI.y -= yExpStep;
		AOI.width += (xExpStep * 2);
		AOI.height += (yExpStep * 2);
	}

	// Draw ROI
	rectangle(VideoFrame, AOI, white, thin);

	// Draw Line from Frame Center to Target
	if (TargetData.ValidData == 1)
	{
		line(VideoFrame, FrameCentre, TargetData.CenterPoint, white, thin);
	}

	return;
}

// ---------- Detect Object Function ----------
void ObjDetection ::RunEnqueueV2(Mat &VideoFrame)
{
	// PreProcess
	preProcess(VideoFrame, static_cast<__half *>(Bindings[0]));

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

	// Inference
	context->enqueueV2(Bindings.data(), rawStream, nullptr);
	cudaStreamSynchronize(rawStream);
	cvStream.waitForCompletion();

#pragma GCC diagnostic pop

	// PostProcess
	postProcess(VideoFrame.cols, VideoFrame.rows);

	// Template Matching
	if (DetectObjectFlag == true && TemplateMatchingFlag == true)
	{
		templateMatch(VideoFrame);
	}

	// Draw Boxes
	drawBoxes(VideoFrame);

	return;
}