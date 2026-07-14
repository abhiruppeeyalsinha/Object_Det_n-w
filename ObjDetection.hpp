#ifndef OBJDETECTION_H_
#define OBJDETECTION_H_

// ---------- Header Inclusion ----------
#include "Includes.h"
#include "Struct.h"

// ---------- IPCard Status Variables ----------
extern bool AIStatusFlag;

// ---------- AI Flag Variables ----------
extern bool TargetDataFlag;
extern bool DetectObjectFlag;
extern bool BirdPlaneDataFlag;
extern bool TemplateMatchingFlag;

// ---------- Camera Variables ----------
extern Size VideoDim;

// ---------- AI Parameters ----------
extern string ModelPath;
extern Int16_t ROI_Loc[2];
extern UInt16_t ROI_Dim[2];
extern string ClassName[4];
extern struct _TargetData_ TargetData;

// ---------- Class Declaration ----------
class ObjDetection
{
private:
	UInt32_t Detections;
	UInt32_t DetectionSize;
	UInt32_t ClassCnt;
	Size Yolov5_Dim;

	float ConfThreshold;
	float IOUThreshold;

	vector<struct _AIData_> TargetDataArr;

	float TempConfThreshold;
	UInt16_t TemplateFrameCnt;
	UInt8_t ActiveArr, StoreArr, MaxTemplateCnt;
	vector<struct _Template_> TemplateArr[2];

	unique_ptr<IRuntime> runtime;
	unique_ptr<ICudaEngine> engine;
	unique_ptr<IExecutionContext> context;
	vector<void *> Bindings;
	vector<size_t> BindingSizes;

	cudaStream_t rawStream;
	cuda::Stream cvStream;

public:
	// Constructor
	ObjDetection(void);

	// Destructor
	~ObjDetection(void);

	// Get Memory Size
	size_t getMemorySize(Dims dims, size_t ElementSize);

	// Load Engine File
	bool LoadEngineFile(void);

	// PreProcess
	void preProcess(Mat VideoFrame, __half *InputBuff);

	// PostProcess
	void postProcess(UInt16_t VideoWidth, UInt16_t VideoHeight);

	// Template Matching
	void templateMatch(Mat &VideoFrame);

	// Draw Boxes & Sort Using ROI
	void drawBoxes(Mat &VideoFrame);

	// Detect Object Function
	void RunEnqueueV2(Mat &VideoFrame);
};

#endif /* OBJDETECTION_H_ */