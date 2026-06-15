// main.cpp
#include <iostream>
#include <string>

#include "auto_drone_tracker.hpp"

using namespace std;

// ════════════════════════════════════════════════════════════════

int main()
{
    const string ir_source =
        "rtsp://192.168.1.101:554/live1";

    const string rgb_source =
        "/home/abhirupsinha/Videos/Videos/EO_dual_drone_vid.avi";

    const string eo_engine_path =
        "/home/abhirupsinha/Desktop/P207/TrT_Tracking/AI_model/EO_MOdel.engine";

    const string ir_engine_path =
        "/home/abhirupsinha/Desktop/P207/TrT_Tracking/AI_model/IR_Model.engine";

    // =========================================================
    // STEP 1 : SELECT CAMERA
    // =========================================================

    cout << "\n=================================================\n";
    cout << " SELECT TRACKING CAMERA\n";
    cout << "=================================================\n";
    cout << " [1] IR Camera    (AI: IR_Model.engine, output 1x25200x9)\n";
    cout << " [2] EO/RGB Camera (AI: EO_MOdel.engine, output 1x300x6)\n\n";

    string cam_sel;

    cout << " Select : ";
    getline(cin, cam_sel);

    bool use_ir = (cam_sel != "2");

    string tracking_source;
    string engine_path;
    TRTOutputLayout trt_output_layout;

    if (use_ir)
    {
        tracking_source = ir_source;
        engine_path = ir_engine_path;
        trt_output_layout = TRTOutputLayout::IR_25200x9;

        cout << "\n[INFO] Using IR stream for tracking\n";
        cout << "[INFO] AI model profile: IR output (1,25200,9)\n";
    }
    else
    {
        tracking_source = rgb_source;
        engine_path = eo_engine_path;
        trt_output_layout = TRTOutputLayout::EO_300x6;

        cout << "\n[INFO] Using EO stream for tracking\n";
        cout << "[INFO] AI model profile: EO output (1,300,6)\n";
    }

    // =========================================================
    // STEP 2 : USER CONFIGURATION FIRST
    // =========================================================

    AutoDroneTracker tracker(
        tracking_source,
        engine_path,
        trt_output_layout,
        0.40f);

    // =========================================================
    // STEP 3 : START TRACKER
    // =========================================================

    cout << "\n[INFO] System Ready\n";
    cout << "\n[INFO] Tracking Started\n";
    cout << "Press ESC to exit\n\n";

    tracker.run();

    return 0;
}
