#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <thread>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#endif

// Invisible Zero-Width Unicode Constants (encoded in UTF-8 bytes)
// U+200B (Zero-Width Space) -> 0xEF 0xBB 0xBF in UTF-8
// U+200D (Zero-Width Joiner) -> 0xE2 0x80 0x8D in UTF-8
// U+FEFF (BOM / Delimiter) -> 0xEF 0xBB 0xBF in UTF-8
const std::string ZERO_WIDTH_0 = "\xE2\x80\x8B"; 
const std::string ZERO_WIDTH_1 = "\xE2\x80\x8D"; 
const std::string DELIMITER    = "\xEF\xBB\xBF"; 

class WatermarkEngine {
public:
    // Convert ASCII string to Zero-Width UTF-8 String
    static std::string encodeToZeroWidth(const std::string& secretText) {
        std::string zeroWidthResult = DELIMITER;

        for (char c : secretText) {
            // Convert byte to 8 bits
            for (int i = 7; i >= 0; --i) {
                bool bit = (c >> i) & 1;
                zeroWidthResult += (bit ? ZERO_WIDTH_1 : ZERO_WIDTH_0);
            }
        }
        zeroWidthResult += DELIMITER;
        return zeroWidthResult;
    }

    // Extract original ASCII string from Zero-Width UTF-8 String
    static std::string decodeFromZeroWidth(const std::string& text) {
        size_t firstDelim = text.find(DELIMITER);
        size_t lastDelim = text.rfind(DELIMITER);

        if (firstDelim == std::string::npos || lastDelim == std::string::npos || firstDelim == lastDelim) {
            return "[!] No valid watermark found.";
        }

        std::string payload = text.substr(firstDelim + DELIMITER.length(), lastDelim - (firstDelim + DELIMITER.length()));
        std::string bitStream = "";

        // Parse zero-width characters into binary string
        for (size_t i = 0; i < payload.length(); ) {
            if (payload.compare(i, ZERO_WIDTH_0.length(), ZERO_WIDTH_0) == 0) {
                bitStream += "0";
                i += ZERO_WIDTH_0.length();
            } else if (payload.compare(i, ZERO_WIDTH_1.length(), ZERO_WIDTH_1) == 0) {
                bitStream += "1";
                i += ZERO_WIDTH_1.length();
            } else {
                i++; // Skip non-matching bytes if any noise exists
            }
        }

        // Reconstruct characters from bits
        std::string decodedText = "";
        for (size_t i = 0; i < bitStream.length(); i += 8) {
            if (i + 8 > bitStream.length()) break;
            std::string byteStr = bitStream.substr(i, 8);
            char ch = static_cast<char>(std::bitset<8>(byteStr).to_ulong());
            decodedText += ch;
        }

        return decodedText;
    }
};

#ifdef _WIN32
// Native Windows Clipboard Handlers
class WinClipboard {
public:
    static std::string getText() {
        if (!OpenClipboard(nullptr)) return "";
        HANDLE hData = GetClipboardData(CF_TEXT);
        if (!hData) {
            CloseClipboard();
            return "";
        }
        char* pszText = static_cast<char*>(GlobalLock(hData));
        std::string text = pszText ? pszText : "";
        GlobalUnlock(hData);
        CloseClipboard();
        return text;
    }

    static void setText(const std::string& text) {
        if (!OpenClipboard(nullptr)) return;
        EmptyClipboard();
        HGLOBAL hGlob = GlobalAlloc(GMEM_MOVEABLE, text.size() + 1);
        if (!hGlob) {
            CloseClipboard();
            return;
        }
        memcpy(GlobalLock(hGlob), text.c_str(), text.size() + 1);
        GlobalUnlock(hGlob);
        SetClipboardData(CF_TEXT, hGlob);
        CloseClipboard();
    }
};
#endif

int main(int argc, char* argv[]) {
    // Mode 1: Extraction / Decoding Mode
    if (argc > 1 && std::string(argv[1]) == "--decode") {
        std::cout << "Paste the suspicious text here and press ENTER:\n";
        std::string leakedText;
        std::getline(std::cin, leakedText);
        std::cout << "\n[+] Extraction Result: " 
                  << WatermarkEngine::decodeFromZeroWidth(leakedText) << std::endl;
        return 0;
    }

    // Mode 2: Clipboard Injection Background Daemon
    std::string deviceID = "DEV-WIN-9082"; // Hardcode machine ID or pull via system calls
    std::string watermarkPayload = WatermarkEngine::encodeToZeroWidth("ID:" + deviceID);

    std::cout << "[+] C++ Watermark Daemon running... (Tracking Device: " << deviceID << ")\n";

#ifdef _WIN32
    std::string lastText = "";
    while (true) {
        std::string currentText = WinClipboard::getText();

        // Inject watermark if text copied is non-empty, changed, and not already watermarked
        if (!currentText.empty() && currentText != lastText && currentText.find(DELIMITER) == std::string::npos) {
            std::string protectedText = currentText + watermarkPayload;
            WinClipboard::setText(protectedText);
            lastText = protectedText;
            std::cout << "[!] Watermark successfully injected into active clipboard text." << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
#else
    std::cout << "Clipboard daemon loop requires platform-specific windowing APIs on Linux/macOS.\n";
#endif

    return 0;
}
