# Phoenix Whisper 🎙️✨

**A professional, resumable, and self-deploying transcription tool powered by OpenAI's Whisper. Designed for long-form videos, it offers a truly "zero-setup" experience and adaptive performance.**

![Screenshot of the tool in action](URL_TO_YOUR_SCREENSHOT.png)  <!-- ضع هنا رابط لصورة الأداة وهي تعمل -->

---

This isn't just another script for Whisper. It's a complete, battle-tested application built to handle the real-world challenges of transcribing very long audio and video files reliably. It was born out of the frustration of dealing with crashes, complex setups, and lost progress.

## 🔥 Key Features

*   🚀 **Zero-Setup Experience (The Phoenix Engine)**
    *   **Self-Installing:** On its first run, the tool automatically creates its own virtual environment and installs all necessary dependencies. No manual `pip install` needed!
    *   **Pre-Flight Checks:** Before starting, it validates your system (FFmpeg, Git, Internet, Disk Space) to prevent common errors.

*   🛡️ **Ultimate Reliability (Never Lose Your Work)**
    *   **Resumable Progress:** Stop the transcription at any time (`Ctrl+C`). When you run it again, it picks up exactly where it left off. Perfect for 3+ hour videos.
    *   **Smart Error Handling:** Automatically handles transcription timeouts and other common issues without crashing.

*   ⚡ **Adaptive Performance**
    *   **Hardware-Aware:** Intelligently detects your hardware (CPU cores, GPU availability) to configure the optimal number of parallel workers.
    *   **Dynamic Chunking:** Adjusts video chunk size based on your system's resources to balance speed and memory usage.

*   ✨ **Superior User Experience**
    *   **Rich CLI Interface:** A beautiful and informative command-line interface powered by `rich` that keeps you updated on the progress.
    *   **Unified Live Subtitles:** Always maintains a single, clean, and sorted partial subtitle file (`_IN_PROGRESS.srt`) that you can preview at any time.

## 🏁 Quick Start

Getting started is as simple as it gets.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/phoenix-whisper.git
    cd phoenix-whisper
    ```

2.  **Run the application:**
    ```bash
    python run.py
    ```

That's it! On the first run, the script will set up its own environment. Subsequent runs will start the application instantly.

## 🔧 For Developers (Optional )

If you wish to contribute to the project, you can set up the development environment.

1.  **Create and activate the virtual environment manually:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install development dependencies:**
    ```bash
    pip install -r requirements-dev.txt
    ```

3.  **Run tests:**
    ```bash
    pytest
    ```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Made with ❤️ and a lot of trial and error.
