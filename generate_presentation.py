import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = Presentation()
    # Set 16:9 widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_slide_layout = prs.slide_layouts[6]

    # Theme Colors
    BG_COLOR = RGBColor(15, 23, 42)        # Slate 900 #0f172a
    CARD_BG = RGBColor(30, 41, 59)         # Slate 800 #1e293b
    CARD_BORDER = RGBColor(51, 65, 85)     # Slate 700 #334155
    ACCENT_CYAN = RGBColor(14, 165, 233)   # Sky 500 #0ea5e9
    ACCENT_GREEN = RGBColor(16, 185, 129)  # Emerald 500 #10b981
    ACCENT_ORANGE = RGBColor(245, 158, 11) # Amber 500 #f59e0b
    ACCENT_PURPLE = RGBColor(168, 85, 247) # Purple 500 #a855f7
    TEXT_MAIN = RGBColor(248, 250, 252)    # Slate 50 #f8fafc
    TEXT_MUTED = RGBColor(148, 163, 184)   # Slate 400 #94a3b8
    TEXT_BODY = RGBColor(203, 213, 225)    # Slate 300 #cbd5e1

    def add_bg(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_COLOR
        bg.line.fill.background()
        return bg

    def add_header(slide, category, title, subtitle=None):
        # Category badge/pill
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.45), Inches(11.7), Inches(0.35))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_CYAN

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.6))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_MAIN

        if subtitle:
            p_sub = tf_title.add_paragraph()
            p_sub.text = subtitle
            p_sub.font.size = Pt(13)
            p_sub.font.color.rgb = TEXT_MUTED

    def add_card(slide, left, top, width, height, title, content_items, border_color=None, bg_color=None):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color if bg_color else CARD_BG
        shape.line.color.rgb = border_color if border_color else CARD_BORDER
        shape.line.width = Pt(1.5)

        tb = slide.shapes.add_textbox(Inches(left + 0.25), Inches(top + 0.2), Inches(width - 0.5), Inches(height - 0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        if title:
            p_t = tf.paragraphs[0]
            p_t.text = title
            p_t.font.size = Pt(16)
            p_t.font.bold = True
            p_t.font.color.rgb = ACCENT_CYAN
            p_t.space_after = Pt(8)

        for i, item in enumerate(content_items):
            p = tf.add_paragraph() if (title or i > 0) else tf.paragraphs[0]
            if isinstance(item, tuple):
                bold_part, reg_part = item
                r1 = p.add_run()
                r1.text = "• " + bold_part + ": "
                r1.font.bold = True
                r1.font.size = Pt(12)
                r1.font.color.rgb = TEXT_MAIN
                r2 = p.add_run()
                r2.text = reg_part
                r2.font.size = Pt(12)
                r2.font.color.rgb = TEXT_BODY
            else:
                p.text = "• " + item
                p.font.size = Pt(12)
                p.font.color.rgb = TEXT_BODY
            p.space_after = Pt(6)

    def add_metric_badge(slide, left, top, width, height, label, value, subtext, color):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        shape.fill.solid()
        shape.fill.fore_color.rgb = CARD_BG
        shape.line.color.rgb = color
        shape.line.width = Pt(2)

        tb = slide.shapes.add_textbox(Inches(left + 0.15), Inches(top + 0.15), Inches(width - 0.3), Inches(height - 0.3))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p1 = tf.paragraphs[0]
        p1.text = label.upper()
        p1.font.size = Pt(10)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_MUTED

        p2 = tf.add_paragraph()
        p2.text = value
        p2.font.size = Pt(26)
        p2.font.bold = True
        p2.font.color.rgb = color

        p3 = tf.add_paragraph()
        p3.text = subtext
        p3.font.size = Pt(11)
        p3.font.color.rgb = TEXT_BODY

    # ==========================================
    # SLIDE 1: TITLE SLIDE
    # ==========================================
    slide1 = prs.slides.add_slide(blank_slide_layout)
    add_bg(slide1)

    # Main Title Container
    title_box = slide1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(3.5))
    tf1 = title_box.text_frame
    tf1.word_wrap = True

    p1 = tf1.paragraphs[0]
    p1.text = "INDUSTRIAL DEFECT DETECTION SYSTEM"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_MAIN
    p1.space_after = Pt(10)

    p2 = tf1.add_paragraph()
    p2.text = "Real-Time Edge AI Computer Vision & Automated PLC Sorting Pipeline"
    p2.font.size = Pt(18)
    p2.font.bold = True
    p2.font.color.rgb = ACCENT_CYAN
    p2.space_after = Pt(16)

    p3 = tf1.add_paragraph()
    p3.text = "Production-Ready Architecture for Steel Strip Surface Defect Quality Control"
    p3.font.size = Pt(14)
    p3.font.color.rgb = TEXT_MUTED

    # Bottom Metadata Cards
    add_card(slide1, 1.0, 5.0, 3.5, 1.6, "Core Technologies", [
        ("Framework", "FastAPI & Python 3.13"),
        ("Inference Engine", "ONNX Runtime (CPU/CUDA)"),
        ("Model", "YOLOv8s Defect Detector")
    ], border_color=ACCENT_CYAN)

    add_card(slide1, 4.9, 5.0, 3.5, 1.6, "Edge Capabilities", [
        ("Latency", "~4.8ms GPU / ~28.9ms CPU"),
        ("Industrial Protocol", "WebSocket PLC Telemetry"),
        ("Monitoring", "Prometheus & Glassmorphic UI")
    ], border_color=ACCENT_GREEN)

    add_card(slide1, 8.8, 5.0, 3.5, 1.6, "Deployment", [
        ("Containerization", "Docker & Docker Compose"),
        ("CI/CD", "Automated GitHub Actions"),
        ("Test Coverage", "33/33 Pytest Suite Passed")
    ], border_color=ACCENT_PURPLE)

    # ==========================================
    # SLIDE 2: PROBLEM STATEMENT & INDUSTRIAL CONTEXT
    # ==========================================
    slide2 = prs.slides.add_slide(blank_slide_layout)
    add_bg(slide2)
    add_header(slide2, "Industrial Challenge", "Manufacturing Context & Problem Statement", "Overcoming limitations of traditional manual inspection in high-speed rolling mills")

    add_card(slide2, 0.8, 1.6, 5.6, 5.2, "Traditional Inspection Pitfalls", [
        ("High Line Speeds", "Steel strip lines run at >10 m/s, making manual human inspection physically impossible and prone to eye-fatigue error."),
        ("Micro-Scale Defects", "Defects like crazing and fine scratches exhibit low contrast against raw metallic sheen, escaping human sight."),
        ("Expensive Scrap Losses", "Uncaught defects travel downstream, turning entire coils into scrap or causing catastrophic mechanical failures in client dies."),
        ("Lack of Traceability", "Traditional lines lack automated timestamped coordinate logs and real-time defect telemetry for audit compliance.")
    ], border_color=ACCENT_ORANGE)

    add_card(slide2, 6.9, 1.6, 5.6, 5.2, "Our Edge AI Solution", [
        ("Sub-30ms Real-Time Inference", "Hardware-accelerated ONNX engine delivers real-time defect isolation within the tight timing window of industrial conveyor belts."),
        ("Per-Class Optimized Thresholds", "Custom F1-calibrated confidence tuning prevents false-positives while guaranteeing high recall for critical flaws."),
        ("Simulated PLC Synchronization", "Instantly emits WebSocket JSON coordinate payloads to trigger robotic sorting mechanisms and reject gates."),
        ("Full-Stack Observability", "Prometheus scrape endpoint and glassmorphic telemetry dashboard provide continuous line visibility.")
    ], border_color=ACCENT_GREEN)

    # ==========================================
    # SLIDE 3: TARGET DEFECT TAXONOMY (NEU-DET)
    # ==========================================
    slide3 = prs.slides.add_slide(blank_slide_layout)
    add_bg(slide3)
    add_header(slide3, "Defect Classes", "NEU Metal Surface Defect Taxonomy", "Comprehensive detection across 6 critical hot-rolled steel surface flaws")

    defects = [
        ("Patches (88.1% AP50)", "Thick, high-contrast surface irregularities. Threshold: 0.50. High precision & clear boundary delineation.", ACCENT_GREEN),
        ("Inclusion (77.7% AP50)", "Non-metallic particles trapped inside metal matrix. Threshold: 0.39. Granular multi-point clusters.", ACCENT_CYAN),
        ("Pitted Surface (76.3% AP50)", "Localized pitting/cavities from acid pickling. Threshold: 0.30. Distributed pinhole craters.", ACCENT_PURPLE),
        ("Scratches (75.1% AP50)", "Linear mechanical abrasions from rollers. Threshold: 0.22. Long diagonal thin trails.", ACCENT_CYAN),
        ("Rolled-in Scale (57.5% AP50)", "Iron oxide scales pressed during rolling. Threshold: 0.25. Irregular dark textured patches.", ACCENT_ORANGE),
        ("Crazing (43.3% AP50)", "Network of spiderweb micro-cracks. Threshold: 0.23. Subtle hairline stress patterns.", ACCENT_ORANGE)
    ]

    for i, (d_title, d_desc, d_col) in enumerate(defects):
        row = i // 3
        col = i % 3
        l = 0.8 + col * 4.0
        t = 1.6 + row * 2.6
        add_card(slide3, l, t, 3.7, 2.3, d_title, [d_desc], border_color=d_col)

    # ==========================================
    # SLIDE 4: END-TO-END SYSTEM ARCHITECTURE
    # ==========================================
    slide4 = prs.slides.add_slide(blank_slide_layout)
    add_bg(slide4)
    add_header(slide4, "Pipeline Design", "End-to-End System Architecture", "Data flow from camera ingest to edge inference, visualization, and PLC automation")

    add_card(slide4, 0.8, 1.6, 3.6, 5.2, "1. Ingestion Layer", [
        ("Multi-Source Ingestion", "Ingests live USB/IP industrial camera, local MP4 video file, or looping validation directory."),
        ("VideoStreamProcessor", "Decodes frames, maintains state, handles auto-reconnect and resolution normalization."),
        ("Dynamic Switching", "Enables runtime switching between sources via REST API without server restart.")
    ], border_color=ACCENT_CYAN)

    add_card(slide4, 4.8, 1.6, 3.7, 5.2, "2. AI Inference Engine", [
        ("ONNX Runtime Core", "Hardware-accelerated engine supporting TensorRT, CUDA, and OpenVINO/CPU."),
        ("Letterbox Preprocessing", "Preserves aspect ratio with symmetric zero-padding and normalized float32 tensors."),
        ("Class-Wise NMS", "Non-Maximum Suppression (IoU=0.45) with per-class optimized confidence thresholds.")
    ], border_color=ACCENT_GREEN)

    add_card(slide4, 8.9, 1.6, 3.6, 5.2, "3. Actuation & UI Layer", [
        ("MJPEG Video Stream", "Multipart live annotated video stream with bounding boxes and latency telemetry."),
        ("WebSocket PLC Broadcast", "Coordinates dispatched to sorting actuators within milliseconds of detection."),
        ("Prometheus Metrics", "Counters and Gauges exported at /metrics for Grafana / line monitoring.")
    ], border_color=ACCENT_PURPLE)

    # ==========================================
    # SLIDE 5: EDGE AI OPTIMIZATION & ONNX ACCELERATION
    # ==========================================
    slide5 = prs.slides.add_slide(blank_slide_layout)
    add_bg(slide5)
    add_header(slide5, "Model Optimization", "YOLOv8s & Hardware-Accelerated ONNX", "Ultra-low latency execution on edge computing nodes")

    add_metric_badge(slide5, 0.8, 1.6, 2.7, 1.6, "GPU Latency", "4.8 ms", "RTX 2050 / TensorRT", ACCENT_GREEN)
    add_metric_badge(slide5, 3.8, 1.6, 2.7, 1.6, "CPU Latency", "28.9 ms", "Standard Intel/AMD CPU", ACCENT_CYAN)
    add_metric_badge(slide5, 6.8, 1.6, 2.7, 1.6, "Throughput", "180+ FPS", "GPU Pipeline Capacity", ACCENT_PURPLE)
    add_metric_badge(slide5, 9.8, 1.6, 2.7, 1.6, "Global mAP@50", "69.7%", "Validation Benchmark", ACCENT_GREEN)

    add_card(slide5, 0.8, 3.5, 5.6, 3.4, "Why ONNX Runtime over Raw PyTorch?", [
        ("Zero Python Overhead", "Optimized C++ execution graph eliminates GIL bottleneck and PyTorch tensor overhead."),
        ("Cross-Hardware Portability", "Seamlessly targets x86 edge servers, NVIDIA Jetson, or ARM industrial gateways without code modification."),
        ("Operator Fusion & Constant Folding", "Pre-computes graph constants and fuses Conv+BatchNorm+SiLU kernels for peak memory bandwidth efficiency.")
    ], border_color=ACCENT_CYAN)

    add_card(slide5, 6.9, 3.5, 5.6, 3.4, "Letterbox Preprocessing Architecture", [
        ("Aspect Ratio Preservation", "Calculates exact scaling factor and applies minimum padding to 640x640 input grid."),
        ("Inverse Coordinate Mapping", "Maps bounding box predictions back to original raw image resolution seamlessly."),
        ("Vectorized Array Operations", "Fully implemented in NumPy/OpenCV for sub-millisecond preprocessing latency (~1.8ms).")
    ], border_color=ACCENT_PURPLE)

    # ==========================================
    # SLIDE 6: PER-CLASS THRESHOLD OPTIMIZATION
    # ==========================================
    slide6 = prs.slides.add_slide(blank_slide_layout)
    add_bg(slide6)
    add_header(slide6, "Precision Engineering", "Dynamic Class-Specific Confidence Thresholds", "Balancing Precision vs Recall based on defect severity and visual distinctiveness")

    add_card(slide6, 0.8, 1.6, 5.6, 5.2, "The Problem with Global Thresholds", [
        ("Unequal Defect Saliency", "A global threshold of 0.50 catches Patches (88.1% AP) perfectly, but drops recall on Crazing (43.3% AP) down to 20%."),
        ("Safety vs False Alarm Tradeoff", "Critical safety flaws (Crazing, Scratches) require maximum recall to prevent catastrophic failures."),
        ("Visual Granularity Disparity", "Inclusions and Pitted surfaces produce smaller bounding boxes compared to broad roll patches."),
        ("F1-Score Empirical Tuning", "Thresholds were mathematically tuned on 360 validation images to maximize F1-score per class.")
    ], border_color=ACCENT_ORANGE)

    add_card(slide6, 6.9, 1.6, 5.6, 5.2, "Optimized Threshold Configuration", [
        ("Patches -> 0.50", "High visual contrast allows strict thresholding with 91.2% precision."),
        ("Inclusion -> 0.39", "Optimized to isolate small particle clusters from background grain."),
        ("Pitted Surface -> 0.30", "Filters out lighting reflections while catching surface pinholes."),
        ("Rolled-in Scale -> 0.25", "Prevents texture confusion with unrolled metal substrate."),
        ("Crazing -> 0.23", "Boosts recall for spiderweb cracks blending into metal sheen."),
        ("Scratches -> 0.22", "Maintains detection across faint, low-contrast linear scratch lines.")
    ], border_color=ACCENT_GREEN)

    # ==========================================
    # SLIDE 7: REAL-TIME WEB DASHBOARD & TELEMETRY
    # ==========================================
    slide7 = prs.slides.add_slide(blank_slide_layout)
    add_bg(slide7)
    add_header(slide7, "Operator Interface", "Glassmorphic Web UI & Live Telemetry", "Modern, operator-focused monitoring dashboard built with FastAPI & Chart.js")

    add_card(slide7, 0.8, 1.6, 3.7, 5.2, "1. Live Video Stream", [
        ("Multipart MJPEG", "Delivers smooth video streaming at 30+ FPS directly into the browser without plugins."),
        ("Real-Time Overlay", "Dynamic bounding boxes, class labels, confidence scores, and per-frame latency badges."),
        ("Source Control", "Webcam, directory loop, video file, or instant file drag-and-drop upload.")
    ], border_color=ACCENT_CYAN)

    add_card(slide7, 4.8, 1.6, 3.7, 5.2, "2. Real-Time Telemetry", [
        ("Chart.js Time-Series", "Visualizes real-time FPS and latency fluctuations over sliding time windows."),
        ("Defect Class Distribution", "Live doughnut chart updating defect frequencies by category."),
        ("Status Badges", "Instant system health, camera uptime, and active connection counter.")
    ], border_color=ACCENT_GREEN)

    add_card(slide7, 8.9, 1.6, 3.6, 5.2, "3. Quality Audit Logs", [
        ("Inspection Event Table", "Timestamped event stream recording every detected defect coordinate."),
        ("JSON Audit Export", "One-click /api/export_report for ISO quality management and shift reports."),
        ("REST Detection Sandbox", "Interactive single-image upload with bounding box visualization.")
    ], border_color=ACCENT_PURPLE)

    # ==========================================
    # SLIDE 8: SIMULATED PLC AUTOMATION & INDUSTRIAL SIGNALS
    # ==========================================
    slide8 = prs.slides.add_slide(blank_slide_layout)
    add_bg(slide8)
    add_header(slide8, "Industrial Automation", "Simulated PLC Actuation & Sorting Protocol", "Zero-latency synchronization with Programmable Logic Controllers (PLCs)")

    add_card(slide8, 0.8, 1.6, 5.6, 5.2, "WebSocket PLC Broadcast Protocol", [
        ("Event-Driven Dispatch", "Whenever a defect is confirmed by the ONNX detector, a structured JSON payload is broadcast to all active PLC endpoints."),
        ("Thread-Safe Async Loop", "Background video processing thread dispatches safely into the asyncio event loop via asyncio.run_coroutine_threadsafe."),
        ("Connection Resilience", "Automatic cleanup of disconnected clients (_safe_send_ws) prevents memory leaks or stream stalling.")
    ], border_color=ACCENT_CYAN)

    add_card(slide8, 6.9, 1.6, 5.6, 5.2, "Payload Schema & Sorting Logic", [
        ("Coordinate Precision", "xmin, ymin, xmax, ymax formatted in both absolute pixels and normalized ratios."),
        ("Confidence & Class ID", "Provides actuator with class priority (e.g. Scratches vs Crazing) to trigger differentiated sort bins."),
        ("Timestamp & Frame Index", "Precise millisecond timestamps for encoder wheel synchronization along conveyor lines."),
        ("Actuator Triggers", "Triggers pneumatic ejector gates, marking sprayers, and alarm beacons.")
    ], border_color=ACCENT_GREEN)

    # ==========================================
    # SLIDE 9: DEVOPS, TESTING & REPRODUCIBILITY
    # ==========================================
    slide9 = prs.slides.add_slide(blank_slide_layout)
    add_bg(slide9)
    add_header(slide9, "Quality Assurance", "Enterprise DevOps, Testing & Containerization", "Built for high reliability, automated verification, and rapid deployment")

    add_card(slide9, 0.8, 1.6, 3.7, 5.2, "Docker Containerization", [
        ("Multi-Stage Dockerfile", "Lightweight Python 3.13 image with minimal OS graphics libraries (libgl1, libglib2.0)."),
        ("Docker Compose Stack", "Orchestrates the FastAPI service and Prometheus scraper in an isolated network."),
        ("Automated Health Checks", "Configured HTTP container health monitoring at /api/health with auto-restart.")
    ], border_color=ACCENT_CYAN)

    add_card(slide9, 4.8, 1.6, 3.7, 5.2, "Continuous Integration", [
        ("GitHub Actions Pipeline", "Automated CI runs on every push and PR against main/master branches."),
        ("Automated Test Suite", "Executes full test matrix across API routes, ONNX inference, and video stream processors."),
        ("Zero Dependency Drift", "Explicit requirements lock ensuring reproducible builds on any environment.")
    ], border_color=ACCENT_GREEN)

    add_card(slide9, 8.9, 1.6, 3.6, 5.2, "100% Test Pass Rate", [
        ("33 Automated Tests", "Comprehensive pytest coverage across 4 dedicated test modules."),
        ("tests/test_api.py", "14 tests verifying all REST endpoints, form uploads, and JSON contracts."),
        ("tests/test_onnx_inference.py", "10 tests validating tensor math, NMS, and class metadata."),
        ("tests/test_video_stream.py", "4 tests checking directory loops, overlays, and camera failover.")
    ], border_color=ACCENT_PURPLE)

    # ==========================================
    # SLIDE 10: ERROR ANALYSIS & FUTURE ROADMAP
    # ==========================================
    slide10 = prs.slides.add_slide(blank_slide_layout)
    add_bg(slide10)
    add_header(slide10, "Continuous Improvement", "Error Analysis & Technology Roadmap", "Documented failure modes, mitigation strategies, and next-phase innovations")

    add_card(slide10, 0.8, 1.6, 5.6, 5.2, "Failure Mode Analysis & Mitigation", [
        ("Crazing Micro-Cracks", "Downsampling to 640x640 reduces gradient visibility. Mitigation: CLAHE contrast enhancement & lowered 0.23 threshold."),
        ("Rolled-in Scale Granularity", "Texture overlap with base steel grain. Mitigation: Albumentations RandomBrightnessContrast and ISONoise training pipeline."),
        ("Diagonal Scratches", "Fragmented bounding boxes on long diagonal abrasions. Mitigation: IoU NMS tuning (0.45) and rotation augmentations.")
    ], border_color=ACCENT_ORANGE)

    add_card(slide10, 6.9, 1.6, 5.6, 5.2, "Future Roadmap", [
        ("FP16 / INT8 TensorRT Quantization", "Targeting ~2.2ms latency for high-speed lines (>25 m/s) on NVIDIA Jetson Orin."),
        ("OPC-UA / MQTT Industrial Protocols", "Native integration with Siemens S7, Rockwell Allen-Bradley, and Beckhoff PLCs."),
        ("Multi-Camera Multi-View Sync", "Simultaneous top-and-bottom surface inspection using dual asynchronous ONNX streams."),
        ("Active Learning Feedback Loop", "One-click operator flagging of false-positives for automated retraining.")
    ], border_color=ACCENT_GREEN)

    # ==========================================
    # SLIDE 11: SUMMARY & CONCLUSION
    # ==========================================
    slide11 = prs.slides.add_slide(blank_slide_layout)
    add_bg(slide11)
    add_header(slide11, "Evaluation Summary", "Project Impact & Evaluation Takeaways", "High-performance edge solution ready for real-world industrial deployment")

    add_card(slide11, 0.8, 1.6, 5.6, 5.2, "Key Achievements", [
        ("Sub-30ms Edge Speed", "Real-time surface defect detection operating at 30+ FPS CPU / 180+ FPS GPU."),
        ("69.7% Global mAP50", "High accuracy across 6 complex industrial defect classes on the NEU-DET benchmark."),
        ("F1-Optimized Tuning", "Class-specific confidence thresholds maximizing safety-critical defect recall."),
        ("Zero-Downtime Multi-Source", "Seamless runtime switching between live webcam, validation loop, and video files."),
        ("Industry 4.0 Ready", "WebSocket PLC triggers, Prometheus metrics, and full Docker orchestration.")
    ], border_color=ACCENT_GREEN)

    add_card(slide11, 6.9, 1.6, 5.6, 5.2, "Evaluator Quick-Start", [
        ("1-Command Docker Deployment", "docker compose up --build (Spins up FastAPI at :8000 and Prometheus at :9090)."),
        ("Live Interactive Dashboard", "Open http://localhost:8000 to interact with the real-time detection UI."),
        ("Interactive OpenAPI Swagger", "Visit http://localhost:8000/docs to test all REST endpoints directly."),
        ("100% Passing Test Suite", "Run py -3.13 -m pytest tests/ -v for automated verification of all 33 tests.")
    ], border_color=ACCENT_CYAN)

    # Save presentation
    output_path = "Industrial_Defect_Detection_Presentation.pptx"
    prs.save(output_path)
    print(f"Presentation generated successfully: {output_path}")

if __name__ == "__main__":
    create_presentation()
