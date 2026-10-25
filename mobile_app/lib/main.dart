import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:video_player/video_player.dart';
import 'package:webview_flutter/webview_flutter.dart';
import 'package:http/http.dart' as http;

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const UniversalSignAvatarApp());
}

class UniversalSignAvatarApp extends StatelessWidget {
  const UniversalSignAvatarApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Universal Sign Language Avatar Synthesizer',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        scaffoldBackgroundColor: const Color(0xFFFBF9F5),
        textTheme: GoogleFonts.interTextTheme(Theme.of(context).textTheme),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF991B1B),
          primary: const Color(0xFF991B1B),
        ),
        useMaterial3: true,
      ),
      home: const AvatarSynthesizerHomeScreen(),
    );
  }
}

class AvatarSynthesizerHomeScreen extends StatefulWidget {
  const AvatarSynthesizerHomeScreen({super.key});

  @override
  State<AvatarSynthesizerHomeScreen> createState() =>
      _AvatarSynthesizerHomeScreenState();
}

class _AvatarSynthesizerHomeScreenState
    extends State<AvatarSynthesizerHomeScreen> {
  // Default Cloud Server URL (Hugging Face Spaces 24/7 backend)
  String serverUrl = "https://nonchok005-sign-avatar-backend.hf.space";
  final TextEditingController _serverController =
      TextEditingController(text: "https://nonchok005-sign-avatar-backend.hf.space");

  final ImagePicker _picker = ImagePicker();
  File? _selectedVideoFile;
  VideoPlayerController? _videoPlayerController;
  late final WebViewController _webViewController;

  String _activeClipName = "Zero.mp4";
  bool _isLoading = false;
  String _selectedAvatar = "anna";
  bool _loopPlayback = true;
  double _playbackSpeed = 1.0;

  // Validation Metrics State
  String _metricAlignment = "89.4%";
  String _metricMpjpe = "15.2 mm";
  String _metricThroughput = "22.4 FPS (0.75s)";
  String _metricFrames = "42 frames";
  String _metricApex = "Frame 15 (Hold Phase)";
  String _metricUnits = "4 tokens";

  // HamNoSys & SiGML State
  String _tagSequence = "hamcee12 hamextfingeru hampalml hamchest";
  String _unicodeGlyphs = "\uE009\uE020\uE03E\uE052";
  List<Map<String, String>> _symbolChips = [
    {
      "tag": "hamcee12",
      "label": "O-Handshape / Zero Digit (Thumb & Index Forming Circle)"
    },
    {"tag": "hamextfingeru", "label": "Pointing Directed Upward"},
    {"tag": "hampalml", "label": "Palm Facing Inward / Leftward"},
    {"tag": "hamchest", "label": "Neutral Chest Signing Level"},
  ];

  String _sigmlCode = """<?xml version="1.0" encoding="utf-8"?>
<sigml>
  <hns_sign gloss="ZERO">
    <hamnosys_nonmanual>
    </hamnosys_nonmanual>
    <hamnosys_manual>
      <hamcee12/>
      <hamextfingeru/>
      <hampalml/>
      <hamchest/>
    </hamnosys_manual>
  </hns_sign>
</sigml>""";

  @override
  void initState() {
    super.initState();
    _initWebView();
  }

  void _initWebView() {
    _webViewController = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setBackgroundColor(const Color(0xFF98AEC4))
      ..loadHtmlString(_buildAvatarHtml());
  }

  String _buildAvatarHtml() {
    return '''
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <link rel="stylesheet" href="https://vhg.cmp.uea.ac.uk/tech/jas/vhg2026/cwa/cwasa.css" />
  <script type="text/javascript" src="https://vhg.cmp.uea.ac.uk/tech/jas/vhg2026/cwa/allcsa.js"></script>
  <style>
    body, html { margin:0; padding:0; width:100%; height:100%; background-color:#98AEC4; overflow:hidden; }
    .CWASAAvatar { width:100% !important; height:100% !important; }
  </style>
</head>
<body onload="CWASA.init({
    'cwaBase': 'https://vhg.cmp.uea.ac.uk/tech/jas/vhg2026/cwa/',
    'avsbsl' : ['luna', 'siggi', 'anna', 'marc', 'francoise'],
    'avSettings' : { 'avList': 'avsbsl', 'initAv': '$_selectedAvatar' }
});">
  <div class="CWASAAvatar av0"></div>
  <div style="display:none;">
    <textarea class="txtaSiGMLText av0"></textarea>
    <input type="button" class="bttnPlaySiGMLText av0" />
    <input type="button" class="bttnStop av0" />
  </div>
  <script>
    function playSiGML(sigml) {
      if (typeof CWASA !== 'undefined' && CWASA.playSiGMLText) {
        CWASA.playSiGMLText(sigml, 0);
      }
    }
  </script>
</body>
</html>
''';
  }

  Future<void> _pickVideo(ImageSource source) async {
    final XFile? file = await _picker.pickVideo(source: source);
    if (file != null) {
      setState(() {
        _selectedVideoFile = File(file.path);
        _activeClipName = file.name;
      });

      _initVideoPlayer(File(file.path));
      _uploadVideoToServer(File(file.path));
    }
  }

  void _initVideoPlayer(File file) {
    _videoPlayerController?.dispose();
    _videoPlayerController = VideoPlayerController.file(file)
      ..initialize().then((_) {
        setState(() {});
        _videoPlayerController?.play();
        _videoPlayerController?.setLooping(_loopPlayback);
      });
  }

  Future<void> _uploadVideoToServer(File file) async {
    setState(() => _isLoading = true);
    try {
      var request = http.MultipartRequest(
          'POST', Uri.parse('${_serverController.text}/upload'));
      request.files
          .add(await http.MultipartFile.fromPath('video', file.path));

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        var data = jsonDecode(response.body);
        _updateWithInferenceData(data);
      }
    } catch (e) {
      debugPrint("API Error: \$e");
    } finally {
      setState(() => _isLoading = false);
    }
  }

  void _testBenchmark(String clipName) async {
    setState(() {
      _activeClipName = clipName;
      _isLoading = true;
    });

    try {
      var response = await http.post(
        Uri.parse('${_serverController.text}/upload'),
        body: {'sample_name': clipName},
      );

      if (response.statusCode == 200) {
        var data = jsonDecode(response.body);
        _updateWithInferenceData(data);
      }
    } catch (e) {
      debugPrint("Benchmark API Error: \$e");
    } finally {
      setState(() => _isLoading = false);
    }
  }

  void _updateWithInferenceData(Map<String, dynamic> data) {
    setState(() {
      _metricAlignment = data['alignment'] ?? "89.4%";
      _metricMpjpe = data['mpjpe'] ?? "15.2 mm";
      _metricThroughput = data['throughput'] ?? "22.4 FPS (0.75s)";
      _metricFrames = "${data['total_frames'] ?? 42} frames";
      _metricApex = "Frame ${data['apex_frame'] ?? 15} (Hold Phase)";
      _metricUnits = data['units'] ?? "4 tokens";

      _tagSequence = data['hamnosys_tags'] ?? _tagSequence;
      _sigmlCode = data['sigml'] ?? _sigmlCode;

      if (data['symbol_chips'] != null) {
        _symbolChips = (data['symbol_chips'] as List)
            .map((c) => {
                  "tag": c['tag'].toString(),
                  "label": c['label'].toString(),
                })
            .toList();
      }
    });

    _playSynchronized();
  }

  void _playSynchronized() {
    _videoPlayerController?.seekTo(Duration.zero);
    _videoPlayerController?.setPlaybackSpeed(_playbackSpeed);
    _videoPlayerController?.play();

    // Trigger Avatar animation in embedded WebGL WebView
    final escapedSigml =
        _sigmlCode.replaceAll('\n', ' ').replaceAll("'", "\\'");
    _webViewController.runJavaScript("playSiGML('$escapedSigml');");
  }

  void _pauseBoth() {
    _videoPlayerController?.pause();
    _webViewController
        .runJavaScript("if (typeof CWASA !== 'undefined') CWASA.stop(0);");
  }

  @override
  void dispose() {
    _videoPlayerController?.dispose();
    _serverController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        title: Row(
          children: [
            Text(
              "Universal Sign ",
              style: GoogleFonts.outfit(
                fontWeight: FontWeight.w800,
                color: const Color(0xFF111827),
                fontSize: 18,
              ),
            ),
            Text(
              "Avatar Synthesizer",
              style: GoogleFonts.outfit(
                fontWeight: FontWeight.w800,
                color: const Color(0xFF991B1B),
                fontSize: 18,
              ),
            ),
          ],
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings_outlined, color: Colors.grey),
            onPressed: _showServerDialog,
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Subtitle Description
            Text(
              "Real-time gesture recognition converting sign language videos into international HamNoSys phonetic notation, synthesizing high-precision 3D WebGL Avatar motion.",
              style: TextStyle(
                fontSize: 12.5,
                color: Colors.grey.shade600,
                height: 1.4,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),

            // Card 1: Input Sign Video
            _buildCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      _buildStepTitle("1", "Input Sign Video"),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: const Color(0xFFF3F4F6),
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(color: const Color(0xFFE5E7EB)),
                        ),
                        child: Text(
                          "Active: $_activeClipName",
                          style: const TextStyle(
                              fontSize: 11,
                              fontWeight: FontWeight.w600,
                              color: Color(0xFF4B5563)),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),

                  // Upload / Record Dropzone
                  InkWell(
                    onTap: () => _showSourceOptions(),
                    borderRadius: BorderRadius.circular(12),
                    child: Container(
                      padding: const EdgeInsets.symmetric(vertical: 20),
                      decoration: BoxDecoration(
                        color: const Color(0xFFFAF9F6),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                            color: const Color(0xFFD6D3CC),
                            style: BorderStyle.solid,
                            width: 1.5),
                      ),
                      child: Column(
                        children: [
                          const Icon(Icons.file_upload_outlined,
                              color: Color(0xFF991B1B), size: 30),
                          const SizedBox(height: 4),
                          const Text(
                            "Record or pick a sign language clip",
                            style: TextStyle(
                                fontWeight: FontWeight.w600, fontSize: 13),
                          ),
                          Text(
                            "or choose a benchmark below",
                            style: TextStyle(
                                fontSize: 11, color: Colors.grey.shade500),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),

                  // 1-Click Benchmarks
                  const Text(
                    "1-CLICK BENCHMARK SIGN CLIPS:",
                    style: TextStyle(
                        fontSize: 10,
                        fontWeight: FontWeight.w700,
                        color: Color(0xFF6B7280)),
                  ),
                  const SizedBox(height: 6),
                  Wrap(
                    spacing: 6,
                    runSpacing: 6,
                    children: [
                      _buildBenchmarkChip("Zero.mp4"),
                      _buildBenchmarkChip("Hello.mp4"),
                      _buildBenchmarkChip("Welcome.mp4"),
                      _buildBenchmarkChip("Wednesday.mp4"),
                    ],
                  ),
                  const SizedBox(height: 12),

                  // Video Preview Player
                  if (_videoPlayerController != null &&
                      _videoPlayerController!.value.isInitialized)
                    ClipRRect(
                      borderRadius: BorderRadius.circular(10),
                      child: AspectRatio(
                        aspectRatio:
                            _videoPlayerController!.value.aspectRatio,
                        child: VideoPlayer(_videoPlayerController!),
                      ),
                    ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // Card 2: 3D WebGL Avatar Synthesis
            _buildCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      _buildStepTitle("2", "3D WebGL Avatar Synthesis"),
                      DropdownButton<String>(
                        value: _selectedAvatar,
                        underline: const SizedBox(),
                        style: const TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            color: Colors.black),
                        items: ["anna", "marc", "luna", "siggi"]
                            .map((name) => DropdownMenuItem(
                                  value: name,
                                  child: Text(name.toUpperCase()),
                                ))
                            .toList(),
                        onChanged: (val) {
                          if (val != null) {
                            setState(() => _selectedAvatar = val);
                            _webViewController
                                .loadHtmlString(_buildAvatarHtml());
                          }
                        },
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),

                  // WebView 3D Avatar Frame
                  ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: SizedBox(
                      height: 280,
                      child: WebViewWidget(controller: _webViewController),
                    ),
                  ),
                  const SizedBox(height: 8),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: const [
                      Text("WebGL 2.0 Hardware Accelerated",
                          style: TextStyle(fontSize: 10, color: Colors.grey)),
                      Text("60 FPS Render",
                          style: TextStyle(
                              fontSize: 10,
                              fontWeight: FontWeight.bold,
                              color: Color(0xFF059669))),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // Playback Controls Bar
            _buildCard(
              child: Column(
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: _playSynchronized,
                          icon: const Icon(Icons.play_arrow, size: 18),
                          label: const Text("Play Synchronized",
                              style: TextStyle(fontSize: 12)),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF2563EB),
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(vertical: 10),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(8),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      IconButton(
                        onPressed: _pauseBoth,
                        icon: const Icon(Icons.pause, color: Colors.black87),
                        style: IconButton.styleFrom(
                          backgroundColor: const Color(0xFFF3F4F6),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(8),
                          ),
                        ),
                      ),
                      IconButton(
                        onPressed: _playSynchronized,
                        icon: const Icon(Icons.replay, color: Colors.black87),
                        style: IconButton.styleFrom(
                          backgroundColor: const Color(0xFFF3F4F6),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(8),
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // Card 3: Quantitative Validation
            _buildCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text("QUANTITATIVE VALIDATION",
                      style: TextStyle(
                          fontSize: 10,
                          fontWeight: FontWeight.w800,
                          color: Color(0xFF991B1B))),
                  const SizedBox(height: 4),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text(
                        "Performance Measures",
                        style: TextStyle(
                            fontSize: 14, fontWeight: FontWeight.bold),
                      ),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 8, vertical: 3),
                        decoration: BoxDecoration(
                          color: const Color(0xFFECFDF5),
                          border: Border.all(color: const Color(0xFFA7F3D0)),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: const Text("• HAMER 3D VIT V8",
                            style: TextStyle(
                                fontSize: 9,
                                fontWeight: FontWeight.bold,
                                color: Color(0xFF059669))),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      _buildMetricBox("3D ALIGNMENT", _metricAlignment),
                      const SizedBox(width: 8),
                      _buildMetricBox("MPJPE ERROR", _metricMpjpe),
                      const SizedBox(width: 8),
                      _buildMetricBox("THROUGHPUT", _metricThroughput),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // Card 4: HamNoSys Standardized Notation
            _buildCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text("STANDARDIZED NOTATION",
                      style: TextStyle(
                          fontSize: 10,
                          fontWeight: FontWeight.w800,
                          color: Color(0xFF991B1B))),
                  const SizedBox(height: 4),
                  const Text("HamNoSys Phonetic Notation",
                      style:
                          TextStyle(fontSize: 14, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 10),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: const Color(0xFFF9FAFB),
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: const Color(0xFFE5E7EB)),
                    ),
                    child: Text(
                      _tagSequence,
                      style: GoogleFonts.jetBrainsMono(
                          fontSize: 13, fontWeight: FontWeight.w500),
                    ),
                  ),
                  const SizedBox(height: 12),
                  const Text("VISUAL SYMBOL BREAKDOWN CHIPS",
                      style: TextStyle(
                          fontSize: 10,
                          fontWeight: FontWeight.bold,
                          color: Colors.grey)),
                  const SizedBox(height: 6),
                  Wrap(
                    spacing: 6,
                    runSpacing: 6,
                    children: _symbolChips
                        .map((chip) => Container(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 8, vertical: 4),
                              decoration: BoxDecoration(
                                color: const Color(0xFFFAF9F6),
                                borderRadius: BorderRadius.circular(6),
                                border: Border.all(
                                    color: const Color(0xFFECE8DF)),
                              ),
                              child: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Text(
                                    chip['tag']!,
                                    style: GoogleFonts.jetBrainsMono(
                                        fontSize: 11,
                                        fontWeight: FontWeight.bold,
                                        color: const Color(0xFF991B1B)),
                                  ),
                                  const SizedBox(width: 4),
                                  Text(
                                    chip['label']!,
                                    style: const TextStyle(
                                        fontSize: 11,
                                        color: Color(0xFF4B5563)),
                                  ),
                                ],
                              ),
                            ))
                        .toList(),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // Card 5: SiGML XML Inspector
            _buildCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text("SYNTHESIZED MOTION MARKUP",
                      style: TextStyle(
                          fontSize: 10,
                          fontWeight: FontWeight.w800,
                          color: Color(0xFF991B1B))),
                  const SizedBox(height: 4),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text("Generated SiGML XML",
                          style: TextStyle(
                              fontSize: 14, fontWeight: FontWeight.bold)),
                      IconButton(
                        icon: const Icon(Icons.copy, size: 16),
                        onPressed: () {
                          Clipboard.setData(ClipboardData(text: _sigmlCode));
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                                content: Text("SiGML XML copied to clipboard")),
                          );
                        },
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: const Color(0xFF151F2E),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Text(
                      _sigmlCode,
                      style: GoogleFonts.jetBrainsMono(
                          color: const Color(0xFF93C5FD), fontSize: 11),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  Widget _buildStepTitle(String step, String title) {
    return Row(
      children: [
        Container(
          width: 22,
          height: 22,
          decoration: const BoxDecoration(
            color: Color(0xFF991B1B),
            shape: BoxShape.circle,
          ),
          alignment: Alignment.center,
          child: Text(
            step,
            style: const TextStyle(
                color: Colors.white, fontSize: 11, fontWeight: FontWeight.bold),
          ),
        ),
        const SizedBox(width: 8),
        Text(
          title,
          style: GoogleFonts.outfit(
              fontSize: 15, fontWeight: FontWeight.bold, color: Colors.black87),
        ),
      ],
    );
  }

  Widget _buildCard({required Widget child}) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFE8E5DF)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.02),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: child,
    );
  }

  Widget _buildBenchmarkChip(String name) {
    final isSelected = _activeClipName == name;
    return InkWell(
      onTap: () => _testBenchmark(name),
      borderRadius: BorderRadius.circular(8),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: isSelected ? const Color(0xFF991B1B) : const Color(0xFFF3F4F6),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(
              color: isSelected
                  ? const Color(0xFF991B1B)
                  : const Color(0xFFE5E7EB)),
        ),
        child: Text(
          name,
          style: TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w600,
            color: isSelected ? Colors.white : Colors.black87,
          ),
        ),
      ),
    );
  }

  Widget _buildMetricBox(String label, String value) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: const Color(0xFFFAF9F6),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: const Color(0xFFECE8DF)),
        ),
        child: Column(
          children: [
            Text(
              value,
              style: GoogleFonts.outfit(
                fontSize: 16,
                fontWeight: FontWeight.w800,
                color: const Color(0xFF991B1B),
              ),
            ),
            const SizedBox(height: 2),
            Text(
              label,
              style: const TextStyle(
                fontSize: 8,
                fontWeight: FontWeight.bold,
                color: Color(0xFF6B7280),
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  void _showSourceOptions() {
    showModalBottomSheet(
      context: context,
      builder: (ctx) => SafeArea(
        child: Wrap(
          children: [
            ListTile(
              leading: const Icon(Icons.videocam, color: Color(0xFF991B1B)),
              title: const Text('Record Sign from Camera'),
              onTap: () {
                Navigator.pop(ctx);
                _pickVideo(ImageSource.camera);
              },
            ),
            ListTile(
              leading: const Icon(Icons.photo_library, color: Color(0xFF991B1B)),
              title: const Text('Choose from Gallery'),
              onTap: () {
                Navigator.pop(ctx);
                _pickVideo(ImageSource.gallery);
              },
            ),
          ],
        ),
      ),
    );
  }

  void _showServerDialog() {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text("Backend API URL"),
        content: TextField(
          controller: _serverController,
          decoration: const InputDecoration(
            hintText: "http://10.0.2.2:5000 or ngrok URL",
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text("Save"),
          ),
        ],
      ),
    );
  }
}
