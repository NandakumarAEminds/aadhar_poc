import 'dart:io';
import 'dart:math';
import 'dart:ui';

import 'package:camera/camera.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;

import '../ML/Recognition.dart';
import '../ML/Recognizer.dart';
import '../Util.dart';
import '../main.dart';
import 'HomeScreen.dart';

class RegistrationScreen extends StatefulWidget {
  const RegistrationScreen({super.key});

  @override
  State<RegistrationScreen> createState() => _RecognitionScreenState();
}

class _RecognitionScreenState extends State<RegistrationScreen> {
  dynamic controller;
  bool isBusy = false;
  late Size size;

  late CameraDescription description = cameras[1];
  CameraLensDirection camDirec = CameraLensDirection.front;
  late List<Recognition> recognitions = [];
  dynamic _scanResults;
  CameraImage? frame;
  late img.Image croppedFace;
  img.Image? image;

  // Face detector instance
  late FaceDetector faceDetector;

  // Face recognizer instance
  late Recognizer recognizer;

  // Step counter for face angles
  int _currentStep = 0;

  // Expected face angles
  List<String> faceAngles = ["Front", "Right", "Left", "Down", "Up"];

  // Icons for each face angle
  final Map<String, IconData> angleIcons = {
    'Front': Icons.front_hand,
    'Left': Icons.arrow_back,
    'Right': Icons.arrow_forward,
    'Up': Icons.arrow_upward,
    'Down': Icons.arrow_downward,
  };

  bool getEmb = false;
  img.Image? frontFace;

  // Device orientation mapping
  final _orientations = {
    DeviceOrientation.portraitUp: 0,
    DeviceOrientation.landscapeLeft: 90,
    DeviceOrientation.portraitDown: 180,
    DeviceOrientation.landscapeRight: 270,
  };

  // Store face embeddings from different positions
  List<List<double>> embeddings = [];

  int registrationStep = 0;

  // Instructions for each registration step
  List<String> positionInstructions = [
    "Look straight ahead",
    "Turn left",
    "Turn right",
    "Look up",
    "Look down",
  ];

  bool dialogShown = false;
  bool register = false;

  @override
  void initState() {
    super.initState();

    // Initialize face detector
    var options = FaceDetectorOptions(performanceMode: FaceDetectorMode.accurate);
    faceDetector = FaceDetector(options: options);

    // Initialize face recognizer
    recognizer = Recognizer();

    // Initialize camera stream
    initializeCamera();
  }

  // Initialize camera controller and start image stream
  initializeCamera() async {
    controller = CameraController(
      description,
      ResolutionPreset.medium,
      imageFormatGroup: Platform.isAndroid ? ImageFormatGroup.nv21 : ImageFormatGroup.bgra8888,
      enableAudio: false,
    );

    await controller.initialize().then((_) {
      if (!mounted) return;
      setState(() {});
      controller.startImageStream((image) {
        if (!isBusy) {
          isBusy = true;
          frame = image;
          doFaceDetectionOnFrame();
        }
      });
    });
  }

  @override
  void dispose() {
    controller?.dispose();
    super.dispose();
  }

  int _validFrameCount = 0;
  final int _requiredValidFrames = 3;

  // Detect face on current frame
  void doFaceDetectionOnFrame() async {
    InputImage? inputImage = getInputImage();
    if (inputImage == null) return;

    List<Face> faces = await faceDetector.processImage(inputImage);

    if (faces.isNotEmpty && _isFaceProperlyAligned(faces[0], _currentStep)) {
      final tempImage = Platform.isIOS ? Util.convertBGRA8888ToImage(frame!) : Util.convertNV21(frame!);
      final rotated = img.copyRotate(tempImage, angle: camDirec == CameraLensDirection.front ? 270 : 90);
      final cropped = img.copyCrop(
        rotated,
        x: faces[0].boundingBox.left.toInt(),
        y: faces[0].boundingBox.top.toInt(),
        width: faces[0].boundingBox.width.toInt(),
        height: faces[0].boundingBox.height.toInt(),
      );

      if (_isFaceSharp(cropped)) {
        _validFrameCount++;
        if (_validFrameCount >= _requiredValidFrames && !getEmb) {
          getEmb = true;
          _validFrameCount = 0;
          performFaceRecognition(faces[0], cropped);
        }
      } else {
        _validFrameCount = 0;
      }
    } else {
      _validFrameCount = 0;
    }

    isBusy = false;
  }

  // Check if face is sharp enough using variance of Laplacian
  bool _isFaceSharp(img.Image faceImage) {
    final grayscale = img.grayscale(faceImage);
    final laplacian = img.sobel(grayscale);
    final pixels = laplacian.getBytes();

    double mean = pixels.reduce((a, b) => a + b) / pixels.length;
    double variance = pixels.map((p) => pow(p - mean, 2)).reduce((a, b) => a + b) / pixels.length;

    return variance > 1500;
  }

  // Check if face is aligned according to the current step
  bool _isFaceProperlyAligned(Face face, int step) {
    switch (step) {
      case 0:
        return face.headEulerAngleY!.abs() < 10 && face.headEulerAngleX!.abs() < 10 && face.boundingBox.width > 80;
      case 1:
        return face.headEulerAngleY! < -20;
      case 2:
        return face.headEulerAngleY! > 20;
      case 3:
        return face.headEulerAngleX! < -15;
      case 4:
        return face.headEulerAngleX! > 10;
      default:
        return false;
    }
  }

  // Convert CameraImage to MLKit's InputImage
  InputImage? getInputImage() {
    final camera = camDirec == CameraLensDirection.front ? cameras[1] : cameras[0];
    final sensorOrientation = camera.sensorOrientation;

    InputImageRotation? rotation;
    if (Platform.isIOS) {
      rotation = InputImageRotationValue.fromRawValue(sensorOrientation);
    } else if (Platform.isAndroid) {
      var rotationCompensation = _orientations[controller!.value.deviceOrientation];
      if (rotationCompensation == null) return null;

      rotationCompensation = camera.lensDirection == CameraLensDirection.front
          ? (sensorOrientation + rotationCompensation) % 360
          : (sensorOrientation - rotationCompensation + 360) % 360;

      rotation = InputImageRotationValue.fromRawValue(rotationCompensation);
    }

    if (rotation == null) return null;

    final format = InputImageFormatValue.fromRawValue(frame!.format.raw);
    if (format == null ||
        (Platform.isAndroid && format != InputImageFormat.nv21) ||
        (Platform.isIOS && format != InputImageFormat.bgra8888)) {
      return null;
    }

    if (frame!.planes.length != 1) return null;
    final plane = frame!.planes.first;

    return InputImage.fromBytes(
      bytes: plane.bytes,
      metadata: InputImageMetadata(
        size: Size(frame!.width.toDouble(), frame!.height.toDouble()),
        rotation: rotation,
        format: format,
        bytesPerRow: plane.bytesPerRow,
      ),
    );
  }

  // Start face registration by resetting state
  void startFaceRegistration() {
    embeddings.clear();
    registrationStep = 0;
    promptForNextPosition();
  }

  // Prompt user for next face angle
  void promptForNextPosition() {
    if (registrationStep >= positionInstructions.length) {
      completeRegistration();
      return;
    }
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(positionInstructions[registrationStep])),
    );
  }

  // Save embedding for current step
  void captureEmbedding(Recognition recognition) {
    embeddings.add(recognition.embeddings);
    registrationStep++;

    if (registrationStep < positionInstructions.length) {
      promptForNextPosition();
    } else {
      completeRegistration();
    }
  }

  // Finalize and save registration
  void completeRegistration() {
    recognizer.registerFaceInDB(
      textEditingController.text,
      embeddings,
      Uint8List.fromList(img.encodeBmp(frontFace!)),
    );

    textEditingController.text = "";
    dialogShown = false;
    Navigator.pop(context);

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text("Face Registered with multiple angles")),
    );

    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (context) => const HomeScreen()),
    );
  }

  // Perform face recognition and advance step
  void performFaceRecognition(Face face, img.Image croppedFace) async {
    print("in perform face recognition");
    recognitions.clear();

    image = Platform.isIOS ? Util.convertBGRA8888ToImage(frame!) : Util.convertNV21(frame!);
    image = img.copyRotate(image!, angle: camDirec == CameraLensDirection.front ? 270 : 90);

    frontFace ??= croppedFace;

    Recognition recognition = recognizer.recognize(croppedFace, face.boundingBox);
    embeddings.add(recognition.embeddings);

    if (!mounted) return;

    setState(() {
      if (_currentStep < faceAngles.length - 1) {
        _currentStep++;
      } else {
        if (!dialogShown) {
          showFaceRegistrationDialogue(frontFace!);
        }
      }
      isBusy = false;
      getEmb = false;
      _scanResults = recognitions;
      print("in perform face current step=" + _currentStep.toString());
    });
  }

  // Show face registration dialog for entering name
  TextEditingController textEditingController = TextEditingController();
  void showFaceRegistrationDialogue(img.Image croppedFace) {
    dialogShown = true;
    textEditingController.clear();
    showDialog(
      context: context,
      barrierDismissible: false,
      builder:
          (ctx) => Dialog(
        backgroundColor: Colors.transparent,
        insetPadding: const EdgeInsets.symmetric(
          horizontal: 20,
          vertical: 60,
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(20),
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
            child: Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.1),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: Colors.white.withOpacity(0.2)),
              ),
              child: SingleChildScrollView(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Text(
                      "Register Your Face",
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 20,
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 20),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(100),
                      child: Image.memory(
                        Uint8List.fromList(img.encodeBmp(croppedFace)),
                        width: 150,
                        height: 150,
                        fit: BoxFit.cover,
                      ),
                    ),
                    const SizedBox(height: 20),
                    TextField(
                      controller: textEditingController,
                      style: const TextStyle(color: Colors.white),
                      decoration: InputDecoration(
                        hintText: "Enter your name",
                        hintStyle: const TextStyle(color: Colors.white70),
                        filled: true,
                        fillColor: Colors.white.withAlpha(80),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide: BorderSide.none,
                        ),
                      ),
                    ),
                    const SizedBox(height: 20),
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton.icon(
                        onPressed: () {
                          recognizer.registerFaceInDB(
                            textEditingController.text.trim(),
                            embeddings,
                            Uint8List.fromList(img.encodeBmp(frontFace!)),
                          );
                          Navigator.pop(context);
                          Navigator.pop(context); // Close dialog
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                              content: Text("Face Registered"),
                            ),
                          );
                        },
                        icon: const Icon(Icons.check),
                        label: const Text("Register"),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.deepPurple.shade300,
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    List<Widget> stackChildren = [];
    size = MediaQuery.of(context).size;
    if (controller != null) {
      //TODO View for displaying the live camera footage
      stackChildren.add(
        Positioned(
          top: 0.0,
          left: 0.0,
          width: size.width,
          height: size.height,
          child: Container(
            child:
            (controller.value.isInitialized)
                ? AspectRatio(
              aspectRatio: controller.value.aspectRatio,
              child: CameraPreview(controller),
            )
                : Container(),
          ),
        ),
      );


    }

    //TODO View for displaying the bar to switch camera direction or for registering faces
    stackChildren.add(
      Positioned(
        top: 30,
        left: 20,
        right: 20,
        child: ClipRRect(
          borderRadius: BorderRadius.circular(10),
          child: LinearProgressIndicator(
            value: _currentStep / faceAngles.length.toDouble(),
            backgroundColor: Colors.white.withAlpha(80),
            valueColor: AlwaysStoppedAnimation<Color>(
              Colors.deepPurple.shade300,
            ),
            minHeight: 10,
          ),
        ),
      ),
    );
    stackChildren.add(
      Positioned(
        bottom: 40,
        left: 20,
        right: 20,
        child: ClipRRect(
          borderRadius: BorderRadius.circular(20),
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white.withAlpha(30),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: Colors.white.withOpacity(0.2)),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  AnimatedSwitcher(
                    duration: const Duration(milliseconds: 300),
                    child: Icon(
                      angleIcons[faceAngles[_currentStep]],
                      key: ValueKey(_currentStep),
                      color: Colors.white,
                      size: 40,
                    ),
                  ),
                  const SizedBox(height: 10),
                  AnimatedSwitcher(
                    duration: const Duration(milliseconds: 300),
                    child: Text(
                      "Position your face: ${faceAngles[_currentStep]}",
                      key: ValueKey(_currentStep),
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );

    return SafeArea(
      child: Scaffold(
        resizeToAvoidBottomInset: false,
        backgroundColor: Colors.black,
        body: Container(
          margin: const EdgeInsets.only(top: 0),
          color: Colors.black,
          child: Stack(children: stackChildren),
        ),
      ),
    );
  }
}



