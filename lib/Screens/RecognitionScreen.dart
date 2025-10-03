import 'package:camera/camera.dart';
import 'package:flutter/services.dart';

import 'dart:io';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;


import '../ML/Recognition.dart';
import '../ML/Recognizer.dart';
import '../Util.dart';
import '../main.dart';


class RecognitionScreen extends StatefulWidget {
  const RecognitionScreen({super.key});

  @override
  State<RecognitionScreen> createState() => _RecognitionScreenState();
}

class _RecognitionScreenState extends State<RecognitionScreen> {
  dynamic controller;
  bool isBusy = false;
  late Size size;
  late CameraDescription description = cameras[1];
  CameraLensDirection camDirec = CameraLensDirection.front;
  late List<Recognition> recognitions = [];

  //TODO declare face detector
  late FaceDetector faceDetector;

  //TODO declare face recognizer
  late Recognizer recognizer;
  String? currentChallenge;
  bool challengePassed = false;
  DateTime lastChallengeTime = DateTime.now();
  Duration challengeInterval = Duration(seconds: 6);
  bool first = true;
  Rect? previousFaceRect;
  DateTime? previousFaceTime;

  @override
  void initState() {
    super.initState();

    //TODO initialize face detector
    var options = FaceDetectorOptions(performanceMode: FaceDetectorMode.accurate,enableClassification: true);
    faceDetector = FaceDetector(options: options);
    //TODO initialize face recognizer
    recognizer = Recognizer();
    //TODO initialize camera footage
    initializeCamera();
  }

  //TODO code to initialize the camera feed
  initializeCamera() async {
    controller = CameraController(description, ResolutionPreset.medium,imageFormatGroup: Platform.isAndroid
        ? ImageFormatGroup.nv21 // for Android
        : ImageFormatGroup.bgra8888,enableAudio: false); // for iOS);
    await controller.initialize().then((_) {
      if (!mounted) {
        return;
      }
      setState(() {
        controller;
      });
      controller.startImageStream((image) => {
        if (!isBusy) {isBusy = true, frame = image, doFaceDetectionOnFrame()}
      });
    });
  }

  //TODO close all resources
  @override
  void dispose() {
    controller?.dispose();
    super.dispose();
  }

  //TODO face detection on a frame
  dynamic _scanResults;
  CameraImage? frame;
  doFaceDetectionOnFrame() async {
    //TODO convert frame into InputImage format

    InputImage? inputImage = getInputImage();
    //TODO pass InputImage to face detection model and detect faces

    List<Face> faces = await faceDetector.processImage(inputImage!);


    //TODO perform face recognition on detected faces
    performFaceRecognition(faces);
    // setState(() {
    //   _scanResults = faces;
    //   isBusy = false;
    // });
  }

  img.Image? image;
  bool register = false;

  // TODO perform Face Recognition
  performFaceRecognition(List<Face> faces) async {
    recognitions.clear();

    //TODO convert CameraImage to Image and rotate it so that our frame will be in a portrait
    image = Platform.isIOS?Util.convertBGRA8888ToImage(frame!) as img.Image?:Util.convertNV21(frame!);
    image =img.copyRotate(image!, angle: camDirec == CameraLensDirection.front?270:90);


    for (Face face in faces) {
      double? left = face.leftEyeOpenProbability;
      double? right = face.rightEyeOpenProbability;

      Rect faceRect = face.boundingBox;
      img.Image croppedFace = img.copyCrop(
        image!,
        x: faceRect.left.toInt(),
        y: faceRect.top.toInt(),
        width: faceRect.width.toInt(),
        height: faceRect.height.toInt(),
      );

      Recognition recognition = recognizer.recognize(croppedFace, faceRect);
      recognitions.add(recognition);
    }

    if (!mounted) return;
    setState(() {
      isBusy  = false;
      _scanResults = recognitions;
    });

  }



  final _orientations = {
    DeviceOrientation.portraitUp: 0,
    DeviceOrientation.landscapeLeft: 90,
    DeviceOrientation.portraitDown: 180,
    DeviceOrientation.landscapeRight: 270,
  };

  //TODO convert CameraImage to InputImage
  InputImage? getInputImage() {
    final camera =
    camDirec == CameraLensDirection.front ? cameras[1] : cameras[0];
    final sensorOrientation = camera.sensorOrientation;

    InputImageRotation? rotation;
    if (Platform.isIOS) {
      rotation = InputImageRotationValue.fromRawValue(sensorOrientation);
    } else if (Platform.isAndroid) {
      var rotationCompensation =
      _orientations[controller!.value.deviceOrientation];
      if (rotationCompensation == null) return null;
      if (camera.lensDirection == CameraLensDirection.front) {
        // front-facing
        rotationCompensation = (sensorOrientation + rotationCompensation) % 360;
      } else {
        // back-facing
        rotationCompensation =
            (sensorOrientation - rotationCompensation + 360) % 360;
      }
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


  Widget buildResult() {
    if (_scanResults == null ||
        controller == null ||
        !controller.value.isInitialized) {
      return const Center(child: Text(''));
    }
    final Size imageSize = Size(
      controller.value.previewSize!.height,
      controller.value.previewSize!.width,
    );
    CustomPainter painter = FaceDetectorPainter(imageSize, _scanResults, camDirec);
    return CustomPaint(
      painter: painter,
    );
  }

  //TODO toggle camera direction
  void _toggleCameraDirection() async {
    if (camDirec == CameraLensDirection.back) {
      camDirec = CameraLensDirection.front;
      description = cameras[1];
    } else {
      camDirec = CameraLensDirection.back;
      description = cameras[0];
    }
    await controller.stopImageStream();
    setState(() {
      controller;
    });

    initializeCamera();
  }

  @override
  Widget build(BuildContext context) {
    size = MediaQuery.of(context).size;
    List<Widget> stackChildren = [];

    if (controller != null && controller.value.isInitialized) {
      // Camera Preview
      stackChildren.add(Positioned.fill(
        child: AspectRatio(
          aspectRatio: controller.value.aspectRatio,
          child: CameraPreview(controller),
        ),
      ));

      // Detected face overlays
      stackChildren.add(Positioned.fill(child: buildResult()));
    }


    stackChildren.add(
      Positioned(
        bottom: 30,
        left: 20,
        right: 20,
        child: ClipRRect(
          borderRadius: BorderRadius.circular(20),
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 15, sigmaY: 15),
            child: Container(
              height: 60,
              decoration: BoxDecoration(
                color: const Color.fromRGBO(255, 255, 255, 0.08),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: Colors.white.withAlpha(50)),
              ),
              child: Center(
                child: IconButton(
                  icon: const Icon(Icons.cameraswitch_rounded, color: Colors.white),
                  iconSize: 34,
                  tooltip: 'Switch Camera',
                  onPressed: _toggleCameraDirection,
                ),
              ),
            ),
          ),
        ),
      ),
    );

    return SafeArea(
      child: Scaffold(
        backgroundColor: Colors.black,
        body: Stack(children: stackChildren),
      ),
    );
  }

}