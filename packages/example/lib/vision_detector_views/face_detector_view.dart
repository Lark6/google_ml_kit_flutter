import 'dart:io';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_ml_kit_example/img_util.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'detector_view.dart';
import 'painters/face_detector_painter.dart';

class FaceDetectorView extends StatefulWidget {
  @override
  State<FaceDetectorView> createState() => _FaceDetectorViewState();
}

class _FaceDetectorViewState extends State<FaceDetectorView> {
  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      enableContours: true,
      enableLandmarks: true,
    ),
  );

  late Interpreter _interpreter;
  late IsolateInterpreter _isolateInterpreter;
  bool _canProcess = true;
  bool _isBusy = false;
  CustomPaint? _customPaint;
  String? _text;
  int flag = 0;
  var _cameraLensDirection = CameraLensDirection.front;

  int _frameCount = 0; // 프레임 카운터
  int _startTime = DateTime.now().millisecondsSinceEpoch; // FPS 측정 시작 시간
  int _currentFPS = 0; // 1초 동안의 FPS


  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/MobileNetV2(200).tflite',
        options: InterpreterOptions()..threads = 4,
        );
      _isolateInterpreter = await IsolateInterpreter.create(address: _interpreter.address);
      print('Model loaded successfully');
    } catch (e) {
      print('Error loading model: $e');
    }
  }

  @override
  void dispose() {
    _canProcess = false;
    _faceDetector.close();
    _interpreter.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        DetectorView(
          title: 'Face Detector',
          customPaint: _customPaint,
          text: _text,
          onImage: _processImage,
          initialCameraLensDirection: _cameraLensDirection,
          onCameraLensDirectionChanged: (value) => _cameraLensDirection = value,
        ),
        Center(
          child: Text(_text ?? 'no data', style: TextStyle(color: flag == 0? Colors.white : Colors.greenAccent, fontSize: 32)),
        ),
        Positioned(
        top: 20,
        right: 20,
        child: Text(
          'FPS: 7', // FPS 표시
          style: TextStyle(
            color: Colors.yellow,
            fontSize: 18,
            fontWeight: FontWeight.bold,
          ),
        ),)
      ],
    );
  }

  Future<void> _processImage(InputImage inputImage) async {
    if (!_canProcess || _isBusy) return;
    _isBusy = true;


    final faces = await _faceDetector.processImage(inputImage);
    if (faces.isEmpty) {
      _isBusy = false;
      return;
    }

    final squareRect = _getSquareRect(faces, inputImage.metadata!.size);

    final image = Platform.isAndroid ? decodeYUV420SP(inputImage): decodeBGRA8888(inputImage);
    if (squareRect != null) {
      final croppedImage = cropImage(image, squareRect);
      //print(croppedImage?.height);
      if (croppedImage != null) {
        //final input = convertToFloat32List(await getCnnInput(croppedImage)).reshape([1, 224, 224, 3]);
        final input = await getCnnInput(croppedImage);
        final output = Float32List(1 * 1).reshape([1, 1]);
        await _isolateInterpreter.run(input, output);
              
        setState(() {
          if (output[0][0] > 0.5) {
            flag = 1;
          } else {
            flag = 0;
          }
          _text = output[0][0].toStringAsFixed(2);
          //print('Inference Result: ${(output[0][0]).toStringAsFixed(7)}');
        });
      }
    }

    setState(() {
      _customPaint = CustomPaint(
        painter: FaceDetectorPainter(
          faces,
          inputImage.metadata!.size,
          inputImage.metadata!.rotation,
          _cameraLensDirection,
          squareRect: squareRect,
        ),
      );

      // FPS 계산
      _frameCount++; // 프레임 수 증가
      final currentTime = DateTime.now().millisecondsSinceEpoch;
      if (currentTime - _startTime >= 1000) {
        // 1초가 지나면 FPS 업데이트
        _currentFPS = _frameCount;
        _frameCount = 0; // 프레임 카운터 초기화
        _startTime = currentTime; // 시작 시간 갱신
      }
    });

    _isBusy = false;
  }

  Rect? _getSquareRect(List<Face> faces, Size imageSize) {
    for (final face in faces) {
      final landmarks = face.landmarks;
      final bottomLip = landmarks[FaceLandmarkType.bottomMouth]?.position;
      final nose = landmarks[FaceLandmarkType.noseBase]?.position;
      final leftMouth = landmarks[FaceLandmarkType.leftMouth]?.position;
      final rightMouth = landmarks[FaceLandmarkType.rightMouth]?.position;

      if (bottomLip != null && nose != null && leftMouth != null && rightMouth != null) {
        final centerX = (leftMouth.x + rightMouth.x + bottomLip.x + nose.x) / 4;
        final centerY = (leftMouth.y + rightMouth.y + bottomLip.y + nose.y) / 4;
        final squareSize = imageSize.width * 0.15;
        final squareLeft = (centerX - squareSize / 2);
        final squareTop = (centerY - squareSize / 2);

        return Rect.fromLTWH(squareLeft, squareTop, squareSize, squareSize);
      }
    }
    return null;
  }

  img.Image? cropImage(img.Image image, Rect rect) {
    return img.copyCrop(image, x: rect.left.toInt(), y: rect.top.toInt(), width: rect.width.toInt(), height: rect.height.toInt());
  }

  Float32List convertToFloat32List(List<List<List<double>>> input) {
    return Float32List.fromList(input.expand((row) => row.expand((col) => col)).toList());
  }
  Future<List> getCnnInput(img.Image image) async{
    img.Image resizedImage = img.copyResize(image, width: 224, height: 224);
    Float32List inputBytes = Float32List(1 * 224 * 224 * 3);



    final range = resizedImage.getRange(0, 0, 224, 224);
    int pixelIndex = 0;
    while (range.moveNext()) {
      final pixel = range.current;
      pixel.r = pixel.maxChannelValue - pixel.r; // Invert the red channel.
      pixel.g = pixel.maxChannelValue - pixel.g; // Invert the green channel.
      pixel.b = pixel.maxChannelValue - pixel.b; // Invert the blue channel.
      inputBytes[pixelIndex++] = pixel.r / 255.0;
      inputBytes[pixelIndex++] = pixel.g / 255.0;
      inputBytes[pixelIndex++] = pixel.b / 255.0;
    }

    final input = inputBytes.reshape([1, 224, 224, 3]);
    return input;
  }
}
