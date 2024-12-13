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
  var _cameraLensDirection = CameraLensDirection.front;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/MobileNetV2(200).tflite',
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
    return DetectorView(
      title: 'Face Detector',
      customPaint: _customPaint,
      text: _text,
      onImage: _processImage,
      initialCameraLensDirection: _cameraLensDirection,
      onCameraLensDirectionChanged: (value) => _cameraLensDirection = value,
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
    final image = decodeYUV420SP(inputImage);
    if (squareRect != null) {
      final croppedImage = cropImage(image, squareRect);
      if (croppedImage != null) {
        //final input = convertToFloat32List(await getCnnInput(croppedImage)).reshape([1, 224, 224, 3]);
        final input = await getCnnInput2(croppedImage);
        final output = Float32List(1 * 1).reshape([1, 1]);
        await _isolateInterpreter.run(input, output);
        
        setState(() {
          _text = 'Inference Result: ${output[0]}';
          print('Inference Result: ${(output[0][0]).toStringAsFixed(7)}');
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
        final squareSize = imageSize.width * 0.2;
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
  Future<List> getCnnInput2(img.Image image) async{
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

  Future<List<List<List<double>>>> getCnnInput(img.Image image) async {
  //print('cnn input');
  // 이미지를 [224, 224] 크기로 리사이즈합니다.
  img.Image resizedImage = img.copyResize(image, width: 224, height: 224);

  // 결과를 저장할 배열을 초기화합니다.
  List<List<List<double>>> result = List.generate(224, 
    (_) => List.generate(224, 
      (_) => List.filled(3, 0.0)));
  // 픽셀 데이터를 [0, 1] 범위로 정규화하여 배열에 저장합니다.
  for (int y = 0; y < resizedImage.height; y++) {
    for (int x = 0; x < resizedImage.width; x++) {
      img.Color pixel = resizedImage.getPixel(x, y);
      double r = pixel.r / 255.0;
      double g = pixel.g / 255.0;
      double b = pixel.b / 255.0;
      result[y][x][0] = r; // Red
      result[y][x][1] = g; // Green
      result[y][x][2] = b; // Blue
    }
  }
  //img.Pixel pixel = resizedImage.getPixel(100, 100);
  //print(pixel);
  //print(result.shape);
  return result;
}

  Uint8List convertYUV420ToRGBA(Uint8List bytes, int width, int height) {
    final int frameSize = width * height;
    final Uint8List rgba = Uint8List(4 * frameSize);

    for (int j = 0, yp = 0; j < height; j++) {
      int uvp = frameSize + (j >> 1) * width, u = 0, v = 0;
      for (int i = 0; i < width; i++, yp++) {
        int y = 0xff & bytes[yp];
        if ((i & 1) == 0) {
          v = 0xff & bytes[uvp++];
          u = 0xff & bytes[uvp++];
        }
        y = y < 16 ? 16 : y;

        int r = (1192 * (y - 16) + 1634 * (v - 128)) >> 10;
        int g = (1192 * (y - 16) - 833 * (v - 128) - 400 * (u - 128)) >> 10;
        int b = (1192 * (y - 16) + 2066 * (u - 128)) >> 10;

        r = r.clamp(0, 255);
        g = g.clamp(0, 255);
        b = b.clamp(0, 255);

        rgba[yp * 4] = r;
        rgba[yp * 4 + 1] = g;
        rgba[yp * 4 + 2] = b;
        rgba[yp * 4 + 3] = 255;
      }
    }
    return rgba;
  }
}
