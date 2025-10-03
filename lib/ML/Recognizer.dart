import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import '../DB/DatabaseHelper.dart';
import 'Recognition.dart';

class Recognizer {
  late Interpreter interpreter;
  late InterpreterOptions _interpreterOptions;
  static const int WIDTH = 160;
  static const int HEIGHT = 160;
  static const int OUTPUT = 512;
  final dbHelper = DatabaseHelper();
  Map<String, List<List<dynamic>>> registered =
      {}; // Store multiple embeddings per name

  @override
  String get modelName => 'assets/facenet.tflite';

  Recognizer({int? numThreads}) {
    _interpreterOptions = InterpreterOptions();

    if (numThreads != null) {
      _interpreterOptions.threads = numThreads;
    }
    loadModel();
    initDB();
  }

  initDB() async {
    await dbHelper.init();
    loadRegisteredFaces();
  }

  void loadRegisteredFaces() async {
    registered.clear();
    final allRows = await dbHelper.queryAllRows();

    for (final row in allRows) {
      String name = row[DatabaseHelper.columnName];

      // Decode JSON stored embeddings
      List<List<dynamic>> embeddings =
          (jsonDecode(row[DatabaseHelper.columnEmbedding]) as List)
              .map((e) => (e as List).map((v) => v.toDouble()).toList())
              .toList();

      registered[name] = embeddings;
    }
  }

  void registerFaceInDB(String name, List<List<double>> embeddings, Uint8List faceImage) async {
    String embeddingJson = jsonEncode(embeddings); // Store embeddings as JSON

    Map<String, dynamic> row = {
      DatabaseHelper.columnName: name,
      DatabaseHelper.columnEmbedding: embeddingJson,
      'image': faceImage,

    };

    final id = await dbHelper.insert(row);
    print('Inserted row ID: $id');

    loadRegisteredFaces();
  }

  Future<void> loadModel() async {
    try {
      interpreter = await Interpreter.fromAsset(modelName);
    } catch (e) {
      print('Unable to create interpreter, Caught Exception: ${e.toString()}');
    }
  }

  List<dynamic> imageToArray(img.Image inputImage) {
    img.Image resizedImage =
        img.copyResize(inputImage, width: WIDTH, height: HEIGHT);
    List<double> flattenedList = resizedImage.data!
        .expand((channel) => [channel.r, channel.g, channel.b])
        .map((value) => value.toDouble())
        .toList();

    Float32List float32Array = Float32List.fromList(flattenedList);
    int channels = 3;
    int height = HEIGHT;
    int width = WIDTH;
    Float32List reshapedArray = Float32List(1 * height * width * channels);

    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          int index = c * height * width + h * width + w;
          reshapedArray[index] =
              (float32Array[c * height * width + h * width + w] - 127.5) /
                  127.5;
        }
      }
    }
    return reshapedArray.reshape([1, WIDTH, HEIGHT, 3]);
  }

  Recognition recognize(img.Image image, Rect location) {
    var input = imageToArray(image);
    print(input.shape.toString());

    List output = List.filled(1 * OUTPUT, 0).reshape([1, OUTPUT]);

    final runs = DateTime.now().millisecondsSinceEpoch;
    interpreter.run(input, output);
    final run = DateTime.now().millisecondsSinceEpoch - runs;
    print('Time to run inference: $run ms $output');

    List<double> outputArray = output.first.cast<double>();

    Pair pair = findNearest(outputArray);
    print("Best match distance = ${pair.distance}");

    return Recognition(pair.name, location, outputArray, pair.distance);
  }

  Pair findNearest(List<double> emb) {
    Pair bestMatch = Pair("Unknown", 0);
    double threshold = 0.3;
    print("registered= ${registered.length}");

    for (MapEntry<String, List<List<dynamic>>> entry in registered.entries) {
      final String name = entry.key;
      final List<List<dynamic>> storedEmbeddings = entry.value;

      for (List<dynamic> storedEmb in storedEmbeddings) {
        double similarity = cosineSimilarity(emb, storedEmb);
        print("registered similarity= $similarity");
        if (similarity > bestMatch.distance && similarity > threshold) {
          bestMatch = Pair(name, similarity);
        }
      }
    }

    return bestMatch;
  }

  double cosineSimilarity(List<dynamic> emb1, List<dynamic> emb2) {
    double dotProduct = 0.0, normA = 0.0, normB = 0.0;

    for (int i = 0; i < emb1.length; i++) {
      dotProduct += emb1[i] * emb2[i];
      normA += emb1[i] * emb1[i];
      normB += emb2[i] * emb2[i];
    }

    return dotProduct / (sqrt(normA) * sqrt(normB));
  }

  void close() {
    interpreter.close();
  }
}

class Pair {
  String name;
  double distance;
  Pair(this.name, this.distance);
}
