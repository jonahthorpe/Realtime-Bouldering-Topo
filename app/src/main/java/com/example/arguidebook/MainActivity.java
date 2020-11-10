package com.example.arguidebook;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;

import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.YuvImage;

import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.SystemClock;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;
import android.util.Log;
import android.util.Size;

import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.arguidebook.env.ImageUtils;
import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.TensorFlowLite;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;


import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;


/*
https://developer.android.com/training/camerax/preview 03/07/2020
For setting up camera lifecycle
 */

public class MainActivity extends AppCompatActivity
        {
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private PreviewView previewView;
    private ExecutorService executor;
    private Interpreter interpreter;
    private AssetManager assetManager;
    private Canvas mCanvas;
    private Paint mPaint = new Paint();
    private Bitmap mBitmap;
    private ImageView mImageView;
    private YuvToRgbConverter converter;
    private final String ARCH = "DN";
    private final String MODEL = "yolov4.tflite";
    private final int NUM_HOLDS = 6;
    private final int SIZE = 416;
    private ByteBuffer imgData = ByteBuffer.allocateDirect(SIZE * SIZE * 3 * 4);
    private  int[] intValues = new int[SIZE * SIZE];
    private ImageProcessor imageProcessor;
    private TensorImage tfImageBuffer = new TensorImage(DataType.UINT8);




    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // handle camera permissions
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, 1888);
        }

        previewView = findViewById(R.id.previewView);
        executor = Executors.newSingleThreadExecutor();

        imgData.order(ByteOrder.nativeOrder());
        //converter = new YuvToRgbConverter(this);

        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                // No errors need to be handled for this Future.
                // This should never be reached.
            }
        }, ContextCompat.getMainExecutor(this));

        assetManager = getAssets();
        try {
            interpreter = new Interpreter(loadModelFile(assetManager, MODEL));
            Log.i("detections", interpreter.getInputTensor(0).shape()[2] + "");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // set up the style for the topo
        mPaint.setColor(Color.RED);
        mPaint.setStyle(Paint.Style.STROKE);
        mPaint.setStrokeWidth(2.0f);
        mPaint.setTextSize(32f);
        mImageView = findViewById(R.id.imageView);

    }

    @SuppressLint("UnsafeExperimentalUsageError")
    void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder()
                .build();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        preview.setSurfaceProvider(previewView.createSurfaceProvider());

        ImageAnalysis imageAnalysis =
                new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

        imageAnalysis.setAnalyzer(executor, imageProxy -> {
            //int rotationDegrees = imageProxy.getImageInfo().getRotationDegrees();

            // get bitmap
            Bitmap previewBitmap = previewView.getBitmap();
            //Bitmap imageBitmap =  Bitmap.createBitmap(imageProxy.getWidth(), imageProxy.getHeight(), Bitmap.Config.ARGB_8888);
            //converter.yuvToRgb(imageProxy.getImage(), imageBitmap);
            // resize bitmap
            Bitmap bitmap = Bitmap.createScaledBitmap(previewBitmap, SIZE, SIZE, true);

            /*
            https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android/app/src/main/java/org/tensorflow/lite/examples/detection/tflite/TFLiteObjectDetectionAPIModel.java
            03/07/2020 Code to prepare and normalise input image
             */
            ////////////////////////////////////////////////////////////////////////////////////
            // normalise and put in byte buffer
            ByteBuffer input = ByteBuffer.allocateDirect(SIZE * SIZE * 3 * 4).order(ByteOrder.nativeOrder());
            input.rewind();
            bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
            for (int i = 0; i < SIZE; ++i) {
                for (int j = 0; j < SIZE; ++j) {
                    int pixelValue = intValues[i * SIZE + j];
                    input.putFloat((((pixelValue >> 16) & 0xFF) - 128.0f) / 128.0f);
                    input.putFloat((((pixelValue >> 8) & 0xFF) - 128.0f) / 128.0f);
                    input.putFloat(((pixelValue & 0xFF) - 128.0f) / 128.0f);
                }
            }
            // change inference and drawing depending on model type
            switch (ARCH){
                case "TF":
                    inference_TF(input);
                    break;
                case "DN":
                    inference_DN(input);
            }

            imageProxy.close();

        });

        Camera camera = cameraProvider.bindToLifecycle(this, cameraSelector, imageAnalysis, preview);

    }

    private void inference_DN(ByteBuffer input) {

        float[][][] outputLocations = new float[1][2535][4];
        float[][][] outputClasses = new float[1][2535][6];

        Object[] inputArray = {input};
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, outputLocations);
        outputMap.put(1, outputClasses);

        long start = SystemClock.uptimeMillis();
        interpreter.runForMultipleInputsOutputs(inputArray, outputMap);
        long end = SystemClock.uptimeMillis();
        long duration = end - start;
        Log.i("inference", duration + "");

        // defing what to times the output by to fit it to image
        float screen_mult_width = previewView.getWidth() / SIZE;
        float screen_mult_height = previewView.getHeight() / SIZE;
        float [] bestScores = new float[6];
        int [][] bestCenters = new int[6][2];

        runOnUiThread(() -> {

            mBitmap = Bitmap.createBitmap(previewView.getWidth(), previewView.getHeight(), Bitmap.Config.ARGB_8888);
            mImageView.setImageBitmap(mBitmap);
            mCanvas = new Canvas(mBitmap);

            // iterate over all detections and select the best for each class/hold
            for (int i = 0; i < 2535; i++) {
                for (int hold = 0; hold < NUM_HOLDS; hold++) {
                    if (outputClasses[0][i][hold] > bestScores[hold]) {
                        bestScores[hold] = outputClasses[0][i][hold];
                        bestCenters[hold][0] = (int) (outputLocations[0][i][0] * screen_mult_width);
                        bestCenters[hold][1] = (int) (outputLocations[0][i][1] * screen_mult_height);
                    }

                    // draw bounding box - to be commented out
                    /*
                    if (outputClasses[0][i][hold] > 0.5f) {
                        final float xCent = outputLocations[0][i][0];
                        final float yCent = outputLocations[0][i][1];
                        final float width = outputLocations[0][i][2];
                        final float height = outputLocations[0][i][3];
                        RectF location = new RectF(
                                (int) ((xCent - width / 2) * screen_mult_width),
                                (int) ((yCent - height / 2) * screen_mult_height),
                                (int) ((xCent + width / 2) * screen_mult_width),
                                (int) ((yCent + height / 2) * screen_mult_height)
                        );
                        mCanvas.drawRect(location, mPaint);
                    }
                     */
                }
            }
            // for each hold, if above a certain confidence value draw topo
            for (int hold = 1; hold < NUM_HOLDS; hold++) {
                if (bestScores[0] > 0.1 && bestScores[hold] >0.1) {
                    mCanvas.drawLine(
                            bestCenters[0][0], bestCenters[0][1],
                            bestCenters[hold][0], bestCenters[hold][1],
                            mPaint
                    );
                    String name = "";
                    switch (hold) {
                        case (1):
                            name = "roof of the world";
                            break;
                        case (2):
                            name = "roof of the world undercuts";
                            break;
                        case (3):
                            name = "roof of the world sit";
                            break;
                        case (4):
                            name = "left arete";
                            break;
                        case (5):
                            name = "right arete";
                            break;
                    }
                    mCanvas.drawText(name, bestCenters[hold][0], bestCenters[hold][1], mPaint);
                }
            }
        });
    }

    private void inference_TF(ByteBuffer input){

        /*
        https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android/app/src/main/java/org/tensorflow/lite/examples/detection/tflite/TFLiteObjectDetectionAPIModel.java
        03/07/2020 Initalising input/output and making inference
         */
        ////////////////////////////////////////////////////////////////////////////////////
        float[][][] outputLocations = new float[1][10][4];
        float[][] outputClasses = new float[1][10];
        float[][] outputScores = new float[1][10];
        float[] numDetections = new float[1];

        Object[] inputArray = {input};
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, outputLocations);
        outputMap.put(1, outputClasses);
        outputMap.put(2, outputScores);
        outputMap.put(3, numDetections);

        ////////////////////////////////////////////////////////////////////////////////////
        long start = SystemClock.uptimeMillis();
        interpreter.runForMultipleInputsOutputs(inputArray, outputMap);
        long end = SystemClock.uptimeMillis();
        long duration = end - start;
        Log.i("inference", duration  +"");

            /*
            // testing output scores
            for (float f : outputScores[0]){
               Log.i("detections", f + "");
            }

             */
        Log.i("detections", outputScores[0][0] + "");

        runOnUiThread(() -> {

            mBitmap = Bitmap.createBitmap(previewView.getWidth(), previewView.getHeight(), Bitmap.Config.ARGB_8888);
            mImageView.setImageBitmap(mBitmap);
            mCanvas = new Canvas(mBitmap);

            for(int result = 0; result < numDetections[0]; result++){
                if (outputScores[0][result] >= 0.5){
                    //Log.i("detections", outputScores[0][result] + "");
                    // inference output [top, left, bottom, right
                    // rectF input [left, top, right, bottom]
                    RectF location = new RectF(
                            (int)  (outputLocations[0][result][1] * previewView.getWidth()),
                            (int)(outputLocations[0][result][0] * previewView.getHeight() ),
                            (int)(outputLocations[0][result][3] * previewView.getWidth() ),
                            (int)(outputLocations[0][result][2] * previewView.getHeight())
                    );
                    Log.i("Detections",
                            location.top + "," +
                                    location.left + "," +
                                    location.bottom + "," +
                                    location.right
                    );

                    mCanvas.drawRect(location, mPaint);
                }
            }
        });
    }

    /** Memory-map the model file in Assets. */
     /*
    https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android/app/src/main/java/org/tensorflow/lite/examples/detection/tflite/TFLiteObjectDetectionAPIModel.java
    03/07/2020 Code to load model
     */
    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /*
   https://stackoverflow.com/questions/56772967/converting-imageproxy-to-bitmap
   03/07/2020 Code to convert image proxy to bitmap
    */
    private Bitmap convertImageProxyToBitmap(ImageProxy imageProxy){
        ImageProxy.PlaneProxy[] planes = imageProxy.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, imageProxy.getWidth(), imageProxy.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 100, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }



}
