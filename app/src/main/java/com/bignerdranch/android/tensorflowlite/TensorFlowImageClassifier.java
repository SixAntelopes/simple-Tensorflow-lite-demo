package com.bignerdranch.android.tensorflowlite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

public class TensorFlowImageClassifier implements Classifier {

    private static final int MAX_RESULTS = 3;
    private static final int BATCH_SIZE = 1;
    private static final int PIXEL_SIZE = 3;
    private static final float THRESHOLD = 0.1f;

    private List<String> labelList;
    private int inputSize;
    private Interpreter interpreter;

    private TensorFlowImageClassifier() {

    }

    static Classifier create(AssetManager assetManager,
                             String modelPath,
                             String labelPath,
                             int inputSize) throws IOException {
        //对象是自身
        TensorFlowImageClassifier classifier = new TensorFlowImageClassifier();
        //标签放到list
        classifier.labelList = classifier.loadLabelList(assetManager, labelPath);
        classifier.inputSize = inputSize;
        //Tensorflow lite解释器
        classifier.interpreter = new Interpreter(classifier.loadModelFile(assetManager, modelPath), new Interpreter.Options());

        return classifier;
    }

    @Override
    public List<Recognition> recognizeImage(Bitmap bitmap) {
        //把 bitmap 传到缓冲区
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
        byte[][] result = new byte[1][labelList.size()];
        interpreter.run(byteBuffer, result);
        return getSortedResultByte(result);
    }

    @Override
    public void close() {
        interpreter.close();
    }

    //加载label（不用改
    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        List<String> labelList = new ArrayList<>();
        //InputStream：是所有字节输入流的超类
        //InputStreamReader：是字节流与字符流之间的桥梁，能将字节流输出为字符流，并且能为字节流指定字符集，可输出一个个的字符
        //BufferedReader：提供通用的缓冲方式文本读取，readLine 读取一个文本行，从字符输入流中读取文本，缓冲各个字符，从而提供字符、数组和行的高效读取
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    //加载model（不用改
    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        //AssetFileDescriptor 是 AssetManager 中一项的文件描述符。
        //openFd()获取 asset 目录下指定文件的 AssetFileDescriptor 对象
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        //getFileDescriptor()返回 FileDescriptor 对象，可用于读取文件中的数据（以及其偏移量和长度）
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        //FileChannel 用于读取、写入、映射和操作文件的通道。
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        //MappedByteBuffer 将此通道的文件区域直接映射到内存中。
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        //分配缓冲区空间，数据量大时 allocateDirect 比 allocate 操作效率高
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE);
        //读写其它类型的数据牵涉到字节序问题，ByteBuffer 会按其字节序（大字节序或小字节序）写入或读出一个其它
        //ByteOrder.nativeOrder()返回本地 jvm 运行的硬件的字节顺序.使用和硬件一致的字节顺序可能使 buffer 更加有效.
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[inputSize * inputSize];
        //getPixels()函数把一张图片，从指定的偏移位置(offset)，指定的位置(x,y)截取指定的宽高(width,height)，把所得图像的每个像素颜色转为 int 值，存入 pixels。
        //stride 参数指定在行之间跳过的像素的数目。图片是二维的，存入一个一维数组中，那么就需要这个参数来指定多少个像素换一行。
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        //The bitmap contains an encoded color for each pixel in ARGB format, so we need to mask the least significant 8 bits to get blue, and next 8 bits to get green and next 8 bits to get blue.
        //Since we have an opaque image, alpha can be ignored.
        int pixel = 0;
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.put((byte) ((val >> 16) & 0xFF));
                byteBuffer.put((byte) ((val >> 8) & 0xFF));
                byteBuffer.put((byte) (val & 0xFF));
            }
        }
        return byteBuffer;
    }

    private List<Recognition> getSortedResultByte(byte[][] labelProbArray) {
        //创建优先队列，以选出三个概率最大的结果
        PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });
        //把识别结果加入优先队列
        for (int i = 0; i < labelList.size(); ++i) {
            float confidence = (labelProbArray[0][i] & 0xff) / 255.0f;
            if (confidence > THRESHOLD) {
                pq.add(new Recognition("" + i,
                        labelList.size() > i ? labelList.get(i) : "unknown",
                        confidence));
            }
        }
        //筛选结果
        final ArrayList<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        return recognitions;
    }

}
