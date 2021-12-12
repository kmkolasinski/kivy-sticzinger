package org.test;
import java.lang.String;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.nio.charset.Charset;
import java.util.List;
import java.io.File;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Interpreter.Options;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;


public class TestClass {
    public Interpreter interpreter;
    public TensorBuffer outputBufferTensor;
    public TensorBuffer inputX;
    public TensorBuffer inputY;

    public int[] getInputShape(int index) {
        return interpreter.getInputTensor(index).shape();
    }

    public DataType getInputDType(int index) {
        return interpreter.getInputTensor(0).dataType();
    }

    public int[] getOutputShape() {
        // tflite incorrectly estimate the output shape when resizing
        int numCols = interpreter.getOutputTensor(0).shape()[1];
        int numRows = getInputShape(0)[0];
        int[] shape = {numRows, numCols};
        return shape;
    }

    public DataType getOutputDType() {
        return interpreter.getOutputTensor(0).dataType();
    }

    public boolean loadModel(String path, int n, boolean opt){

        Interpreter.Options options = new Options();
        options.setNumThreads(n);
        options.setUseXNNPACK(opt);

        try {
            File model = new File(path);
            interpreter = new Interpreter(model, options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        interpreter.allocateTensors();

        return true;
    }

    public float[] predict(String X, String Y) {

        byte[] byteArrayX = X.getBytes(Charset.forName("ASCII"));
        ByteBuffer bufferX = ByteBuffer.wrap(byteArrayX);

        byte[] byteArrayY = Y.getBytes(Charset.forName("ASCII"));
        ByteBuffer bufferY = ByteBuffer.wrap(byteArrayY);

        Object[] inputs = {bufferX, bufferY};

        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, outputBufferTensor.getBuffer().rewind());

        interpreter.runForMultipleInputsOutputs(inputs, outputs);

        return outputBufferTensor.getFloatArray();
    }

    public void resizeInputs(int[] shapeX, int[] shapeY) {

        int[] inputShapeX = getInputShape(0);
        int[] inputShapeY = getInputShape(1);

        boolean xChanged = !Arrays.equals(shapeX, inputShapeX);
        boolean yChanged = !Arrays.equals(shapeY, inputShapeY);

        if (xChanged ) {
            interpreter.resizeInput(0, shapeX);
        }

        if (yChanged) {
            interpreter.resizeInput(1, shapeY);
        }

        if (xChanged || yChanged) {
            interpreter.allocateTensors();
        }

        if (xChanged ) {
            outputBufferTensor = TensorBuffer.createFixedSize(getOutputShape(), getOutputDType());
            inputX = TensorBufferFloat.createFixedSize(getInputShape(0), getInputDType(0));
        }

        if (yChanged) {
            inputY = TensorBufferFloat.createFixedSize(getInputShape(1), getInputDType(1));
        }
    }

}