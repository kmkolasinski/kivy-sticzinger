package org.test;
import java.lang.String;
import java.util.ArrayList;
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
    public TensorBuffer inputTensorBuffer;

    public int[] getInputShape() {
        return interpreter.getInputTensor(0).shape();
    }

    public DataType getInputDType() {
        return interpreter.getInputTensor(0).dataType();
    }

    public int[] getOutputShape() {
        return interpreter.getOutputTensor(0).shape();
    }

    public DataType getOutputDType() {
        return interpreter.getOutputTensor(0).dataType();
    }

    public boolean loadModel(String path){

        Interpreter.Options options = new Options();
        options.setNumThreads(1);
        try {
            File model = new File(path);
            interpreter = new Interpreter(model, options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        interpreter.allocateTensors();
        outputBufferTensor = TensorBuffer.createFixedSize(getOutputShape(), getOutputDType());
        inputTensorBuffer = TensorBufferFloat.createFixedSize(getInputShape(), getInputDType());

        return true;
    }

    public void predict(String array) {

        // System.out.println("Predicting array with N elements: " + array.length);

       // inputTensorBuffer.loadArray(array);

       // interpreter.run(input, self.output_buffer.getBuffer().rewind())

    }

}