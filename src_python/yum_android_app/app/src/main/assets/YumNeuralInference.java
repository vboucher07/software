
/**
 * Android Neural Network Inference for Yum Bot
 * Direct implementation of the 1mil_test.pkl neural network (176 avg score)
 */
public class YumNeuralInference {
    
    public static class NetworkOutput {
        public float value;
        public float[] actionLogits;
        
        public NetworkOutput(float value, float[] actionLogits) {
            this.value = value;
            this.actionLogits = actionLogits;
        }
    }
    
    public static NetworkOutput forward(float[] input, NeuralWeights weights) {
        // Layer 0: Linear(128) + ReLU
        float[] x = matmul(input, weights.w0);
        x = addBias(x, weights.b0);
        x = relu(x);
        
        // Layer 1: Linear(256) + ReLU  
        x = matmul(x, weights.w1);
        x = addBias(x, weights.b1);
        x = relu(x);
        
        // Layer 2: Linear(128) + ReLU
        x = matmul(x, weights.w2);
        x = addBias(x, weights.b2);
        x = relu(x);
        
        // Value head: Linear(1)
        float[] valueOut = matmul(x, weights.w3);
        valueOut = addBias(valueOut, weights.b3);
        float value = valueOut[0];
        
        // Action head: Linear(32)
        float[] actionLogits = matmul(x, weights.w4);
        actionLogits = addBias(actionLogits, weights.b4);
        
        return new NetworkOutput(value, actionLogits);
    }
    
    private static float[] matmul(float[] input, float[][] weights) {
        int outputSize = weights[0].length;
        float[] output = new float[outputSize];
        
        for (int i = 0; i < outputSize; i++) {
            float sum = 0.0f;
            for (int j = 0; j < input.length; j++) {
                sum += input[j] * weights[j][i];
            }
            output[i] = sum;
        }
        return output;
    }
    
    private static float[] addBias(float[] input, float[] bias) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] + bias[i];
        }
        return output;
    }
    
    private static float[] relu(float[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.max(0.0f, input[i]);
        }
        return output;
    }
}
