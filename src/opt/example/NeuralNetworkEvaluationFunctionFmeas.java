package opt.example;

import util.linalg.Vector;
import func.nn.NeuralNetwork;
import opt.EvaluationFunction;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;

/**
 * An evaluation function that uses a neural network
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class NeuralNetworkEvaluationFunctionFmeas implements EvaluationFunction {
    /**
     * The network
     */
    private NeuralNetwork network;
    /**
     * The examples
     */
    private DataSet examples;
    /**
     * The error measure
     */
    private ErrorMeasure measure;

    /**
     * Make a new neural network evaluation function
     * @param network the network
     * @param examples the examples
     * @param measure the error measure
     */
    public NeuralNetworkEvaluationFunctionFmeas(NeuralNetwork network,
            DataSet examples, ErrorMeasure measure) {
        this.network = network;
        this.examples = examples;
        this.measure = measure;
    }

    /**
     * @see opt.OptimizationProblem#value(opt.OptimizationData)
     */
    public double value(Instance d) {
        // set the links
        Vector weights = d.getData();
        network.setWeights(weights);
        // calculate the error
        double outputLabel = 0;
        double actualLabel = 0;
        int class0_pos = 0;
        int class0_count = 0;
        int class1_pos = 0;
        int class1_count = 0;

        for (int i = 0; i < examples.size(); i++) {
            network.setInputValues(examples.get(i).getData());
            network.run();
            outputLabel = network.getOutputValues().get(0);
            actualLabel = examples.get(i).getLabel().getContinuous();
            if( actualLabel == 1 ){
              class1_count++;
              if( (actualLabel-outputLabel) * (actualLabel-outputLabel) < 0.25 ){
                class1_pos++;
              }
            }
            else{
              class0_count++;
              if( (actualLabel-outputLabel) * (actualLabel-outputLabel) < 0.25 ){
                class1_pos++;
              }
            }
        }
        double sens = class1_pos/class1_count;
        double spec = class0_pos/class0_count;

        if( sens + spec == 0 ){
          // return fitness
          return 1;
        }
        else{
          // return fitness 1/error
          return 1 / (1 - ( 2 * ( sens * spec ) / ( sens + spec ) ));
        }
    }
}
