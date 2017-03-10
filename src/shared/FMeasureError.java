package shared;



/**
 * Standard error measure, suitable for use with
 * linear output networks for regression, sigmoid
 * output networks for single class probability,
 * and soft max networks for multi class probabilities.
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class FMeasureError extends AbstractErrorMeasure
        implements GradientErrorMeasure {

    /**
     * @see nn.error.ErrorMeasure#error(double[], nn.Pattern[], int)
     */
    public double value(Instance output, Instance example) {
        int class0_pos = 0;
        int class0_count = 0;
        int class1_pos = 0;
        int class1_count = 0;

        Instance label = example.getLabel();
        for (int i = 0; i < output.size(); i++) {
            if(label.getContinuous(i) == 1){
              class1_count++;
              if( ((output.getContinuous(i) - label.getContinuous(i))
                    * (output.getContinuous(i) - label.getContinuous(i))) < 0.25 ){
                class1_pos++;
              }
            }
            else{
              class0_count++;
              if( ((output.getContinuous(i) - label.getContinuous(i))
                    * (output.getContinuous(i) - label.getContinuous(i))) < 0.25 ){
                class0_pos++;
              }
            }
        }
        double sens = class1_pos/class1_count;
        double spec = class0_pos/class0_count;

        if( sens + spec == 0 ){
          return 1;
        }
        else{
          return 1 - ( 2 * ( sens * spec ) / ( sens + spec ) );
        }
        return 1;
    }

    /**
     * @see nn.error.DifferentiableErrorMeasure#derivatives(double[], nn.Pattern[], int)
     */
    public double[] gradient(Instance output, Instance example) {
        double[] errorArray = new double[output.size()];
        Instance label = example.getLabel();
        for (int i = 0; i < output.size(); i++) {
            errorArray[i] = (output.getContinuous(i) - label.getContinuous(i))
                * example.getWeight();
        }
        return errorArray;
    }

}
