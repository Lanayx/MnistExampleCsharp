using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace MnistExampleCsharp
{
    public class Network
    {
        public int num_layers;
        public int[] sizes;
        public DenseVector[] biases;
        public DenseMatrix[] weights;

        public Network(int[] sizes)//2 3 4
        {
            var r = new Random(0);
            this.num_layers = sizes.Length;
            this.sizes = sizes;
            this.biases = sizes.Skip(1).Select(x => 
                 DenseVector.OfArray((new int[x]).Select(e => GetRandomNumber(-3, 3, r)).ToArray())
                ).ToArray();



            this.weights = new DenseMatrix[sizes.Length-1];
            for (int i = 0; i < sizes.Length-1; i++)
            {
                this.weights[i] = DenseMatrix.OfColumnArrays(Enumerable.Range(0, sizes[i + 1])
                    .Select(e => Enumerable.Range(0, sizes[i]).Select(e1 => GetRandomNumber(-3, 3, r)).ToArray()).ToArray());
            }
        }

        public double GetRandomNumber(double minimum, double maximum, Random random)
        {
            return random.NextDouble() * (maximum - minimum) + minimum;
        }
        public double SigmoidSimple(double z) {
            //"""The sigmoid function."""
            return 1.0 / (1.0 + Math.Exp(z));
        }

        public DenseMatrix Sigmoid(DenseMatrix z)
        {
            //"""The sigmoid function."""
            return (DenseMatrix)z.Map(SigmoidSimple,Zeros.Include);
        }

        public DenseVector Sigmoid(DenseVector z)
        {
            //"""The sigmoid function."""
            return (DenseVector)z.Map(SigmoidSimple, Zeros.Include);
        }

        public void SGD(dynamic[] training_data , int epochs , int mini_batch_size , double eta , dynamic[] test_data)
        {
            //"""Train the neural network using mini-batch stochastic
            //gradient descent.The "training_data" is a list of tuples
            //"(x, y)" representing the training inputs and the desired
            //outputs.  The other non-optional parameters are
            //self-explanatory.If "test_data" is provided then the
            //network will be evaluated against the test data after each
            //epoch, and partial progress printed out.  This is useful for
            ////tracking progress, but slows things down substantially."""
            int n_test = 0;
            if (test_data != null)
                n_test = test_data.Length;
            var n = training_data.Length;
            foreach (var j in Enumerable.Range(0, epochs))
            {
                //random.shuffle(training_data)
                var mini_batches = Split(training_data, mini_batch_size);
                foreach (var mini_batch in mini_batches)
                    Update_mini_batch(mini_batch, eta);
                if (test_data != null)
                    Console.WriteLine("Epoch {0}: {1} / {2}", j, Evaluate(test_data), n_test);
                else
                    Console.WriteLine("Epoch {0} complete",j);
            }
        }

        public void Update_mini_batch(dynamic mini_batch, double eta)
        {
            var nabla_b = [np.zeros(b.shape)for b in self.biases]
            var nabla_w = [np.zeros(w.shape)for w in self.weights]
            for x, y in mini_batch:
                delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.weights = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
        }

        /// <summary>
    /// Splits an array into several smaller arrays.
    /// </summary>
    /// <typeparam name="T">The type of the array.</typeparam>
    /// <param name="array">The array to split.</param>
    /// <param name="size">The size of the smaller arrays.</param>
    /// <returns>An array containing smaller arrays.</returns>
    public static IEnumerable<IEnumerable<T>> Split<T>(T[] array, int size)
        {
            for (var i = 0; i < (float)array.Length / size; i++)
            {
                yield return array.Skip(i * size).Take(size);
            }
        }
    }
}
