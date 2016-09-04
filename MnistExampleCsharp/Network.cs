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
        public Vector<double>[] biases;
        public Matrix<double>[] weights;

        public Network(int[] sizes)//2 3 4
        {
            var r = new Random(0);
            this.num_layers = sizes.Length;
            this.sizes = sizes;
            this.biases = sizes.Skip(1).Select(x => 
                 DenseVector.OfArray((new int[x]).Select(e => GetRandomNumber(-3, 3, r)).ToArray())
                ).ToArray();



            this.weights = new Matrix<double>[sizes.Length-1];
            for (int i = 0; i < sizes.Length-1; i++)
            {
                this.weights[i] = DenseMatrix.OfRowArrays(Enumerable.Range(0, sizes[i + 1])
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

        public Vector<double> Sigmoid(Vector<double> z)
        {
            //"""The sigmoid function."""
            return z.Map(SigmoidSimple, Zeros.Include);
        }

        public Vector<double> SigmoidPrime(Vector<double> z)
        {
            //"""The sigmoid function."""
            return Sigmoid(z).PointwiseMultiply(1- Sigmoid(z));
        }

        public void SGD(Test[] training_data , int epochs , int mini_batch_size , double eta , Test[] test_data)
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

            var rnd = new Random();
            foreach (var j in Enumerable.Range(0, epochs))
            {
                //randomize
                training_data = training_data.OrderBy(x => rnd.Next()).ToArray();
                
                var mini_batches = Split(training_data, mini_batch_size);
                var c = 0;
                foreach (var mini_batch in mini_batches)
                {
                    Update_mini_batch(mini_batch.ToArray(), eta);
                    Console.WriteLine(c++);
                }
                if (test_data != null)
                    Console.WriteLine("Epoch {0}: {1} / {2}", j, Evaluate(test_data), n_test);
                else
                    Console.WriteLine("Epoch {0} complete",j);
            }
        }

        public void Update_mini_batch(Test[] mini_batch, double eta)
        {
            var nabla_b = biases.Select(b => b.Map(value => 0.0)).ToArray();
            var nabla_w = weights.Select(b => b.Map(value => 0.0)).ToArray();
            foreach (var test in mini_batch)
            {
                var backPropRes = Backprop(test.X, test.Y);
                var delta_nabla_b = backPropRes.Item1;
                var delta_nabla_w = backPropRes.Item2;
                for (int i = 0; i < weights.Length; i++)
                {
                    nabla_w[i] += delta_nabla_w[i];
                    nabla_b[i] += delta_nabla_b[i];

                    weights[i] -= nabla_w[i] * eta / mini_batch.Length;
                    biases[i] = biases[i] - nabla_b[i] * eta / mini_batch.Length; 
                }
            }
        }

        public Tuple<Vector<double>[], Matrix<double>[]> Backprop(Vector<double> x, Vector<double> y) {
            //"""Return a tuple ``(nabla_b, nabla_w)`` representing the
            //gradient for the cost function C_x.  ``nabla_b`` and
            //``nabla_w`` are layer-by-layer lists of numpy arrays, similar
            //to ``self.biases`` and ``self.weights``."""
            var nabla_b = biases.Select(b => b.Map(value => 0.0)).ToArray();
            var nabla_w = weights.Select(b => b.Map(value => 0.0)).ToArray();
            //# feedforward
            var activation = x;
            var activations = new List<Vector<double>> {x}; // # list to store all the activations, layer by layer
            var zs = new List<Vector<double>>(); // # list to store all the z vectors, layer by layer

            for (int i = 0; i < weights.Length; i++)
            {
                var z = weights[i]*activation + biases[i];
                zs.Add(z);
                activation = Sigmoid(z);
                activations.Add(activation);
            }
            //# backward pass
            var delta = CostDerivative(activations[activations.Count- 1], y).PointwiseMultiply(SigmoidPrime(zs[zs.Count-1]));
            nabla_b[nabla_b.Length-1] = delta;
            nabla_w[nabla_w.Length - 1] = delta.ToColumnMatrix() * activations[activations.Count - 2].ToRowMatrix();
            //# Note that the variable l in the loop below is used a little
            //# differently to the notation in Chapter 2 of the book.  Here,
            //# l = 1 means the last layer of neurons, l = 2 is the
            //# second-last layer, and so on.  It's a renumbering of the
            //# scheme in the book, used here to take advantage of the fact
            //# that Python can use negative indices in lists.
            for (var i = num_layers - 2; i > 0; i--)
            {
                var z = zs[i-1];
                var sp = SigmoidPrime(z);
                delta = (weights[i].Transpose()*delta).PointwiseMultiply(sp);
                nabla_b[i-1] = delta;
                nabla_w[i-1] = delta.ToColumnMatrix() *activations[i - 1].ToRowMatrix();    
            }

            return new Tuple<Vector<double>[], Matrix<double>[]>(nabla_b, nabla_w);
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
    

    public int Evaluate(Test[] test_data) {
        //"""Return the number of test inputs for which the neural
        //network outputs the correct result.Note that the neural
        //network's output is assumed to be the index of whichever
        //neuron in the final layer has the highest activation."""

        return test_data.Count(test => Feedforward(test.X).MaximumIndex() == test.Y.MaximumIndex());
    }

    public Vector<double> Feedforward(Vector<double> a) {
            //"""Return the output of the network if ``a`` is input."""
        for (int i = 0; i < weights.Length; i++)
        {
            a = Sigmoid(weights[i]*a + biases[i]);
        }
        return a;
    }

    public Vector<double> CostDerivative(Vector<double> output_activations, Vector<double> y)
    { 
        //"""Return the vector of partial derivatives \partial C_x /
        //\partial a for the output activations."""
        return output_activations-y;
    }
  }
}

