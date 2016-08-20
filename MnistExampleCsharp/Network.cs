using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MnistExampleCsharp
{
    public class Network
    {
        public int num_layers;
        public int[] sizes;
        public double[][][] biases;
        public double[][][] weights;

        public Network(int[] sizes)//2 3 4
        {
            var r = new Random(0);
            this.num_layers = sizes.Length;
            this.sizes = sizes;
            this.biases = sizes.Skip(1).Select(x => (new int[x])
                .Select(e =>new [] { GetRandomNumber(-3, 3, r) }).ToArray()).ToArray();

            this.weights = new double[sizes.Length-1][][];
            for (int i = 0; i < sizes.Length-1; i++)
            {
                this.weights[i] = Enumerable.Range(0, sizes[i + 1])
                    .Select(e => Enumerable.Range(0, sizes[i]).Select(e1 => GetRandomNumber(-3, 3, r)).ToArray()).ToArray();
            }
        }

        public double GetRandomNumber(double minimum, double maximum, Random random)
        {
            return random.NextDouble() * (maximum - minimum) + minimum;
        }
    }
}
