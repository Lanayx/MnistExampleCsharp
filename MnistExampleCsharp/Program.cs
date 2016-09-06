using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra.Double;
using Newtonsoft.Json;

namespace MnistExampleCsharp
{
    class Program
    {
        static void Main(string[] args)
        {
            var net = new Network(new[] { 784, 100, 10 });
            var data = DataLoader.Load();
            net.SGD(data.Item1, 30, 10, 3.0, data.Item2);
            Console.WriteLine("Biases");
            Console.WriteLine(JsonConvert.SerializeObject(net.biases));

            Console.WriteLine("Weights");
            Console.WriteLine(JsonConvert.SerializeObject(net.weights));

            //var x = DenseMatrix.OfRowArrays(new[] {1.0, 0.0}, new[] {2.0, 3.0});
            //var y = DenseMatrix.OfRowArrays(new[] { 2.0, 1.0 }, new[] { 0.0,4.0 });
            //var z = x*y;

        }
    }
}
