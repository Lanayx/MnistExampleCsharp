using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;
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
            //Console.WriteLine("Biases");
            //Console.WriteLine(JsonConvert.SerializeObject(net.biases));

            //Console.WriteLine("Weights");
            //Console.WriteLine(JsonConvert.SerializeObject(net.weights));

            //var x = DenseMatrix.OfRowArrays(new[] {1.0, 0.0}, new[] {2.0, 3.0});
            //var y = DenseMatrix.OfRowArrays(new[] { 2.0, 1.0 }, new[] { 0.0,4.0 });
            //var z = x*y;

            //Test2();
        }


        public static void Test()
        {
            var size = 5000;
            var r = new Random();
            var x = DenseMatrix.Create(size,size,(rows, columns) => GetRandomNumber(r));
            var y = DenseMatrix.Create(size, size, (rows, columns) => GetRandomNumber(r));
            Console.WriteLine("Start");
            var start = DateTime.Now;
            DenseMatrix z;
            z = x * y;
            Console.WriteLine(z.RowCount);
            Console.WriteLine((DateTime.Now - start).TotalSeconds);
        }


        public static void Test2()
        {
            var size = 5000;
            var r = new Random();
            var x = new double[size, size];
            for(var i  =0; i < size; i++)
                for (var j = 0; j < size; j++)
                    x[i, j] = GetRandomNumber(r);

            var y = new double[size, size];
            for (var i = 0; i < size; i++)
                for (var j = 0; j < size; j++)
                    y[i, j] = GetRandomNumber(r);

            Console.WriteLine("Start");
            var start = DateTime.Now;
            var z = x.Dot(y);

            Console.WriteLine(z.Rows());
            Console.WriteLine((DateTime.Now - start).TotalSeconds);
        }

        public static double GetRandomNumber(Random rand)
        {
            double u1 = rand.NextDouble(); //these are uniform(0,1) random doubles
            double u2 = rand.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) *
                         Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
        }


    }
}
