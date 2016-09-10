using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math;
using MathNet.Numerics;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Newtonsoft.Json;

namespace MnistExampleCsharp
{
    class Program
    {
        static void Main(string[] args)
        {

            Control.UseNativeOpenBLAS();
            var net = new Network(new[] { 784, 30, 10 });
            var data = DataLoader.Load();
            net.SGD(data.Item1, 30, 10, 1.0, data.Item2);

            //Test();
        }


        public static void Test()
        {
            var size = 5000;
            var x = Matrix<double>.Build.Random(size, size, new MathNet.Numerics.Distributions.Normal());
            var y = Matrix<double>.Build.Random(size, size, new MathNet.Numerics.Distributions.Normal());
            Console.WriteLine("Start");
            var start = DateTime.Now;
            var z = x * y;
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
