using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace MnistExampleCsharp
{
    class Program
    {
        static void Main(string[] args)
        {
            var net = new Network(new [] {1, 2, 1});
            Console.WriteLine("Biases");
            Console.WriteLine(JsonConvert.SerializeObject(net.biases));

            Console.WriteLine("Weights");
            Console.WriteLine(JsonConvert.SerializeObject(net.weights));
        }
    }
}
