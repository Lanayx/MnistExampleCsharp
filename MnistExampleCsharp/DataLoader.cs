using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace MnistExampleCsharp
{
    public class DataLoader
    {
        public static Tuple<Test[], Test[]> Load()
        {
            var trainData = Load("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 50000);
            var testData = Load("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 10000);
            return new Tuple<Test[], Test[]>(trainData, testData);
        }

        public static Test[] Load(string fileName, string labelsFileName, int imagesCount)
        {
            FileStream ifsLabels =
             new FileStream(labelsFileName,
             FileMode.Open); // test labels
            FileStream ifsImages =
             new FileStream(fileName,
             FileMode.Open); // test images

            BinaryReader brLabels =
             new BinaryReader(ifsLabels);
            BinaryReader brImages =
             new BinaryReader(ifsImages);

            int magic1 = brImages.ReadInt32(); // discard
            int numImages = brImages.ReadInt32();
            int numRows = brImages.ReadInt32();
            int numCols = brImages.ReadInt32();

            int magic2 = brLabels.ReadInt32();
            int numLabels = brLabels.ReadInt32();

            var result = new List<Test>();
            var pixels = new double[784];

            // each test image
            for (int di = 0; di < imagesCount; ++di)
            {
                for (int i = 0; i < 28; ++i)
                {
                    for (int j = 0; j < 28; ++j)
                    {
                        byte b = brImages.ReadByte();
                        pixels[i*28 + j] = b/255.0;
                    }
                }

                byte lbl = brLabels.ReadByte();

                result.Add(new Test { X = DenseVector.OfArray(pixels), Y = GetVectorFromNumber(lbl) });
            } // each image

            ifsImages.Close();
            brImages.Close();
            ifsLabels.Close();
            brLabels.Close();

            return result.ToArray();
        }

        private static Vector<double> GetVectorFromNumber(byte lbl)
        {
            var array = new double[10];
            array[lbl] = 1;
            return DenseVector.OfArray(array);
        }
    }
}
