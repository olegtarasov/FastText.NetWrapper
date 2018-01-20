using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FastText.NetWrapper;

namespace TestUtil
{
    class Program
    {
        static void Main(string[] args)
        {
            var fastText = new FastTextWrapper();
            fastText.Train(@"D:\__Models\cooking.train.txt", @"D:\__Models\fasttext");
        }
    }
}
