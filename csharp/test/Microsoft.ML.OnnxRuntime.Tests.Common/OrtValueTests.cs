using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Runtime.InteropServices;
using System.Text;
using Xunit;

namespace Microsoft.ML.OnnxRuntime.Tests
{
    [Collection("OrtValueTests")]
    public class OrtValueTests
    {
        public OrtValueTests()
        {
        }

        [Fact(DisplayName = "PopulateAndReadStringTensor")]
        public void PopulateAndReadStringTensor()
        {
            OrtEnv.Instance();

            string[] strsRom = { "HelloR", "OrtR", "WorldR" };
            string[] strs = { "Hello", "Ort", "World" };
            long[] shape = { 1, 1, 3 };
            var elementsNum = ShapeUtils.GetSizeForShape(shape);
            Assert.Equal(elementsNum, strs.Length);
            Assert.Equal(elementsNum, strsRom.Length);

            using (var strTensor = OrtValue.CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, shape))
            {
                Assert.True(strTensor.IsTensor);
                Assert.False(strTensor.IsSparseTensor);
                Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, strTensor.OnnxType);
                var typeShape = strTensor.GetTensorTypeAndShape();
                {
                    Assert.True(typeShape.IsString);
                    Assert.Equal(shape.Length, typeShape.DimensionsCount);
                    var fetchedShape = typeShape.Shape;
                    Assert.Equal(shape.Length, fetchedShape.Length);
                    Assert.Equal(shape, fetchedShape);
                    Assert.Equal(elementsNum, typeShape.ElementCount);
                }

                using (var memInfo = strTensor.GetTensorMemoryInfo())
                {
                    Assert.Equal("Cpu", memInfo.Name);
                    Assert.Equal(OrtMemType.Default, memInfo.GetMemoryType());
                    Assert.Equal(OrtAllocatorType.DeviceAllocator, memInfo.GetAllocatorType());
                }

                // Verify that everything is empty now.
                for (int i = 0; i < elementsNum; ++i)
                {
                    var str = strTensor.GetStringElement(i);
                    Assert.Empty(str);

                    var rom = strTensor.GetStringElementAsMemory(i);
                    Assert.Equal(0, rom.Length);

                    var bytes = strTensor.GetStringElementAsSpan(i);
                    Assert.Equal(0, bytes.Length);
                }

                // Let's populate the tensor with strings.
                for (int i = 0; i < elementsNum; ++i)
                {
                    // First populate via ROM
                    strTensor.StringTensorSetElementAt(strsRom[i].AsMemory(), i);
                    Assert.Equal(strsRom[i], strTensor.GetStringElement(i));
                    Assert.Equal(strsRom[i], strTensor.GetStringElementAsMemory(i).ToString());
                    Assert.Equal(Encoding.UTF8.GetBytes(strsRom[i]), strTensor.GetStringElementAsSpan(i).ToArray());

                    // Fill via Span
                    strTensor.StringTensorSetElementAt(strs[i].AsSpan(), i);
                    Assert.Equal(strs[i], strTensor.GetStringElement(i));
                    Assert.Equal(strs[i], strTensor.GetStringElementAsMemory(i).ToString());
                    Assert.Equal(Encoding.UTF8.GetBytes(strs[i]), strTensor.GetStringElementAsSpan(i).ToArray());
                }
            }
        }

    }
}