// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Runtime.CompilerServices;

using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

// This runs in a separate package built from EndToEndTests
// and for this reason it can not refer to non-public members
// of Onnxruntime package
namespace Microsoft.ML.OnnxRuntime.Tests
{
    // This is to make sure it does not run in parallel with OrtEnvTests
    // or any other test class within the same collection
    [Collection("Ort Inference Tests")]
    public partial class InferenceTest
    {
        private readonly ITestOutputHelper output;

        public InferenceTest(ITestOutputHelper o)
        {
            this.output = o;
        }
    }
}
