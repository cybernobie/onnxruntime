// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is copied and adapted from the following git repository -
// https://github.com/dotnet/corefx
// Commit ID: bdd0814360d4c3a58860919f292a306242f27da1
// Path: /src/System.Numerics.Tensors/tests/TensorTests.cs
// Original license statement below -

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace Microsoft.ML.OnnxRuntime.Tensors.Tests
{
    public class TensorTests : TensorTestsBase
    {
        [Theory(DisplayName = "ConstructTensorFromArrayRank1")]
        [MemberData(nameof(GetSingleTensorConstructors))]
        public void ConstructTensorFromArrayRank1(TensorConstructor tensorConstructor)
        {
            var tensor = tensorConstructor.CreateFromArray<int>(new[] { 0, 1, 2 });

            Assert.Equal(tensorConstructor.IsReversedStride, tensor.IsReversedStride);
            Assert.Equal(0, tensor[0]);
            Assert.Equal(1, tensor[1]);
            Assert.Equal(2, tensor[2]);
        }

    }        
}