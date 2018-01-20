# trianglepacker
trianglepacker.hpp is a C++11/17 single required source file, that help to packs triangles of a 3D mesh into a texture, 
this method is a fast greedy algorithm, the output is looks like a blender's lightmappack, but the quality is not better than blender,

this library can support rectangle pack, when part of input indices are converted from a quad to two triangles, 
that two triangle become combiend and can result in a quad, see the below images.


![preview1.png](https://github.com/ray-cast/trianglepacker/raw/master/preview1.png)
![preview2.png](https://github.com/ray-cast/trianglepacker/raw/master/preview2.png)

# Usage
```c++
#include "trianglepacker.hpp"

// method 1
// allocate buffer for each output uv
std::vector<float2> uvs(vertices.size());

if (!ray::uvmapper::lightmappack(
    // consecutive triangle positions
    vertices.data(), vertices.size(), 
    // resolution
    512, 512, 
    // scale the vertices
    1.0, 
    // margin between all triangle
    1, 
    // output (a normalized uv coordinate for each input vertex):
    uvs.data()))
{
    std::cerr << "Failed to pack all triangles into the map!" << std::endl;
    return false;
}

// method 2
// allocate buffer for each output vertex
std::vector<float2> positions(indices.size());

// allocate buffer for each output uv
std::vector<float2> uvs(indices.size());

//  allocate for each vertex count
std::size_t count = 0;

if (!ray::uvmapper::lightmappack(
    // triangle positions
    (float*)vertices.data(),
    // indices buffer
    indices.data(), indices.size(), 
    // resolution
    512, 512, 
    // scale the vertices
    1.0, 
    // margin between all triangle and quad
    1, 
    // output (a new vertices buffer for each uv coordinate):
    (float*)positions.data(), 
    // output (a normalized uv coordinate for each output vertex):
    (float*)uvs.data(),
    // output (a count of vertex that has been written)
    count))
{
    std::cerr << "Failed to pack all triangles into the map!" << std::endl;
    return 0;
}
```

[License (MIT)](https://raw.githubusercontent.com/ray-cast/trianglepacker/master/LICENSE.txt)
-------------------------------------------------------------------------------
    MIT License

    Copyright (c) 2018 Rui

	Permission is hereby granted, free of charge, to any person obtaining a
	copy of this software and associated documentation files (the "Software"),
	to deal in the Software without restriction, including without limitation
	the rights to use, copy, modify, merge, publish, distribute, sublicense,
	and/or sell copies of the Software, and to permit persons to whom the
	Software is furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included
	in all copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
	OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
	BRIAN PAUL BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
	AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
	CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
