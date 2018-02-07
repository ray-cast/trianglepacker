# Triangle Packer
trianglepacker.hpp is a C++11/17 single required source file, that help to packs triangles of a 3D mesh into a texture, 
this method is a fast greedy algorithm, the output is looks like a blender's lightmappack, and the quality is near the blender.

this library can supports rectangle pack, when part of input indices are converted from a quad to two triangles, 
that two triangle become combined and can result in a quad, see the below images.


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
    (float*)vertices.data(), vertices.size(), 
    // resolution
    512, 512, 
    // margin between all triangle
    1, 
    // output (a normalized uv coordinate for each input vertex):
    (float*)uvs.data()))
{
    std::cerr << "Failed to pack all triangles into the map!" << std::endl;
    return false;
}

// method 2

// allocate for vertex count
std::size_t count = 0;

std::vector<std::uint16_t> remap(indices.size()); // allocate buffer for each vertex index
std::vector<float2> uvs(indices.size()); // allocate buffer for each output uv
std::vector<std::uint16_t> outIndices(indices.size()); // allocate buffer for each output uv

if (!ray::uvmapper::lightmappack(
    // triangle positions
    (float*)vertices.data(),
    // indices buffer
    indices.data(), indices.size(), 
    // resolution
    512, 512, 
    // margin between all triangle and quad
    1, 
    // output (a pointer to the array of remapped vertices):
    remap.data(), 
    // output (a normalized uv coordinate for each output vertex):
    (float*)uvs.data(),
    // output (a index buffer for each ouput uv):
    outIndices.data()
    // output (a count of vertex that has been written)
    count))
{
    std::cerr << "Failed to pack all triangles into the map!" << std::endl;
    return 0;
}

std::vector<Vertex> newVertices(count);

for (std::size_t i = 0; i < count; i++)
{
    newVertices[i] = vertices[remap[i]];
    newVertices[i].uv = uvs[i];
}

vertices = newVertices;
indices = outIndices;
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
# References
* Packing Lightmaps \[[link](http://blackpawn.com/texts/lightmaps/default.html)\]
