# trianglepacker
this library that help to packs triangles of a 3D mesh into a texture.
![preview1.png](https://github.com/ray-cast/trianglepacker/raw/master/preview1.png)
![preview2.png](https://github.com/ray-cast/trianglepacker/raw/master/preview2.png)

# Usage
```c++
#include "trianglepacker.hpp"

// allocate buffer for each output uv
std::vector<float2> uvs(vertices.size());

if (!ray::uvmapper::lightmappack(
    // consecutive triangle positions
    vertices.data(), vertices.size(), 
    // resolution
    512, 512, 
    // scale the vertices
    1.0, 
    // margin
    1, 
    // output (a normalized uv coordinate for each input vertex):
    uvs.data()))
{
    std::cerr << "Failed to pack all triangles into the map!" << std::endl;
    return false;
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
