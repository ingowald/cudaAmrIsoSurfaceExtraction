// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <stdio.h>

#include <cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include <fstream>
#include <stdexcept>
#include <iostream>

int divUp(int a, int b) { return (a+b-1)/b; }

void cuda_safe_call(cudaError_t error, const std::string& message = "")
{
  if(error)
    throw thrust::system_error(error, thrust::cuda_category(), message);
}

#define EXPLICIT_MORTON 1

struct timer
{
  cudaEvent_t start;
  cudaEvent_t end;

  timer(void)
  {
    cuda_safe_call(cudaEventCreate(&start));
    cuda_safe_call(cudaEventCreate(&end));
    restart();
  }

  ~timer(void)
  {
    cuda_safe_call(cudaEventDestroy(start));
    cuda_safe_call(cudaEventDestroy(end));
  }

  void restart(void)
  {
    cuda_safe_call(cudaEventRecord(start, 0));
  }

  double elapsed(void)
  {
    cuda_safe_call(cudaEventRecord(end, 0));
    cuda_safe_call(cudaEventSynchronize(end));

    float ms_elapsed;
    cuda_safe_call(cudaEventElapsedTime(&ms_elapsed, start, end));
    return ms_elapsed / 1e3;
  }

  double epsilon(void)
  {
    return 0.5e-6;
  }
};

// ------------------------------------------------------------------
// vec3i
// ------------------------------------------------------------------

struct vec3i
{
  inline __host__ __device__ vec3i() {}
  inline __host__ __device__ vec3i(int i) : x(i), y(i), z(i) {}
  inline __host__ __device__ vec3i(int x, int y, int z) : x(x), y(y), z(z) {}
  int x,y,z;
};

inline std::ostream &operator<<(std::ostream &o, const vec3i &v)
{ o << "(" << v.x << "," << v.y << "," << v.z <<")"; return o; }
inline __host__ __device__ vec3i operator+(const vec3i &a, const vec3i &b)
{ return { a.x+b.x, a.y+b.y, a.z+b.z }; }
inline __host__ __device__ vec3i operator-(const vec3i &a, const vec3i &b)
{ return { a.x-b.x, a.y-b.y, a.z-b.z }; }
inline __host__ __device__ vec3i operator*(const vec3i &a, const vec3i &b)
{ return { a.x*b.x, a.y*b.y, a.z*b.z }; }
inline __host__ __device__ vec3i operator*(const vec3i &a, const int b)
{ return { a.x*b, a.y*b, a.z*b }; }
inline __host__ vec3i min(const vec3i &a, const vec3i &b)
{ return vec3i(std::min(a.x,b.x),std::min(a.y,b.y),std::min(a.z,b.z)); }
inline __host__ vec3i max(const vec3i &a, const vec3i &b)
{ return vec3i(std::max(a.x,b.x),std::max(a.y,b.y),std::max(a.z,b.z)); }

inline __device__ __host__ vec3i operator>>(const vec3i v, const int s)
{
  return vec3i(v.x>>s, v.y>>s, v.z>>s);
}

inline __device__ __host__ uint64_t leftShift3(uint64_t x)
{
  x = (x | x << 32) & 0x1f00000000ffffull;
  x = (x | x << 16) & 0x1f0000ff0000ffull; 
  x = (x | x <<  8) & 0x100f00f00f00f00full;
  x = (x | x <<  4) & 0x10c30c30c30c30c3ull;
  x = (x | x <<  2) & 0x1249249249249249ull;
  return x;
}

inline __device__ __host__ uint64_t mortonCode(const vec3i v) {
  return
    (leftShift3(uint32_t(v.z)) << 2) |
    (leftShift3(uint32_t(v.y)) << 1) |
    (leftShift3(uint32_t(v.x)) << 0);
}

inline __host__ __device__
bool operator==(const vec3i &a, const vec3i &b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}


// ------------------------------------------------------------------
// vec3f
// ------------------------------------------------------------------

struct vec3f
{
  inline __host__ __device__ vec3f() {}
  inline __host__ __device__ vec3f(const float x, const float y, const float z) : x(x), y(y), z(z) {}
  inline __host__ __device__ vec3f(const float f) : x(f), y(f), z(f) {}
  inline __host__ __device__ vec3f(const vec3i o) : x(o.x), y(o.y), z(o.z) {}
  
  float x,y,z;
};

inline __host__ __device__ vec3f operator+(const vec3f &a, const vec3f &b)
{ return { a.x+b.x, a.y+b.y, a.z+b.z }; }
inline __host__ __device__ vec3f operator-(const vec3f &a, const vec3f &b)
{ return { a.x-b.x, a.y-b.y, a.z-b.z }; }
inline __host__ __device__ vec3f operator*(const vec3f &a, const vec3f &b)
{ return { a.x*b.x, a.y*b.y, a.z*b.z }; }
inline __host__ __device__ vec3f operator*(const vec3f &a, const float b)
{ return { a.x*b, a.y*b, a.z*b }; }
inline __host__ __device__ bool operator==(const vec3f &a, const vec3f &b)
{ return a.x==b.x && a.y==b.y && a.z==b.z; }

inline __host__ __device__
bool operator<(const vec3f &a, const vec3f &b)
{
  return (a.x < b.x)
    || ((a.x == b.x) &&
        ((a.y < b.y)
         || (a.y == b.y) && (a.z < b.z)));
}

// ------------------------------------------------------------------
// vec4f
// ------------------------------------------------------------------

struct vec4f
{
  inline __host__ __device__ vec4f() {}
  inline __host__ __device__ vec4f(const vec3f v, float w)
    : x(v.x),y(v.y),z(v.z),w(w)
  {}
  inline __host__ __device__ vec4f(float f)
    : x(f),y(f),z(f),w(f)
  {}
  inline __host__ __device__ vec4f(float x, float y, float z, float w)
    : x(x),y(y),z(z),w(w)
  {}
  
  float x,y,z,w;
};

inline __host__ __device__ vec4f operator+(const vec4f &a, const vec4f &b)
{ return { a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w }; }
inline __host__ __device__ vec4f operator*(const vec4f &a, const vec4f &b)
{ return { a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w }; }
inline __host__ __device__ vec4f operator-(const vec4f &a, const vec4f &b)
{ return { a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w }; }
inline __host__ __device__ vec4f operator*(const vec4f &a, const float b)
{ return { a.x*b, a.y*b, a.z*b, a.w*b }; }
inline __host__ __device__ vec4f operator*(const float b, const vec4f &a)
{ return { a.x*b, a.y*b, a.z*b, a.w*b }; }

inline __host__ __device__ bool operator==(const vec4f &a, const vec4f &b)
{ return a.x==b.x && a.y==b.y && a.z==b.z && a.w==b.w; }


inline __host__ __device__ float4 operator+(const float4 &a, const float4 &b)
{ return { a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w }; }
inline __host__ __device__ float4 operator*(const float4 &a, const float4 &b)
{ return { a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w }; }
inline __host__ __device__ float4 operator-(const float4 &a, const float4 &b)
{ return { a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w }; }
inline __host__ __device__ float4 operator*(const float4 &a, const float b)
{ return { a.x*b, a.y*b, a.z*b, a.w*b }; }
inline __host__ __device__ float4 operator*(const float b, const float4 &a)
{ return { a.x*b, a.y*b, a.z*b, a.w*b }; }

inline __host__ __device__ bool operator==(const float4 &a, const float4 &b)
{ return a.x==b.x && a.y==b.y && a.z==b.z && a.w==b.w; }


// ------------------------------------------------------------------
// Cell Coords
// ------------------------------------------------------------------

struct CellCoords {
  inline __host__ __device__ CellCoords neighbor(const vec3i &delta) const
  { return { lower+delta*(1<<level),level }; }
  
  inline __host__ __device__ vec3f center() const
  { return vec3f(lower)+vec3f(0.5f*(1<<level)); }
  
  vec3i lower;
  int   level;
};


inline __host__ __device__ bool operator<(const CellCoords &a, const CellCoords &b)
{
  return
    (a.lower < b.lower)
    ||
    (a.lower == b.lower && b.level < b.level);
}

inline __host__ __device__ bool operator==(const CellCoords &a, const CellCoords &b)
{
  return (a.lower == b.lower) && (a.level == b.level);
}

// ------------------------------------------------------------------
// Cell
// ------------------------------------------------------------------

struct Cell : public CellCoords {
  inline __device__ __host__ float4 asDualVertex() const
  {
    return make_float4(center().x,center().y,center().z,scalar);
  }
  float      scalar;
};

inline __host__ __device__ bool operator==(const Cell &a, const Cell &b)
{
  return ((const CellCoords&)a == (const CellCoords&)b) && (a.scalar == b.scalar);
}

inline __host__ __device__ bool operator!=(const Cell &a, const Cell &b)
{ return !(a==b); }

#if EXPLICIT_MORTON
// ------------------------------------------------------------------
// Explicit morton-ordered array
// ------------------------------------------------------------------
struct Morton {
  uint64_t    morton;
  const Cell *cell;
};

struct CompareMorton {
  inline __host__ __device__ bool operator()(const Morton &a, const Morton &b)
  { return a.morton < b.morton; }
  inline __host__ __device__ bool operator()(const Morton &a, const uint64_t b)
  { return a.morton < b; }
};
#endif



// ------------------------------------------------------------------
// "Fat" Triangle Vertex
// ------------------------------------------------------------------

struct TriangleVertex
{
  vec3f    position;
  uint32_t triangleAndVertexID;
};

struct CompareByCoordsLowerOnly {
  inline __host__  __device__ CompareByCoordsLowerOnly(const vec3i coordOrigin)
    : coordOrigin(coordOrigin)
  {}
  
  __host__ __device__ bool operator()(const Cell &lhs, const CellCoords &rhs) const
  { return (mortonCode(lhs.lower - coordOrigin) < mortonCode(rhs.lower - coordOrigin)); }
  const vec3i coordOrigin;
};

struct CompareVertices {
  __host__ __device__ bool operator()(const TriangleVertex &lhs, const TriangleVertex &rhs) const
  {
    const float4 a = (const float4 &)lhs;
    const float4 b = (const float4 &)rhs;
    
    return (const vec3f&)a < (const vec3f&)b;
  }
};


vec3i readInput(thrust::host_vector<Cell> &h_cells,
                int &maxLevel,
                const std::string &cellFileName,
                const std::string &scalarFileName)
{
  vec3i coordOrigin(1<<30);
  std::ifstream in_cells(cellFileName,std::ios::binary);
  std::ifstream in_scalars(scalarFileName,std::ios::binary);

  vec3i bounds_lower(1<<30);
  vec3i bounds_upper(-(1<<30));
  
  maxLevel = 0;
  while (!in_cells.eof()) {
    Cell cell;
    in_cells.read((char*)&cell,sizeof(CellCoords));
    in_scalars.read((char*)&cell.scalar,sizeof(float));

    if (!(in_cells.good() && in_scalars.good()))
      break;
    
    maxLevel = std::max(maxLevel,cell.level);
    h_cells.push_back(cell);
    bounds_lower = min(bounds_lower,cell.lower);
    bounds_upper = max(bounds_upper,cell.lower+vec3i(1<<cell.level));
    coordOrigin = min(coordOrigin,cell.lower);
    static size_t nextPing = 1;
    if (h_cells.size() >= nextPing) {
      std::cout << "read so far : " << h_cells.size() << "..." << std::endl;
      nextPing *= 2;
    }
  }
  std::cout << "done reading, found " << h_cells.size() << " cells" << std::endl;
  std::cout << "bounds " << bounds_lower << ".." << bounds_upper << " logical size " << (bounds_upper-bounds_lower) << std::endl;
  std::cout << "coord origin: " << coordOrigin << std::endl;

  coordOrigin.x &= ~((1<<maxLevel)-1);
  coordOrigin.y &= ~((1<<maxLevel)-1);
  coordOrigin.z &= ~((1<<maxLevel)-1);
  
  std::cout << "coord origin: " << coordOrigin << std::endl;
  
  return coordOrigin;
}

struct AMR {
  __host__ __device__ inline AMR(
#if EXPLICIT_MORTON
                                 const Morton *const __restrict__ mortonArray,
#endif
                                 const vec3i                    coordOrigin,
                                 const Cell *const __restrict__ cellArray,
                                 const int                      numCells,
                                 const int                      maxLevel)
    :
#if EXPLICIT_MORTON
    mortonArray(mortonArray),
#endif
    coordOrigin(coordOrigin),
    cellArray(cellArray),
    numCells(numCells),
    maxLevel(maxLevel)
  {}
                        
  __host__ __device__ bool findActual(Cell &result, const CellCoords &coords)
  {
#if EXPLICIT_MORTON
    const Morton *const __restrict__ begin = mortonArray;
    const Morton *const __restrict__ end   = mortonArray+numCells;

    const Morton *it = thrust::system::detail::generic::scalar::lower_bound
      (begin,end,mortonCode(coords.lower - coordOrigin),CompareMorton());
    
    if (it == end) return false;

    const Cell found = *it->cell;
    if ((found.lower >> max(coords.level,found.level))
        ==
        (coords.lower >> max(coords.level,found.level))
        // &&
        // (found.level >= coords.level)
        ) {
      result = found;
      return true;
    }

    if (it > begin) {
    const Cell found = *it[-1].cell;
    if ((found.lower >> max(coords.level,found.level))
      ==
      (coords.lower >> max(coords.level,found.level))
      // &&
      // (found.level >= coords.level)
      ) {
    result = found;
    return true;
    }
  }
        
    return false;
#else
    const Cell *const __restrict__ begin = cellArray;
    const Cell *const __restrict__ end   = cellArray+numCells;

    const Cell *it = thrust::system::detail::generic::scalar::lower_bound
      (begin,end,coords,CompareByCoordsLowerOnly(coordOrigin));
    
    if (it == end) return false;
    
    if ((it->lower >> it->level) == (coords.lower >> it->level)
        &&
        (it->level >= coords.level)
        ) {
      result = *it;
      return true;
    }
    
    return false;
#endif
  }
  
  const Cell *const __restrict__ cellArray;
  const int                      numCells;
  const int                      maxLevel;
  const vec3i                    coordOrigin;

#if EXPLICIT_MORTON
  const Morton *const __restrict__ mortonArray;
#endif
};


__constant__ int8_t vtkMarchingCubesTriangleCases[256][16]
= {
  {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 0 0 */
  { 0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 1 1 */
  { 0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 2 1 */
  { 1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 3 2 */
  { 1, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 4 1 */
  { 0, 3, 8, 1, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 5 3 */
  { 9, 11, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 6 2 */
  { 2, 3, 8, 2, 8, 11, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1}, /* 7 5 */
  { 3, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 8 1 */
  { 0, 2, 10, 8, 0, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 9 2 */
  { 1, 0, 9, 2, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 10 3 */
  { 1, 2, 10, 1, 10, 9, 9, 10, 8, -1, -1, -1, -1, -1, -1, -1}, /* 11 5 */
  { 3, 1, 11, 10, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 12 2 */
  { 0, 1, 11, 0, 11, 8, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1}, /* 13 5 */
  { 3, 0, 9, 3, 9, 10, 10, 9, 11, -1, -1, -1, -1, -1, -1, -1}, /* 14 5 */
  { 9, 11, 8, 11, 10, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 15 8 */
  { 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 16 1 */
  { 4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 17 2 */
  { 0, 9, 1, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 18 3 */
  { 4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1}, /* 19 5 */
  { 1, 11, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 20 4 */
  { 3, 7, 4, 3, 4, 0, 1, 11, 2, -1, -1, -1, -1, -1, -1, -1}, /* 21 7 */
  { 9, 11, 2, 9, 2, 0, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1}, /* 22 7 */
  { 2, 9, 11, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1}, /* 23 14 */
  { 8, 7, 4, 3, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 24 3 */
  {10, 7, 4, 10, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1}, /* 25 5 */
  { 9, 1, 0, 8, 7, 4, 2, 10, 3, -1, -1, -1, -1, -1, -1, -1}, /* 26 6 */
  { 4, 10, 7, 9, 10, 4, 9, 2, 10, 9, 1, 2, -1, -1, -1, -1}, /* 27 9 */
  { 3, 1, 11, 3, 11, 10, 7, 4, 8, -1, -1, -1, -1, -1, -1, -1}, /* 28 7 */
  { 1, 11, 10, 1, 10, 4, 1, 4, 0, 7, 4, 10, -1, -1, -1, -1}, /* 29 11 */
  { 4, 8, 7, 9, 10, 0, 9, 11, 10, 10, 3, 0, -1, -1, -1, -1}, /* 30 12 */
  { 4, 10, 7, 4, 9, 10, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1}, /* 31 5 */
  { 9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 32 1 */
  { 9, 4, 5, 0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 33 3 */
  { 0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 34 2 */
  { 8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1}, /* 35 5 */
  { 1, 11, 2, 9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 36 3 */
  { 3, 8, 0, 1, 11, 2, 4, 5, 9, -1, -1, -1, -1, -1, -1, -1}, /* 37 6 */
  { 5, 11, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1}, /* 38 5 */
  { 2, 5, 11, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1}, /* 39 9 */
  { 9, 4, 5, 2, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 40 4 */
  { 0, 2, 10, 0, 10, 8, 4, 5, 9, -1, -1, -1, -1, -1, -1, -1}, /* 41 7 */
  { 0, 4, 5, 0, 5, 1, 2, 10, 3, -1, -1, -1, -1, -1, -1, -1}, /* 42 7 */
  { 2, 5, 1, 2, 8, 5, 2, 10, 8, 4, 5, 8, -1, -1, -1, -1}, /* 43 11 */
  {11, 10, 3, 11, 3, 1, 9, 4, 5, -1, -1, -1, -1, -1, -1, -1}, /* 44 7 */
  { 4, 5, 9, 0, 1, 8, 8, 1, 11, 8, 11, 10, -1, -1, -1, -1}, /* 45 12 */
  { 5, 0, 4, 5, 10, 0, 5, 11, 10, 10, 3, 0, -1, -1, -1, -1}, /* 46 14 */
  { 5, 8, 4, 5, 11, 8, 11, 10, 8, -1, -1, -1, -1, -1, -1, -1}, /* 47 5 */
  { 9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 48 2 */
  { 9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1}, /* 49 5 */
  { 0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1}, /* 50 5 */
  { 1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 51 8 */
  { 9, 8, 7, 9, 7, 5, 11, 2, 1, -1, -1, -1, -1, -1, -1, -1}, /* 52 7 */
  {11, 2, 1, 9, 0, 5, 5, 0, 3, 5, 3, 7, -1, -1, -1, -1}, /* 53 12 */
  { 8, 2, 0, 8, 5, 2, 8, 7, 5, 11, 2, 5, -1, -1, -1, -1}, /* 54 11 */
  { 2, 5, 11, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1}, /* 55 5 */
  { 7, 5, 9, 7, 9, 8, 3, 2, 10, -1, -1, -1, -1, -1, -1, -1}, /* 56 7 */
  { 9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 10, 7, -1, -1, -1, -1}, /* 57 14 */
  { 2, 10, 3, 0, 8, 1, 1, 8, 7, 1, 7, 5, -1, -1, -1, -1}, /* 58 12 */
  {10, 1, 2, 10, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1}, /* 59 5 */
  { 9, 8, 5, 8, 7, 5, 11, 3, 1, 11, 10, 3, -1, -1, -1, -1}, /* 60 10 */
  { 5, 0, 7, 5, 9, 0, 7, 0, 10, 1, 11, 0, 10, 0, 11, -1}, /* 61 7 */
  {10, 0, 11, 10, 3, 0, 11, 0, 5, 8, 7, 0, 5, 0, 7, -1}, /* 62 7 */
  {10, 5, 11, 7, 5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 63 2 */
  {11, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 64 1 */
  { 0, 3, 8, 5, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 65 4 */
  { 9, 1, 0, 5, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 66 3 */
  { 1, 3, 8, 1, 8, 9, 5, 6, 11, -1, -1, -1, -1, -1, -1, -1}, /* 67 7 */
  { 1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 68 2 */
  { 1, 5, 6, 1, 6, 2, 3, 8, 0, -1, -1, -1, -1, -1, -1, -1}, /* 69 7 */
  { 9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1}, /* 70 5 */
  { 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1}, /* 71 11 */
  { 2, 10, 3, 11, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 72 3 */
  {10, 8, 0, 10, 0, 2, 11, 5, 6, -1, -1, -1, -1, -1, -1, -1}, /* 73 7 */
  { 0, 9, 1, 2, 10, 3, 5, 6, 11, -1, -1, -1, -1, -1, -1, -1}, /* 74 6 */
  { 5, 6, 11, 1, 2, 9, 9, 2, 10, 9, 10, 8, -1, -1, -1, -1}, /* 75 12 */
  { 6, 10, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1}, /* 76 5 */
  { 0, 10, 8, 0, 5, 10, 0, 1, 5, 5, 6, 10, -1, -1, -1, -1}, /* 77 14 */
  { 3, 6, 10, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1}, /* 78 9 */
  { 6, 9, 5, 6, 10, 9, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1}, /* 79 5 */
  { 5, 6, 11, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 80 3 */
  { 4, 0, 3, 4, 3, 7, 6, 11, 5, -1, -1, -1, -1, -1, -1, -1}, /* 81 7 */
  { 1, 0, 9, 5, 6, 11, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1}, /* 82 6 */
  {11, 5, 6, 1, 7, 9, 1, 3, 7, 7, 4, 9, -1, -1, -1, -1}, /* 83 12 */
  { 6, 2, 1, 6, 1, 5, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1}, /* 84 7 */
  { 1, 5, 2, 5, 6, 2, 3, 4, 0, 3, 7, 4, -1, -1, -1, -1}, /* 85 10 */
  { 8, 7, 4, 9, 5, 0, 0, 5, 6, 0, 6, 2, -1, -1, -1, -1}, /* 86 12 */
  { 7, 9, 3, 7, 4, 9, 3, 9, 2, 5, 6, 9, 2, 9, 6, -1}, /* 87 7 */
  { 3, 2, 10, 7, 4, 8, 11, 5, 6, -1, -1, -1, -1, -1, -1, -1}, /* 88 6 */
  { 5, 6, 11, 4, 2, 7, 4, 0, 2, 2, 10, 7, -1, -1, -1, -1}, /* 89 12 */
  { 0, 9, 1, 4, 8, 7, 2, 10, 3, 5, 6, 11, -1, -1, -1, -1}, /* 90 13 */
  { 9, 1, 2, 9, 2, 10, 9, 10, 4, 7, 4, 10, 5, 6, 11, -1}, /* 91 6 */
  { 8, 7, 4, 3, 5, 10, 3, 1, 5, 5, 6, 10, -1, -1, -1, -1}, /* 92 12 */
  { 5, 10, 1, 5, 6, 10, 1, 10, 0, 7, 4, 10, 0, 10, 4, -1}, /* 93 7 */
  { 0, 9, 5, 0, 5, 6, 0, 6, 3, 10, 3, 6, 8, 7, 4, -1}, /* 94 6 */
  { 6, 9, 5, 6, 10, 9, 4, 9, 7, 7, 9, 10, -1, -1, -1, -1}, /* 95 3 */
  {11, 9, 4, 6, 11, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 96 2 */
  { 4, 6, 11, 4, 11, 9, 0, 3, 8, -1, -1, -1, -1, -1, -1, -1}, /* 97 7 */
  {11, 1, 0, 11, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1}, /* 98 5 */
  { 8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 11, 1, -1, -1, -1, -1}, /* 99 14 */
  { 1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1}, /* 100 5 */
  { 3, 8, 0, 1, 9, 2, 2, 9, 4, 2, 4, 6, -1, -1, -1, -1}, /* 101 12 */
  { 0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 102 8 */
  { 8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1}, /* 103 5 */
  {11, 9, 4, 11, 4, 6, 10, 3, 2, -1, -1, -1, -1, -1, -1, -1}, /* 104 7 */
  { 0, 2, 8, 2, 10, 8, 4, 11, 9, 4, 6, 11, -1, -1, -1, -1}, /* 105 10 */
  { 3, 2, 10, 0, 6, 1, 0, 4, 6, 6, 11, 1, -1, -1, -1, -1}, /* 106 12 */
  { 6, 1, 4, 6, 11, 1, 4, 1, 8, 2, 10, 1, 8, 1, 10, -1}, /* 107 7 */
  { 9, 4, 6, 9, 6, 3, 9, 3, 1, 10, 3, 6, -1, -1, -1, -1}, /* 108 11 */
  { 8, 1, 10, 8, 0, 1, 10, 1, 6, 9, 4, 1, 6, 1, 4, -1}, /* 109 7 */
  { 3, 6, 10, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1}, /* 110 5 */
  { 6, 8, 4, 10, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 111 2 */
  { 7, 6, 11, 7, 11, 8, 8, 11, 9, -1, -1, -1, -1, -1, -1, -1}, /* 112 5 */
  { 0, 3, 7, 0, 7, 11, 0, 11, 9, 6, 11, 7, -1, -1, -1, -1}, /* 113 11 */
  {11, 7, 6, 1, 7, 11, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1}, /* 114 9 */
  {11, 7, 6, 11, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1}, /* 115 5 */
  { 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1}, /* 116 14 */
  { 2, 9, 6, 2, 1, 9, 6, 9, 7, 0, 3, 9, 7, 9, 3, -1}, /* 117 7 */
  { 7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1}, /* 118 5 */
  { 7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 119 2 */
  { 2, 10, 3, 11, 8, 6, 11, 9, 8, 8, 7, 6, -1, -1, -1, -1}, /* 120 12 */
  { 2, 7, 0, 2, 10, 7, 0, 7, 9, 6, 11, 7, 9, 7, 11, -1}, /* 121 7 */
  { 1, 0, 8, 1, 8, 7, 1, 7, 11, 6, 11, 7, 2, 10, 3, -1}, /* 122 6 */
  {10, 1, 2, 10, 7, 1, 11, 1, 6, 6, 1, 7, -1, -1, -1, -1}, /* 123 3 */
  { 8, 6, 9, 8, 7, 6, 9, 6, 1, 10, 3, 6, 1, 6, 3, -1}, /* 124 7 */
  { 0, 1, 9, 10, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 125 4 */
  { 7, 0, 8, 7, 6, 0, 3, 0, 10, 10, 0, 6, -1, -1, -1, -1}, /* 126 3 */
  { 7, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 127 1 */
  { 7, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 128 1 */
  { 3, 8, 0, 10, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 129 3 */
  { 0, 9, 1, 10, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 130 4 */
  { 8, 9, 1, 8, 1, 3, 10, 6, 7, -1, -1, -1, -1, -1, -1, -1}, /* 131 7 */
  {11, 2, 1, 6, 7, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 132 3 */
  { 1, 11, 2, 3, 8, 0, 6, 7, 10, -1, -1, -1, -1, -1, -1, -1}, /* 133 6 */
  { 2, 0, 9, 2, 9, 11, 6, 7, 10, -1, -1, -1, -1, -1, -1, -1}, /* 134 7 */
  { 6, 7, 10, 2, 3, 11, 11, 3, 8, 11, 8, 9, -1, -1, -1, -1}, /* 135 12 */
  { 7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 136 2 */
  { 7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1}, /* 137 5 */
  { 2, 6, 7, 2, 7, 3, 0, 9, 1, -1, -1, -1, -1, -1, -1, -1}, /* 138 7 */
  { 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1}, /* 139 14 */
  {11, 6, 7, 11, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1}, /* 140 5 */
  {11, 6, 7, 1, 11, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1}, /* 141 9 */
  { 0, 7, 3, 0, 11, 7, 0, 9, 11, 6, 7, 11, -1, -1, -1, -1}, /* 142 11 */
  { 7, 11, 6, 7, 8, 11, 8, 9, 11, -1, -1, -1, -1, -1, -1, -1}, /* 143 5 */
  { 6, 4, 8, 10, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 144 2 */
  { 3, 10, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1}, /* 145 5 */
  { 8, 10, 6, 8, 6, 4, 9, 1, 0, -1, -1, -1, -1, -1, -1, -1}, /* 146 7 */
  { 9, 6, 4, 9, 3, 6, 9, 1, 3, 10, 6, 3, -1, -1, -1, -1}, /* 147 11 */
  { 6, 4, 8, 6, 8, 10, 2, 1, 11, -1, -1, -1, -1, -1, -1, -1}, /* 148 7 */
  { 1, 11, 2, 3, 10, 0, 0, 10, 6, 0, 6, 4, -1, -1, -1, -1}, /* 149 12 */
  { 4, 8, 10, 4, 10, 6, 0, 9, 2, 2, 9, 11, -1, -1, -1, -1}, /* 150 10 */
  {11, 3, 9, 11, 2, 3, 9, 3, 4, 10, 6, 3, 4, 3, 6, -1}, /* 151 7 */
  { 8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1}, /* 152 5 */
  { 0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 153 8 */
  { 1, 0, 9, 2, 4, 3, 2, 6, 4, 4, 8, 3, -1, -1, -1, -1}, /* 154 12 */
  { 1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1}, /* 155 5 */
  { 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 11, -1, -1, -1, -1}, /* 156 14 */
  {11, 0, 1, 11, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1}, /* 157 5 */
  { 4, 3, 6, 4, 8, 3, 6, 3, 11, 0, 9, 3, 11, 3, 9, -1}, /* 158 7 */
  {11, 4, 9, 6, 4, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 159 2 */
  { 4, 5, 9, 7, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 160 3 */
  { 0, 3, 8, 4, 5, 9, 10, 6, 7, -1, -1, -1, -1, -1, -1, -1}, /* 161 6 */
  { 5, 1, 0, 5, 0, 4, 7, 10, 6, -1, -1, -1, -1, -1, -1, -1}, /* 162 7 */
  {10, 6, 7, 8, 4, 3, 3, 4, 5, 3, 5, 1, -1, -1, -1, -1}, /* 163 12 */
  { 9, 4, 5, 11, 2, 1, 7, 10, 6, -1, -1, -1, -1, -1, -1, -1}, /* 164 6 */
  { 6, 7, 10, 1, 11, 2, 0, 3, 8, 4, 5, 9, -1, -1, -1, -1}, /* 165 13 */
  { 7, 10, 6, 5, 11, 4, 4, 11, 2, 4, 2, 0, -1, -1, -1, -1}, /* 166 12 */
  { 3, 8, 4, 3, 4, 5, 3, 5, 2, 11, 2, 5, 10, 6, 7, -1}, /* 167 6 */
  { 7, 3, 2, 7, 2, 6, 5, 9, 4, -1, -1, -1, -1, -1, -1, -1}, /* 168 7 */
  { 9, 4, 5, 0, 6, 8, 0, 2, 6, 6, 7, 8, -1, -1, -1, -1}, /* 169 12 */
  { 3, 2, 6, 3, 6, 7, 1, 0, 5, 5, 0, 4, -1, -1, -1, -1}, /* 170 10 */
  { 6, 8, 2, 6, 7, 8, 2, 8, 1, 4, 5, 8, 1, 8, 5, -1}, /* 171 7 */
  { 9, 4, 5, 11, 6, 1, 1, 6, 7, 1, 7, 3, -1, -1, -1, -1}, /* 172 12 */
  { 1, 11, 6, 1, 6, 7, 1, 7, 0, 8, 0, 7, 9, 4, 5, -1}, /* 173 6 */
  { 4, 11, 0, 4, 5, 11, 0, 11, 3, 6, 7, 11, 3, 11, 7, -1}, /* 174 7 */
  { 7, 11, 6, 7, 8, 11, 5, 11, 4, 4, 11, 8, -1, -1, -1, -1}, /* 175 3 */
  { 6, 5, 9, 6, 9, 10, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1}, /* 176 5 */
  { 3, 10, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1}, /* 177 9 */
  { 0, 8, 10, 0, 10, 5, 0, 5, 1, 5, 10, 6, -1, -1, -1, -1}, /* 178 14 */
  { 6, 3, 10, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1}, /* 179 5 */
  { 1, 11, 2, 9, 10, 5, 9, 8, 10, 10, 6, 5, -1, -1, -1, -1}, /* 180 12 */
  { 0, 3, 10, 0, 10, 6, 0, 6, 9, 5, 9, 6, 1, 11, 2, -1}, /* 181 6 */
  {10, 5, 8, 10, 6, 5, 8, 5, 0, 11, 2, 5, 0, 5, 2, -1}, /* 182 7 */
  { 6, 3, 10, 6, 5, 3, 2, 3, 11, 11, 3, 5, -1, -1, -1, -1}, /* 183 3 */
  { 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1}, /* 184 11 */
  { 9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1}, /* 185 5 */
  { 1, 8, 5, 1, 0, 8, 5, 8, 6, 3, 2, 8, 6, 8, 2, -1}, /* 186 7 */
  { 1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 187 2 */
  { 1, 6, 3, 1, 11, 6, 3, 6, 8, 5, 9, 6, 8, 6, 9, -1}, /* 188 7 */
  {11, 0, 1, 11, 6, 0, 9, 0, 5, 5, 0, 6, -1, -1, -1, -1}, /* 189 3 */
  { 0, 8, 3, 5, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 190 4 */
  {11, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 191 1 */
  {10, 11, 5, 7, 10, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 192 2 */
  {10, 11, 5, 10, 5, 7, 8, 0, 3, -1, -1, -1, -1, -1, -1, -1}, /* 193 7 */
  { 5, 7, 10, 5, 10, 11, 1, 0, 9, -1, -1, -1, -1, -1, -1, -1}, /* 194 7 */
  {11, 5, 7, 11, 7, 10, 9, 1, 8, 8, 1, 3, -1, -1, -1, -1}, /* 195 10 */
  {10, 2, 1, 10, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1}, /* 196 5 */
  { 0, 3, 8, 1, 7, 2, 1, 5, 7, 7, 10, 2, -1, -1, -1, -1}, /* 197 12 */
  { 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 10, -1, -1, -1, -1}, /* 198 14 */
  { 7, 2, 5, 7, 10, 2, 5, 2, 9, 3, 8, 2, 9, 2, 8, -1}, /* 199 7 */
  { 2, 11, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1}, /* 200 5 */
  { 8, 0, 2, 8, 2, 5, 8, 5, 7, 11, 5, 2, -1, -1, -1, -1}, /* 201 11 */
  { 9, 1, 0, 5, 3, 11, 5, 7, 3, 3, 2, 11, -1, -1, -1, -1}, /* 202 12 */
  { 9, 2, 8, 9, 1, 2, 8, 2, 7, 11, 5, 2, 7, 2, 5, -1}, /* 203 7 */
  { 1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 204 8 */
  { 0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1}, /* 205 5 */
  { 9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1}, /* 206 5 */
  { 9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 207 2 */
  { 5, 4, 8, 5, 8, 11, 11, 8, 10, -1, -1, -1, -1, -1, -1, -1}, /* 208 5 */
  { 5, 4, 0, 5, 0, 10, 5, 10, 11, 10, 0, 3, -1, -1, -1, -1}, /* 209 14 */
  { 0, 9, 1, 8, 11, 4, 8, 10, 11, 11, 5, 4, -1, -1, -1, -1}, /* 210 12 */
  {11, 4, 10, 11, 5, 4, 10, 4, 3, 9, 1, 4, 3, 4, 1, -1}, /* 211 7 */
  { 2, 1, 5, 2, 5, 8, 2, 8, 10, 4, 8, 5, -1, -1, -1, -1}, /* 212 11 */
  { 0, 10, 4, 0, 3, 10, 4, 10, 5, 2, 1, 10, 5, 10, 1, -1}, /* 213 7 */
  { 0, 5, 2, 0, 9, 5, 2, 5, 10, 4, 8, 5, 10, 5, 8, -1}, /* 214 7 */
  { 9, 5, 4, 2, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 215 4 */
  { 2, 11, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1}, /* 216 9 */
  { 5, 2, 11, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1}, /* 217 5 */
  { 3, 2, 11, 3, 11, 5, 3, 5, 8, 4, 8, 5, 0, 9, 1, -1}, /* 218 6 */
  { 5, 2, 11, 5, 4, 2, 1, 2, 9, 9, 2, 4, -1, -1, -1, -1}, /* 219 3 */
  { 8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1}, /* 220 5 */
  { 0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 221 2 */
  { 8, 5, 4, 8, 3, 5, 9, 5, 0, 0, 5, 3, -1, -1, -1, -1}, /* 222 3 */
  { 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 223 1 */
  { 4, 7, 10, 4, 10, 9, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1}, /* 224 5 */
  { 0, 3, 8, 4, 7, 9, 9, 7, 10, 9, 10, 11, -1, -1, -1, -1}, /* 225 12 */
  { 1, 10, 11, 1, 4, 10, 1, 0, 4, 7, 10, 4, -1, -1, -1, -1}, /* 226 11 */
  { 3, 4, 1, 3, 8, 4, 1, 4, 11, 7, 10, 4, 11, 4, 10, -1}, /* 227 7 */
  { 4, 7, 10, 9, 4, 10, 9, 10, 2, 9, 2, 1, -1, -1, -1, -1}, /* 228 9 */
  { 9, 4, 7, 9, 7, 10, 9, 10, 1, 2, 1, 10, 0, 3, 8, -1}, /* 229 6 */
  {10, 4, 7, 10, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1}, /* 230 5 */
  {10, 4, 7, 10, 2, 4, 8, 4, 3, 3, 4, 2, -1, -1, -1, -1}, /* 231 3 */
  { 2, 11, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1}, /* 232 14 */
  { 9, 7, 11, 9, 4, 7, 11, 7, 2, 8, 0, 7, 2, 7, 0, -1}, /* 233 7 */
  { 3, 11, 7, 3, 2, 11, 7, 11, 4, 1, 0, 11, 4, 11, 0, -1}, /* 234 7 */
  { 1, 2, 11, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 235 4 */
  { 4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1}, /* 236 5 */
  { 4, 1, 9, 4, 7, 1, 0, 1, 8, 8, 1, 7, -1, -1, -1, -1}, /* 237 3 */
  { 4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 238 2 */
  { 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 239 1 */
  { 9, 8, 11, 11, 8, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 240 8 */
  { 3, 9, 0, 3, 10, 9, 10, 11, 9, -1, -1, -1, -1, -1, -1, -1}, /* 241 5 */
  { 0, 11, 1, 0, 8, 11, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1}, /* 242 5 */
  { 3, 11, 1, 10, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 243 2 */
  { 1, 10, 2, 1, 9, 10, 9, 8, 10, -1, -1, -1, -1, -1, -1, -1}, /* 244 5 */
  { 3, 9, 0, 3, 10, 9, 1, 9, 2, 2, 9, 10, -1, -1, -1, -1}, /* 245 3 */
  { 0, 10, 2, 8, 10, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 246 2 */
  { 3, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 247 1 */
  { 2, 8, 3, 2, 11, 8, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1}, /* 248 5 */
  { 9, 2, 11, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 249 2 */
  { 2, 8, 3, 2, 11, 8, 0, 8, 1, 1, 8, 11, -1, -1, -1, -1}, /* 250 3 */
  { 1, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 251 1 */
  { 1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 252 2 */
  { 0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 253 1 */
  { 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 254 1 */
  {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
}; /* 255 0 */

__constant__ int8_t vtkMarchingCubes_edges[12][2]
= { {0,1}, {1,2}, {3,2}, {0,3},
    {4,5}, {5,6}, {7,6}, {4,7},
    {0,4}, {1,5}, {3,7}, {2,6}};

struct IsoExtractor {
  __device__ __host__ IsoExtractor(const float isoValue,
                                   TriangleVertex *outputArray,
                                   int             outputArraySize,
                                   int            *p_atomicCounter
                                   )
    : isoValue(isoValue),
      outputArray(outputArray),
      outputArraySize(outputArraySize),
      p_atomicCounter(p_atomicCounter)
  {}
  
  const float           isoValue;
  TriangleVertex *const outputArray;
  const int             outputArraySize;
  int            *const p_atomicCounter;
  
  inline int __device__  allocTriangle()
  {
    return atomicAdd(p_atomicCounter,1);
  }
  
  inline void __device__ doMarchingCubesOn(const vec3i mirror,
                                           const Cell zOrder[2][2][2])
  {
    // we have OUR cells in z-order, but VTK case table assumes
    // everything is is VTK 'hexahedron' ordering, so let's rearrange
    // ... and while doing so, also make sure that we flip based on
    // which direction the parent cell created this dual from
    float4 vertex[8] = {
      zOrder[0+mirror.z][0+mirror.y][0+mirror.x].asDualVertex(),
      zOrder[0+mirror.z][0+mirror.y][1-mirror.x].asDualVertex(),
      zOrder[0+mirror.z][1-mirror.y][1-mirror.x].asDualVertex(),
      zOrder[0+mirror.z][1-mirror.y][0+mirror.x].asDualVertex(),
      zOrder[1-mirror.z][0+mirror.y][0+mirror.x].asDualVertex(),
      zOrder[1-mirror.z][0+mirror.y][1-mirror.x].asDualVertex(),
      zOrder[1-mirror.z][1-mirror.y][1-mirror.x].asDualVertex(),
      zOrder[1-mirror.z][1-mirror.y][0+mirror.x].asDualVertex()
    };
    
    int index = 0;
    for (int i=0;i<8;i++)
      if (vertex[i].w > isoValue)
        index += (1<<i);
    if (index == 0 || index == 0xff) return;

    for (const int8_t *edge = &vtkMarchingCubesTriangleCases[index][0];
         edge[0] > -1;
         edge += 3 ) {
      float4 triVertex[3];
      for (int ii=0; ii<3; ii++) {
        const int8_t *vert = vtkMarchingCubes_edges[edge[ii]];
        const float4 v0 = vertex[vert[0]];
        const float4 v1 = vertex[vert[1]];
        const float t = (isoValue - v0.w) / float(v1.w - v0.w);
        triVertex[ii] = (1.f-t)*v0+t*v1;
      }

      if (triVertex[1] == triVertex[0]) continue;
      if (triVertex[2] == triVertex[0]) continue;
      if (triVertex[1] == triVertex[2]) continue;

      const int triangleID = allocTriangle();
      if (triangleID >= 3*outputArraySize) continue;

      for (int j=0;j<3;j++) {
        (int &)triVertex[j].w = (4*triangleID+j);
        (float4&)outputArray[3*triangleID+j] = triVertex[j];
      }
    }
  }
};


#if EXPLICIT_MORTON
__global__ void buildMortonArray(Morton *const __restrict__ mortonArray,
                                 const vec3i coordOrigin,
                                 const Cell  *const __restrict__ cellArray, 
                                 const int numCells)
{
  const size_t threadID = threadIdx.x+size_t(blockDim.x)*blockIdx.x;
  if (threadID >= numCells) return;
  mortonArray[threadID].morton = mortonCode(cellArray[threadID].lower - coordOrigin);
  mortonArray[threadID].cell   = &cellArray[threadID];
}
#endif


__global__ void extractTriangles(
#if EXPLICIT_MORTON
                                 const Morton *const __restrict__ mortonArray,
#endif
                                 const vec3i coordOrigin,
                                 const Cell  *const __restrict__ cellArray, 
                                 const int numCells,
                                 const int maxLevel,
                                 const float isoValue,
                                 TriangleVertex *__restrict__ outVertex,
                                 const int outVertexSize,
                                 int *p_numGeneratedTriangles)
{
  AMR amr(
#if EXPLICIT_MORTON
          mortonArray,
#endif
          coordOrigin,cellArray,numCells,maxLevel);
  
  const size_t threadID = threadIdx.x+size_t(blockDim.x)*blockIdx.x;
                                  
  const int workID             = threadID / 8;
  if (workID >= numCells) return;
  const int directionID        = threadID % 8;
  const Cell currentCell = cellArray[workID];

  const int dz = (directionID & 4) ? 1 : -1;
  const int dy = (directionID & 2) ? 1 : -1;
  const int dx = (directionID & 1) ? 1 : -1;
  
  Cell corner[2][2][2];
  for (int iz=0;iz<2;iz++)
    for (int iy=0;iy<2;iy++)
      for (int ix=0;ix<2;ix++) {
        const vec3i delta = vec3i(dx*ix,dy*iy,dz*iz);
        const CellCoords cornerCoords = currentCell.neighbor(delta);
          
        if (!amr.findActual(corner[iz][iy][ix],cornerCoords)) 
          // corner does not exist - currentcell is on a boundary, and
          // this is not a dual cell
          return;

        if (corner[iz][iy][ix].level < currentCell.level) 
          // somebody else will generate this same cell from a finer
          // level...
          return;
        
        if (corner[iz][iy][ix].level == currentCell.level && corner[iz][iy][ix] < currentCell) 
          // this other cell will generate this dual cell...
          return;
      }

  IsoExtractor isoExtractor(isoValue,outVertex,outVertexSize,p_numGeneratedTriangles);
  isoExtractor.doMarchingCubesOn({dx==-1,dy==-1,dz==-1},corner);
}


__global__ void createVertexArray(int *p_atomicCounter,
                                  const TriangleVertex *const __restrict__ vertices,
                                  int numVertices,
                                  float3 *outVertexArray,
                                  int outVertexArraySize,
                                  int3 *outIndexArray
                                  )
{
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadID >= numVertices) return;

  const TriangleVertex vertex = vertices[threadID];
  if (threadID > 0 && vertex.position == vertices[threadID-1].position)
    // not unique...
    return;

  int vertexArrayID = atomicAdd(p_atomicCounter,1);
  if (vertexArrayID >= outVertexArraySize)
    return;

  outVertexArray[vertexArrayID] = (float3&)vertex.position;

  for (int i=threadID;i<numVertices && vertices[i].position==vertex.position;i++) {
    int triangleAndVertexID = vertices[i].triangleAndVertexID;
    int targetVertexID   = triangleAndVertexID % 4;
    int targetTriangleID = triangleAndVertexID / 4;
    int *triIndices = &outIndexArray[targetTriangleID].x;
    triIndices[targetVertexID] = vertexArrayID;
  }
}



int main(int ac, char **av)
{
  if (ac != 5)
    throw std::runtime_error("cudaAmrIso in.cells in.scalars isoValue outFile.obj");
  
  const std::string cellFileName   = av[1];
  const std::string scalarFileName = av[2];
  const float isoValue             = std::stof(av[3]);
  const std::string outFileName    = av[4];
  
  int maxLevel=0;
  thrust::host_vector<Cell>  h_cells;
  timer time;
  const vec3i coordOrigin = readInput(h_cells,maxLevel,cellFileName,scalarFileName);
  std::cout << "#input read, took " << time.elapsed() << "s" << std::endl;
  timer totalRuntime;
  
  // ------------------------------------------------------------------
  // step 1: upload cells to device, and sort
  // ------------------------------------------------------------------
  time.restart();
  thrust::device_vector<Cell> d_cells = h_cells;
  cudaDeviceSynchronize();
  std::cout << "#cells uploaded, took " << time.elapsed() << "s" << std::endl;

  time.restart();
#if EXPLICIT_MORTON
  thrust::device_vector<Morton> d_mortonArray(h_cells.size());
  {
    size_t numJobs   = h_cells.size();
    int blockSize = 512;
    int numBlocks = (numJobs+blockSize-1)/blockSize;
    buildMortonArray<<<numBlocks,blockSize>>>
      (thrust::raw_pointer_cast(d_mortonArray.data()),
       coordOrigin,
       thrust::raw_pointer_cast(d_cells.data()),
       d_cells.size());
  }
  cudaDeviceSynchronize();
  thrust::sort(d_mortonArray.begin(), d_mortonArray.end(), CompareMorton());
#else
  thrust::sort(d_cells.begin(), d_cells.end(), CompareByCoordsLowerOnly(coordOrigin));
#endif
  cudaDeviceSynchronize();
  std::cout << "#sorted, took " << time.elapsed() << "s" << std::endl;

  // ------------------------------------------------------------------
  // step 2a: run triangle extraction, count triangles
  // ------------------------------------------------------------------
  thrust::device_vector<int> d_atomicCounter(1);
  thrust::device_vector<TriangleVertex> d_triangleVertices(0);

  time.restart();
  {
    d_atomicCounter[0] = 0;
    size_t numJobs   = 8 * h_cells.size();
    int blockSize = 512;
    int numBlocks = (numJobs+blockSize-1)/blockSize;
    // dim3 grid(1024,divUp(numBlocks,1024));
    // std::cout << "launching with grid " << grid.x << " " << grid.y << " blocksize " << blockSize << std::endl;
    extractTriangles<<<numBlocks,blockSize>>>
      (
#if EXPLICIT_MORTON
       thrust::raw_pointer_cast(d_mortonArray.data()),
#endif
       coordOrigin,
       thrust::raw_pointer_cast(d_cells.data()),
       d_cells.size(),
       maxLevel,
       isoValue,
       thrust::raw_pointer_cast(d_triangleVertices.data()),d_triangleVertices.size(),
       thrust::raw_pointer_cast(d_atomicCounter.data())
       );
  }
  cudaDeviceSynchronize();
  std::cout << "#first pass for counting done, took " << time.elapsed() << "s" << std::endl;

  // ------------------------------------------------------------------
  // step 2b: allocate output array, and rerun, this time writing tris
  // ------------------------------------------------------------------
  int numTriangles = d_atomicCounter[0];
  std::cout << "expecting num triangles = " << numTriangles << std::endl;
  d_triangleVertices.resize(3*numTriangles);

  time.restart();
  {
    d_atomicCounter[0] = 0;
    size_t numJobs = 8 * h_cells.size();
    int blockSize  = 512;
    int numBlocks  = (numJobs+blockSize-1)/blockSize;
    extractTriangles<<<numBlocks,//dim3(1024,divUp(numBlocks,1024)),
      blockSize>>>
      (
#if EXPLICIT_MORTON
       thrust::raw_pointer_cast(d_mortonArray.data()),
#endif
       coordOrigin,
       thrust::raw_pointer_cast(d_cells.data()),
       d_cells.size(),
       maxLevel,
       isoValue,
       thrust::raw_pointer_cast(d_triangleVertices.data()),d_triangleVertices.size(),
       thrust::raw_pointer_cast(d_atomicCounter.data())
       );
  }
  cudaDeviceSynchronize();
  std::cout << "#first pass for actual generation, took " << time.elapsed() << "s" << std::endl;

#if 0
  std::ofstream dump("dump.obj");
  thrust::host_vector<TriangleVertex> h_triangleVertices = d_triangleVertices;
  for (int i=0;i<h_triangleVertices.size()/3;i++) {
    for (int j=0;j<3;j++) {
      vec3f v = h_triangleVertices[3*i+j].position;
      dump << "v " << v.x << " "<< v.y << " " << v.z << std::endl;
    }
    dump << "f -1 -2 -3" << std::endl;
  }
#endif

  
  // ==================================================================
  // step 3: create vertex array
  // ==================================================================
  
  // ------------------------------------------------------------------
  // step 3a: sort vertex array
  // ------------------------------------------------------------------
  time.restart();
  thrust::sort(d_triangleVertices.begin(), d_triangleVertices.end(), CompareVertices());
  cudaDeviceSynchronize();
  std::cout << "#sorted vertices, took " << time.elapsed() << "s" << std::endl;
  
  // ------------------------------------------------------------------
  // step 3b: count unique vertices
  // ------------------------------------------------------------------
  thrust::device_vector<float3> d_vertexArray(0);
  thrust::device_vector<int3>   d_indexArray(numTriangles);
  {
    d_atomicCounter[0] = 0;
    int numJobs   = 3*numTriangles;
    int blockSize = 512;
    int numBlocks = (numJobs+blockSize-1)/blockSize;
    createVertexArray<<<numBlocks,blockSize>>>
      (thrust::raw_pointer_cast(d_atomicCounter.data()),
       thrust::raw_pointer_cast(d_triangleVertices.data()),
       d_triangleVertices.size(),
       thrust::raw_pointer_cast(d_vertexArray.data()),
       d_vertexArray.size(),
       thrust::raw_pointer_cast(d_indexArray.data())
       );
  }
  cudaDeviceSynchronize();
  std::cout << "#counted unique vertices, took " << time.elapsed() << "s" << std::endl;

  int numVertices = d_atomicCounter[0];
  std::cout << "expecting num vertices " << numVertices << std::endl;

  // ------------------------------------------------------------------
  // step 3c: writing vertices
  // ------------------------------------------------------------------
  d_vertexArray.resize(numVertices);
  {
    d_atomicCounter[0] = 0;
    int numJobs   = 3*numTriangles;
    int blockSize = 512;
    int numBlocks = (numJobs+blockSize-1)/blockSize;
    createVertexArray<<<numBlocks,blockSize>>>
      (thrust::raw_pointer_cast(d_atomicCounter.data()),
       thrust::raw_pointer_cast(d_triangleVertices.data()),
       d_triangleVertices.size(),
       thrust::raw_pointer_cast(d_vertexArray.data()),
       d_vertexArray.size(),
       thrust::raw_pointer_cast(d_indexArray.data())
       );
  }
  cudaDeviceSynchronize();
  std::cout << "#generated vertex and index array, took " << time.elapsed() << "s" << std::endl;

  std::cout << "total runtime from upload to download : " << totalRuntime.elapsed()
            << "s" << std::endl;

  // ------------------------------------------------------------------
  // step 4: download and write out
  // ------------------------------------------------------------------
  std::ofstream out(outFileName,std::ios::binary);
out.precision(10);
  thrust::host_vector<float3> h_vertexArray = d_vertexArray;
  thrust::host_vector<int3>   h_indexArray = d_indexArray;
  out << "# iso-surface generated by cudaAmrIso tool:" << std::endl;
  out << "# num vertices " << h_vertexArray.size() << std::endl;
  out << "# num triangles " << h_indexArray.size() << std::endl;
  for (int i=0;i<h_vertexArray.size();i++)
    out << "v "
        << h_vertexArray[i].x << " "
        << h_vertexArray[i].y << " "
        << h_vertexArray[i].z << std::endl;
  for (int i=0;i<h_indexArray.size();i++)
    out << "f "
        << (h_indexArray[i].x+1) << " "
        << (h_indexArray[i].y+1) << " "
        << (h_indexArray[i].z+1) << std::endl;
  return 0;
}
